---
code: CSCI1951Z
title: Testing Bias - Part B
mathjax: true
layout: page
categories: media
permalink: /collections/:title
---

<h1> Experimenting With the Ways in Which Design Choices Affect Fairness </h1>

<h3> Task: </h3>

We will be working with a dataset from [Folktables](https://github.com/zykls/folktables), a Python package that provides access to datasets derived from the US Census. The data provided by this library is sourced from the [American Community Survey](https://www.census.gov/programs-surveys/acs), a demographics survey program that gathers information about individuals' educational attainment, income, employment, demographic information, etc. First, we download and process a dataset from California 2018. 

<!-- This is particularly useful for measuring fairness in machine learning models as we will later show. -->

Specifically, we focus the `ACSPublicCoverage` binary classification task, as defined by Folktables, which seeks to predict whether individuals have public coverage based on a given a set of features (i.e., age, sex, race, and a range of disabilities). Given that one of the features used in prediction is the __race__ *sensitive attribute*, which should have no correlation with the outcome, our goal is to utilize different techniques to make our model more fair in its prediction.

<!-- Let us begin by observing the dataset. You might find the [ACS PUMS documentation](https://www.census.gov/programs-surveys/acs/microdata/documentation.html) helpful when interpreting the feature codings. -->

<h3> A. Data & Preprocessing </h3> 

We first begin by observing the data.

{% highlight python %}

from folktables import ACSDataSource, ACSPublicCoverage

# import data source form ACS
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
prediction_task = ACSPublicCoverage

# gather columns defined in the ACSPublicCoverage prediction task
features, label, group = prediction_task.df_to_numpy(acs_data)

# aggregate features and target variable into the same dataframe
acs_public_coverage_data = acs_data[prediction_task.features + [prediction_task.target]]

## data summary 
print(acs_public_coverage_data.head())
print("features used for predictions: "   + str(prediction_task.features))
print("group membership variable: "       + str(prediction_task.group))
print("the target variable of interest: " + str(prediction_task.target))

{% endhighlight %}

```
    AGEP  SCHL  MAR  SEX  DIS  ESP  CIT  MIG  MIL  ANC  NATIVITY  DEAR  DEYE  \
0    30  14.0    1    1    2  NaN    1  3.0  4.0    1         1     2     2   
1    18  14.0    5    2    2  NaN    1  1.0  4.0    1         1     2     2   
2    69  17.0    1    1    1  NaN    1  1.0  2.0    2         1     2     2   
3    25   1.0    5    1    1  NaN    1  1.0  4.0    1         1     1     2   
4    31  18.0    5    2    2  NaN    1  1.0  4.0    1         1     2     2   

   DREM    PINCP  ESR  ST  FER  RAC1P  PUBCOV  
0   2.0  48500.0  6.0   6  NaN      8       1  
1   2.0      0.0  6.0   6  2.0      1       2  
2   2.0  13100.0  6.0   6  NaN      9       1  
3   1.0      0.0  6.0   6  NaN      1       1  
4   2.0      0.0  6.0   6  2.0      1       1

features used for predictions: ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
group membership variable: RAC1P
the target variable of interest: PUBCOV
```

{% highlight python %}

import matplotlib.pyplot as plt

## data visualization 
race_coding = [
    'White alone',                # white alone
    'Black or African American',  # black or african american alone
    'American Indian alone',      # american indian alone
    'Alaska Native alone',        # alaska native alone
    'American Indian and ...',    # american indian and alaska native tribes specified; 
                                  # or american indian and alaska native not specified, and no other races
    'Asian alone',                # asian alone
    'Native Hawaiian',            # native hawaiian and other pacific islander alone
    'Some Other Race alone',      # some other race alone
    'Two or More Races']          # two or more races

# bar graph by group membership
(acs_public_coverage_data
 .groupby([prediction_task.group])
 .size()
 .set_axis(race_coding)
 .plot(kind='bar', rot=-90))

{% endhighlight %}

<img src="https://antonellabasso.github.io/IMAGES/CSCI1951_HW1_img1.png" width="50%" alt="">

{% highlight python %}

health_coverage_coding = ['With public health coverage', 'Without public health coverage']

# bar graph by target of interest
(acs_public_coverage_data
 .groupby([prediction_task.target])
 .size()
 .set_axis(health_coverage_coding)
 .plot(kind='bar', rot=0))

{% endhighlight %}

<img src="https://antonellabasso.github.io/IMAGES/CSCI1951_HW1_img2.png" width="50%" alt=""> <br>

Both bar graphs reflect that neither the race class nor the target label are balanced (i.e., evently distributed), displaying significant differences within them. 

<h3> B. Training </h3> 

<!-- You might want to take a moment to think about how this imbalance in distribution might affect the model's performance and fairness. -->

We now define our training function with logistic regression, using the [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) and [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) functions to initialize the model.

{% highlight python %}

def train(X_train, y_train):
  """
  Defines and trains a logistic regression model on the training data.

  Args:
    X_train (np.ndarray): Training inputs.
    y_train (np.ndarray): Training labels.                 

  Returns:
    sklearn.pipeline.Pipeline: trained model
  """
  # TODO: train model

  LR_pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
  return LR_pipeline

{% endhighlight %}

<h3> C. Evaluation </h3> 

Implementing the three fairness measurements discussed in [*Fairness and Machine Learning: Limitations and Opportunities*](https://fairmlbook.org/) and defined below&mdash;*independence*, *separation* and *sufficiency*, we can evalute how fair our model is in predicting status of public health coverage.

Let \\(Y\\) be the binary target variable, \\(\hat{Y}\\) be the model's predicted outcome and \\(A\\) be some sensitive attiribute.

- Random variables \\(\(A, \hat{Y}\)\\) satisty *independence*, i.e., \\(A \perp \hat{Y}\\), if 

$$ \frac{P\{ \hat{Y} = 1\ | A = a\}}{P\{ \hat{Y} = 1\ | A = b\}} = 1. $$ 

- Random variables \\(\(A, Y, \hat{Y}\)\\) satisty *separation*, i.e., \\(A \perp \hat{Y} \mid Y\\), if for groups in \\(A\\), say \\(a\\) and \\(a'\\), 
  
$$ \frac{P\{ \hat {Y} | Y = 1, A = a\}} {P \{\hat{Y} | Y = 1, A = a'\}} = 1; $$

$$ \frac{P\{ \hat {Y} | Y = 0, A = a\}} {P \{\hat{Y} | Y = 0, A = a'\}} = 1. $$  

- Random variables \\(\(A, Y, \hat{Y}\)\\) satisty *sufficiency*, i.e., \\(A \perp Y \mid \hat{Y}\\), iff for all values \\(\hat{y}\\) of \\(\hat{Y}\\) and groups in \\(A\\), say \\(a\\) and \\(a'\\),
  
$$ \frac{P\{Y = 1 | \hat{Y} = \hat{y}, A = a\}}{P\{Y = 1 | \hat{Y} = \hat{y}, A = a'\}} = 1. $$ 

*(NOTE: Separation is the same as equalizing true positive and false positive rates accross groups.)*

{% highlight python %}

from operator import index

# independence 
def independence(y_hat, group):
  """
  Computes an independence metric between two groups.

  Args:
    y_hat (np.ndarray): Classifier predictions.
    group (np.ndarray): Array of indices corresponding to group membership.
      For our purposes, we focus on comparing groups 1 and 2. These correspond 
      to the 'White alone' and 'Black or African American' groups.           

  Returns:
    float: independence measure
  """
  # TODO: compute measure

  idx1 = np.where(group == 1)[0]
  idx2 = np.where(group == 2)[0]

  P1 = sum(y_hat[(idx1),])/len(y_hat[(idx1),])
  P2 = sum(y_hat[(idx2),])/len(y_hat[(idx2),])

  indep = P2/P1
  return indep

# separation
def separation(y_hat, y_true, group):
  """
  Computes a separation metric between two specific groups.

  Args:
    y_hat  (np.ndarray): Classifier predictions.
    y_true (np.ndarray): Data labels.
    group  (np.ndarray): Array of indices corresponding to group membership.
      For our purposes, we focus on comparing groups 1 and 2. These correspond 
      to the 'White alone' and 'Black or African American' groups. 

  Returns:
    float: separation true positive
    float: separation false positive
  """
  # TODO: compute measure

  idx1_1 = np.intersect1d(np.where(y_true == 1)[0], np.where(group == 1)[0])
  idx1_2 = np.intersect1d(np.where(y_true == 1)[0], np.where(group == 2)[0])
  idx0_1 = np.intersect1d(np.where(y_true == 0)[0], np.where(group == 1)[0])
  idx0_2 = np.intersect1d(np.where(y_true == 0)[0], np.where(group == 2)[0])

  P1_1 = sum(y_hat[(idx1_1),])/len(y_hat[(idx1_1),])
  P1_2 = sum(y_hat[(idx1_2),])/len(y_hat[(idx1_2),])

  P0_1 = sum(y_hat[(idx0_1),])/len(y_hat[(idx0_1),])
  P0_2 = sum(y_hat[(idx0_2),])/len(y_hat[(idx0_2),])

  TP = P1_2/P1_1
  FP = P0_2/P0_1
  return TP, FP

# sufficiency
def sufficiency(y_hat, y_true, group):
  """
  Computes a sufficiency metric between two specific groups.

  Args:
    y_hat  (np.ndarray): Classifier predictions.
    y_true (np.ndarray): Data labels.
    group  (np.ndarray): Array of indices corresponding to group membership.
      For our purposes, we focus on comparing groups 1 and 2. These correspond 
      to the 'White alone' and 'Black or African American' groups. 

  Returns:
    float: sufficiency metric
  """
  # TODO: compute metric

  idx1_1 = np.intersect1d(np.where(y_hat == 1)[0], np.where(group == 1)[0])
  idx1_2 = np.intersect1d(np.where(y_hat == 1)[0], np.where(group == 2)[0])

  P1_1 = sum(y_true[(idx1_1),])/len(y_true[(idx1_1),])
  P1_2 = sum(y_true[(idx1_2),])/len(y_true[(idx1_2),])

  suff = P1_2/P1_1
  return suff

# evaluation function
def eval(yhat, y_test, group_test, model_title):
  print("Results from the " + model_title + " model: ")
  print("the indepence of prediction and group is ", independence(yhat, group_test))
  true_s, false_s = separation(yhat, y_test, group_test)
  print("the true positive separation is ", true_s)
  print("the false positive separation is ", false_s)
  print("the sufficiency of the prediction and the group is", sufficiency(yhat, y_test, group_test))

y_hat_example = np.asarray([True, True, False, False, True, False, False, False, True, True])
y_test_example = np.asarray([True, True, True,  False, False, False, False, True, False, True])
group_test_example = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

eval(y_hat_example, y_test_example, group_test_example, "unit-test")

{% endhighlight %}

```
Results from the unit-test model: 
the indepence of prediction and group is  0.6666666666666667
the true positive separation is  0.75
the false positive separation is  0.6666666666666666
the sufficiency of the prediction and the group is 0.75
```

<h3> D. The Full Workflow </h3> 

Finally, we connect the whole pipeline with training and see how fair our model is. We will:

1. Do an 80-20 `train_test_split` on the dataset with `random_state = 0`.
2. Train our linear regression model.
3. Use the trained model to make predictions on the test dataset.
4. Evaluate the model with fairness measurements. 

{% highlight python %}

# split the data into training and testing sets
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features, label, group, test_size=0.2, random_state=0)
model = train(X_train, y_train)
yhat = model.predict(X_test)

eval(yhat, y_test, group_test, "baseline")

{% endhighlight %}

```
Results from the baseline model: 
the indepence of prediction and group is  1.6135056165484982
the true positive separation is  1.3052337292915481
the false positive separation is  1.2975609756097561
the sufficiency of the prediction and the group is 1.196133899104196
```

<h1> II. Resampling </h1>

<h3> Task: </h3>



<h1> III. Cost-Sensitive Learning </h1>

<h3> Task: </h3>


