---
title: Testing Bias
mathjax: true
layout: post
categories: media
permalink: /collections/:title
---

<h1> I. Analyzing a Real-World Scenario for Sources of Bias </h1>

[*Categories of AI Bias*](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf) 
![Categories of AI Bias](https://antonellabasso.github.io/IMAGES/categories_AI_bias.png)

<h3> Task: </h3> 

Using the figure above, we identify potential sources of bias in the following example scenario and list them with justification, distinguishing whether these are likely to appear in the data collection/representation stage, the training/model building phase, or the inference/action stage of the data processing pipeline. 

<h3> Scenario: </h3> 

Health care systems use automated tools to determine whether a patient should be inducted into an intensive "care management system" to cure what ails them. The idea is to determine which patients can benefit the most from being admitted to the system because it is expensive, and not everyone can be enrolled. It is hard to know who will benefit the most, so the systems instead try to predict a proxy: who will incur the most health care costs. The idea is that if your health care costs are predicted to be large, you probably need intensive treatments to be cured, and so you're likely to benefit. The algorithm that makes this prediction uses the following training data set: input features for an individual are a list of insurance claim data from the previous year. Each such claim consists of

1. Demographic information (age, sex, but not race)
2. Type of insurance (private, medicare, medicaid, hmo, etc.)
3. Disease diagnosis (usually expressed in a standard code)
4. Procedures performed (X-rays, MRIs, surgeries, etc., each expressed with a specific code)
5. Medications
6. Costs

The goal is to predict the cost of care this year. Formally, this is a regression problem with a mix of numerical and categorical variables, and a numeric output. 

<h3> Response: </h3>  

This is my favorite example of racial discrimination in automated decision making. I commonly use it to explain (to others) how an algorithm may still favor a particular social group(s) in prediction, despite being "blind" to group membership in the data. Namely, this case exemplifies the ways in which *algorithmic neutrality* not only fails to prevent biased decisions, but leads individuals to believe that their outcomes are fair and accept injustice willingly. It’s no surprise, given our nation’s historical legacies of oppression and racism towards Black and Brown individuals, that communities of color continue to experience systemic injustices on disproportionate levels. Among these are both the degree and quality of healthcare provision, as reflected in insurance claims data, which proves that healthcare expenditures in the U.S. are disparately low for marginalized groups. Thus, any model that utilizes this data to predict future healthcare costs is bound to replicate the systemic biases embedded within it to generate an adverse *feedback loop* of discriminatory outcomes&mdash;that is, unless used in tandem with a processing strategy explicitly designed to remove them. In this way, healthcare expenditure becomes a *proxy* for race, making its use as a basis for allocating those in more urgent need of care an implicit method for discriminating against vulnerable populations and keeping marginalized individuals on the margins. 

**Selection/Sampling Bias:**
- *ecological fallacy*: "Occurs when an inference is made about an individual based on their membership within a group." The implicit assumption that disadvantaged racial groups receive the same form of care or benefits than their privileged counterparts could be interpreted as an ecological fallacy. 
- *detection bias*: "Systematic differences between groups in how outcomes are determined and may cause an over- or underestimation of the size of the effect." Detection bias in this case stems from the fact that marginalized and non-marginalized racial groups benefit from healthcare on disparate levels which in turn leads the system to underestimate the care needs of individuals of color.
- *measurement bias*: "Arises when features and labels are proxies for desired quantities, potentially leaving out important factors or introducing group or input-dependent noise that leads to differential performance." Similar to measurement bias, the computational  bias reflected in this scenario can be attributed to the fact that the desired quantity is itself a proxy for a factor we do not wish to have any influence over decisions. 

**Processing/Validation Bias:**
- *model selection bias*: "...Model selection bias also occurs when an explanatory variable has a weak relationship with the response variable." Evidently, there is a weak relationship between the predictor and outcome in this case, given that insurance claims data (itself contaminated with systemic biases), rather than being able to distinguish those with more critical health conditions who may largely benefit from care management, reflects which (groups of) individuals have been previously prioritized by the system and are hence responsible for the majority of health care expenses regardless of actual health status.
- *survivorship bias*: "Tendency for people to focus on the items, observations, or people that 'survive' or make it past a selection process, while overlooking those that did not." Decision-makers using the model discussed in this example are likely to display survivorship bias (at least to some extent), as their presumed goal is to select individuals with high projected care costs for admission into the care management system, while overlooking those with low cost predictions.

**Use & Interpretation Bias:**
- *feedback loop bias*: "Effects that may occur when an algorithm learns from user behavior and feeds that behavior back into the model." If the overall intent is to use this model to predict future healthcare care costs beyond the following year, feedback loop bias will be an inevitable consequence due to the perpetual use of biased data for prediction&mdash;in accordance with the renowned saying “bias in, bias out”.  


<h1> II.  Experimenting With the Ways in Which Design Choices Affect Fairness </h1>
<h2> i. Data \& Preprocessing </h2>
