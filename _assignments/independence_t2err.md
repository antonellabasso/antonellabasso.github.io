---
code: PHP2515
title: Independence & Type I Errors 
mathjax: true
layout: page
categories: media
permalink: /collections/:title
---

#### Antonella Basso, Anusha Kumar and Breanna Richards

<h1> I. Introduction </h1>

Independence is among the most critical assumptions in regression, modeling, and hypothesis testing. In statistics, two events are considered independent iff the occurrence of one has no effect on the probability of the other. Ensuring that sampled data are independent protects us from both making biased population estimates and falling victim to numerous *type I errors* (or “false positives”) (source 1). That is, false rejections of the null hypothesis when it is in fact true, which in many cases may have dire consequences. To illustrate this phenomenon, consider the following example:

> Suppose we are interested in collecting the height measurements of boys and girls in a 4th grade class for comparative estimates of 4th grade students in the US. Our goal is to test whether the mean heights of boys and girls are equal (null hypothesis) against the *alternative hypothesis* that they are not equal. Suppose further that the null hypothesis is true, and that within the class we’ve chosen to randomly sample, there are three brothers and a pair of identical twin sisters shorter than most of their peers. In conducting the corresponding hypothesis test, we implicitly assume that our data are independent. However, the fact that some of our subjects are biologically related indicates that our data are dependent, as genetics tend to produce similar physical traits (such as height) among kin. This dependence within our data consequently increases our likelihood of wrongfully rejecting the null hypothesis as some of our observations provide additional information about others. That is, our chances of obtaining a *p-value* less than 0.05 (significance level, \\(\alpha\\)) is now in fact greater than 5%, making us more prone to committing a type I error.

Scenarios such as this one are rather frequent in the real world and exceedingly difficult to avoid, making the risk of false positives much more prevalent. In this paper we delve further into what happens when the assumption of independence is violated in the context of hypothesis testing. Specifically, we simulate dependent data to explore its effects on false positives and generate plots to illustrate this relationship.

<h1> II. Simulation Study </h1>

Consider the following data generation process for outcomes \\(y_i\\), for subject \\(i = 1, 2, ..., n\\): 

$$ y_1 \sim N\(0,1\), $$

$$ y_i \mid y_{i−1} \sim N\(\rho \cdot y_{i−1}, 1\) &\text{for } i=2,...,n. $$

As generated, each observation \\(y_i\\) and corresponding distribution depends on the former, \\(y_{i−1}\\) for \\(\rho \neq \\) 0, giving us dependent sample data. However, note that if \\(\rho = 0\\), then \\(y_1 \sim N\(0, 1\)\\), just as \\(y_i \mid y_{i−1} \sim N\(0, 1\)\\) for \\(i = 2, ..., n\\), which not only indicates that the data are independent, but also identically distributed. That is, each observation comes from the same distribution, and thus, has the same expectation, \\(\mu\\). If, on the other hand, \\(\rho\\), 0 and the data are dependent, these expectations are no longer constant, meaning that, almost surely, they are not identically distributed. Hence, violating yet another key assumption in hypothesis testing. Together, these assumptions of independence and identical distribution form what is known as *independent and identically distributed* (or “iid”), which, in practical terms, means that no two observations in any one data set are related and all are taken from the same probability distribution. Now, suppose we observe data \\(\lbrace y_i \rbrace_{i=1:n}\\), generated with the DGP outlined above. Our goal is to test the null hypothesis that the mean is zero (\\(H_0 : \mu = 0\\)), against the alternative hypothesis that the mean is not zero \\((H_1 : \mu \neq 0)\\). Note that for this study, we hold that the null hypothesis is true.

<h1> III. Independent Data: \\(\rho = 0\\) </h1>

We begin by simulating independent data in R via the DGP mentioned above, setting \\(\rho = 0\\). Keep in mind that, as mentioned, this means that our data is (\\(y_i \sim N\(0, 1\)\\), for \\(i = 1, ..., n\\)) and hence, the expected value of \\(y\\) remains zero, namely \\(E[Y] = \mu = 0\\). Specifically, we generate 1,000 data sets of sample size \\(n = 1,000\\), and for each, we run a t-test with unknown variance, recording the decision to accept or reject the null hypothesis based on a type I error rate of \\(\alpha = 0.05\\). That is, for each data set, we generate a corresponding p-value and compute the proportion, which we denote \\(\hat{\alpha}\\)), of those which exceed our significance level of 0.05 out of the total 1,000 decisions made. As expected, doing so yielded \\(\hat{\alpha} = 0.048 ≈ 0.05\\), indicating that nearly 5% of the 1,000 t-tests conducted incorrectly rejected the null hypothesis, assumming that \\(H_0\\) is true. And, given that our data are iid, obtaining a type I error rate close to 0.05 demonstrates that the t-test assumptions are met, thus producing t-statistics that follow a t-distribution (more on this later).

<h1> VI. Dependent Data: \\(\rho \in \lbrace 0, 0.1, 0.2, ..., 0.9 \rbrace\\) </h1>

Seeing how independent data yields results that conform to the t-test assumptions, we now proceed by increasing the value of ρ to observe the influence of dependent data on \\(\hat{\alpha}\\). As before, we simulate 1,000 data sets with \\(n = 1,000\\), this time with \\(\rho \in \{0, 0.1, 0.2, ..., 0.9\}\\), resulting in a total of ten simulations, and hence, ten \\(\hat{\alpha}\\) values, one for each value of \\(\rho\\). The graph below illustrates the change in \\(\hat{\alpha}\\) for each incremental change in \\(\rho\\).

<img src="https://antonellabasso.github.io/IMAGES/independence_t2err_img1.png" width="50%" alt="">

