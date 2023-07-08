---
code: PHP2530
title: Estimating Mortgage Lending Discrimination with Hierarchical Bayesian Modeling
mathjax: true
layout: page
categories: media
permalink: /collections/:title
---

<h1> I. Introduction </h1>

Despite explicit laws to protect against lending discrimination&mdash;such as the Fair Housing Act (FHA), the Equal Credit Opportunity Act (ECOA),
and the Community Reinvestment Act (CRA) (source 1)&mdash;Black families in the U.S. continue to face alarming levels of injustice in loan lending scenarios, 
which exacerbates racial inequities like those of homeownership. Specifically, when lenders choose to ignore an individual’s race in making decisions, 
instead using income and credit score as a basis for loan approval, they ignore the prominent role that systemic racism plays in economic disparity and
implicitly contribute to widening the racial wealth gap (source 2). In an attempt to bring attention to our nation’s racially disproportionate and systemic 
denial of loans that harms American citizens and prevents wealth building in communities of color, this analysis aims to utilize hierarchical 
Bayesian modeling to gauge the degree of discrimination in loan lending decisions across all 50 states. In addition to discussing the data used 
for estimating our parameters of interest, we outline our chosen model as well as its implementation using an MCMC algorithm, and discuss our results. 
Ultimately, we hope to show how Bayesian methods may be used to infer about our institutional decision making practices and not only discover 
discrimination but estimate it in a way that may influence more adequate and substantive policies.

<h1> II. Data </h1>

The data used for this analysis, obtained from the Consumer Financial Protection Bureau, spans thousands of 2017 loan application records from 
financial institutions across the U.S. that have been made publicly available under the Home Mortgage Disclosure Act (HMDA) (source 3).
The records gathered reflect individual applications wherein one of the following eight potential actions were taken (source 4).

1. Loan originated
2. Application approved but not accepted
3. Application denied by financial institution
4. Application withdrawn by applicant
5. File closed for incompleteness
6. Loan purchased by the institution
7. Preapproval request denied by financial institution
8. Preapproval request approved but not accepted (optional reporting)

Given our focus on individuals whose applications were either approved or denied, we filter out observations involving actions 1, 4, 5, or 8, 
and categorize those pertaining to actions 2 or 6 and actions 3 or 7 as approved and denied cases, respectively. Moreover, as we are interested
in lending differences between Black and white applicants specifically, we also exclude individuals from other racial groups as well as those 
who did not specify group membership from our analysis. Having compiled and preprocessed the data to reflect the proportions of Black and white 
applicants that were approved for a loan, we are left with two observations for each of the 50 states&mdash;one for each racial group&mdash;totaling 
100 observations for the 2017 year.

<h1> III. Model </h1>

Under a causally-motivated understanding of discrimination, we seek to estimate the effect of racial group membership (source 5) on one’s chances of
being approved for a loan within a given state. Specifically, we wish to determine whether loan approval rates vary systematically as a function of 
racial group membership across states. Thus, given these aims and the nature of our data, we implement a hierarchical binomial model for clustered data
similar to that adopted by Jackman in an analysis of a cluster-randomized experiment (source 6) for “assessing the effects of a voter mobilization 
treatment in the 2004 U.S. presidential election” (source 7). For our purposes, we assume observations are clustered by state and that each cluster 
comprises two observations, namely, 2017 count data for each racial group of interest.

Focusing on the success rate to guide our estimation of discriminatory effects in loan approval across states, we implement the following hierarchical binomial model. For \\(i = 1,2,...,100\\), let \\(n_i\\) be the total number of applicants and \\(r_i\\) be the number of approvals such that \\(p_i = r_i / n_i\\) gives the \\(i\\)th observation’s approval rate. Moreover, let \\(S_i \in \\{1,2,...,50\\}\\) be the state indicator and \\(W_i \in \\{0, 1\\}\\) be the racial group indicator for observation \\(i\\), such that \\(S_i = 1\\) and \\(W_i = 1\\) jointly denote the white racial group observation for the first state alphabetically&mdash;Alabama (AL). Assuming normally distributed effects \\(\alpha_{S_i}\\) and \\(\delta_{S_i}\\) with normal mean parameter densities \\(\mu_{\alpha}\\) and \\(\mu_{\delta}\\), we have


\\[ r_i \sim \text{Bin}\(n_i,p_i\), \\] 
\\[ \text{logit}\(p_i\) = \alpha_{S_i} + \delta_{S_i}W_i, \\] 
\\[ \alpha_{S_i} \sim \text{N}\(\mu_{\alpha},\sigma_{\alpha}^2\), \\] 
\\[ \delta_{S_i} \sim \text{N}\(\mu_{\delta},\sigma_{\delta}^2\), \\] 
\\[ \mu_{\alpha} \sim \text{N}\(0,2^2\), \\] 
\\[ \mu_{\delta} \sim \text{N}\(0,2^2\), \\] 
\\[ \sigma_{\alpha} \sim \text{Unif}\(0,2\), \\] 
\\[ \sigma_{\delta} \sim \text{Unif}\(0,2\), \\]

where \\(\sigma_{\alpha}^2\\) and \\(\sigma_{\delta}^2\\) are the hierarchical variance parameters. We assume noninformative uniform hyperpriors on \\(\sigma_{\alpha}\\) and \\(\sigma_{\delta}\\), which according to Gelman et al. should not result in an improper posterior density due to the large number of clusters in the data, and posed no problems for Jackman despite nonconjugacy (source 7, 8).

Given our lack of information to distinguish between the \\(n_i\\) applicants for each observation, we take lending decisions to be exchangeable Bernoulli trials, each given a success rate of \\(p_i\\), and like Jackman, implement a binomial model for loan approvals \\(r_i\\) at the observation level, assuming these are independent conditional on \\(p_i\\) (source 7). Additionally, aside from state name and racial group, the fact that we lack information to differentiate between state observations allows us to both assumme exchangeability for \\(p_i\\) conditional on \\(S_i\\) and \\(W_i\\), as well as model state-level effects exchangeably with a common prior distribution as shown above. 

As we are unsure about the degree of discrimination present on both state and national levels purely from the records collected for the 2017 year&mdash;themselves subject to preprocessing error&mdash;we remain ambiguous about discriminatory effects so as to allow the data to play a larger role in informing us about state-level variations in racial discrepancies of loan approval rates. For this reason, we assign normal priors to the baseline levels \\(\alpha_{S_i}\\) and discriminatory effects \\(\delta_{S_i}\\), supposing that their expected values are also normally distributed with mean 0. Specifically, since the assumption that \\(\text{E}\[\delta_{S_i}\] = 0\\) is rather optimistic regarding discrimination, we instead posit that the ground truth must be a compromise between the observed data, our knowledge of existing systemic racism, and what we ought to expect given the aforementioned policy interventions. As such, we hold that our assumptions that \\(\text{E}\[\delta_{S_i}\] = \mu_{\delta}\\) and \\(\text{E}\[\mu_{\delta}\] = 0\\), with given variances, better capture our uncertainty about the extent of discrimination in these contexts.

<h1> IV. Implementation </h1>

Similar to Jackman’s proposed method, we implement our hierarchical model within a single-chain MCMC algorithm to generate posterior estimates of the state-level effects and corresponding hyperparameters. Using the `rjags` package for our implementation, we give all parameters initial values of 0 and generate 50,000 samples after a 1,000 iteration burn-in, subsequently saving every fifth iteration for inference.

Figures 1-4 demonstrate that our chain successfully converges to the target posterior densities rather quickly, seeing as parameter values remain stabilized after burn-in. Given the large number of effect parameters in our model, we only provide trace plots for the hyperparameters as they reflect similar states of achieved convergence. Table 1 below provides their corresponding posterior estimates as means of the 10,000 saved samples and 95\% highest density region (HDR) credible intervals. Notably, we see little posterior uncertainty in \\(\sigma_{\delta}\\) despite the much larger 95\% HDR interval for \\(\mu_{\delta}\\). However, with a posterior mean of approximately 0.72 and a credible interval that still does not contain 0, it follows that our model supports the intuition that positive state-level discriminatory effects in fact exist across states on average.

<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig1.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig2.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig3.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig4.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig5.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig6.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_fig7.png" width="80%" alt=""></center>

<center><img src="https://antonellabasso.github.io/IMAGES/LD_tbl1.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_tbl2.png" width="80%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/LD_tbl3.png" width="80%" alt=""></center>

