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


