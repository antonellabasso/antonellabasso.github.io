---
code: PHP2560
title: Exposing Racial Disparities in Police Violence 
mathjax: true
layout: page
categories: media
permalink: /collections/:title
---

<!--
# POTENTIAL TITLES:
# Demonstrating Racial Injustice in the US Police/Policing System
# Demystifying Racial Gap/DiscrepancyInjustice in US Police Shootings
# The Reality of Racial Injustice in the US Police/Policing System
# Exposing Racial Injustice in US Policing 
# Exposing the Truth About Police Violence in the US
# Exposing the Truth About Police Violence in the US
# Exposing Racial Disparities in (US) Policing\Police Violence
-->

<center><b> Research Questions: </b></center> 

- To what extent does one’s race impact their likelihood of being shot or killed by police in the US?
- How does location (at the state-level) contribute to one’s chances of falling victim to police brutality?
- Is there a potential relationship between a state’s political climate and the level of racial inequality evidenced in police shootings?
- What inferences can we make about the US criminal justice system and the role that race plays in police violence?

> The police shooting data used for this analysis, when considered in isolation, has the potential to create misleading information that undermines our core research questions. Therefore, our objective is to employ appropriate programming and data science tools to contextualize and transform the data, enabling us to obtain more reliable insights for addressing them effectively. 

<h1> I. Introduction </h1>

Unwarranted deaths at the hands of law enforcement, like those of George Floyd and Breonna Taylor, have recently become a more pressing issue in the US. Many such extreme instances of police brutality have become realized as a genuine threat to marginalized communities and given rise to social movements, like “Black Lives Matter”, as a way to combat structural racism and catalyze more drastic police reforms. Though the presence of racial disparities in police violence is not a new issue in the US, evidence like that which we provide in this work is vital to both gauging the true magnitude of the problem and encouraging positive change. Moreover, while our data are publicly available, it is important to emphasize that a lack of adequate visualizations and representations of such information often contributes to the gap in public awareness of the extent to which marginalized communities and individuals are harmed by our institutions. To that end, we hope to shed some light on the racial discrepancies evidenced by documented police shootings in recent years, primarily focusing on Black and white racial groups and their varying degrees of susceptibility to police violence across the country.

<h1> II. Data </h1>

Our analysis is based on data obtained from three primary sources:

1. [*US Police Shootings*](https://www.kaggle.com/datasets/ahsen1330/us-police-shootings)
- This *Kaggle* dataset provides basic information about individuals shot or killed by police in the US between 2015 and 2020, including their name, age, gender, race, and details about the incidents.
- Moreover, it includes factors such as the location, date, shooting circumstances, whether the person was armed, whether they exhibited signs of mental illness, and whether the incident was recorded.

2. [*Bridged-Race Population Estimates 1990-2020 Results*](https://wonder.cdc.gov/controller/datarequest/D178)
- This *CDC WONDER* dataset offers estimated population figures for different racial groups in each state from 1990 to 2020.
- Its purpose in our analysis is to provide contextual information about racial populations in each state.

3. [*List of Blue States and Red States*](https://www.gkgigs.com/list-of-blue-states-and-red-states/)
- This *GkGigs* dataset provides a list of states along with their dominant political party affiliation (blue/Democrat or red/Republican).
- We use this information to categorize states based on their political climate.

Upon initial examination, the first dataset alone does not provide a comprehensive understanding of racial disparities in police shootings. A simple observation of the total number of Black individuals compared to white individuals shot or killed by law enforcement over the past five years reveals a significantly larger number of white individuals, which contradicts our expectations. However, relying solely on raw numbers of shootings is insufficient for making valid inferences about targeted racial groups. To mitigate biased results, we incorporate the second dataset, which provides state-level racial population estimates. This allows us to analyze the number of people shot or killed in each racial group relative to the population composition of each racial category in a given area. By examining the data in terms of proportions, we aim to make unbiased comparisons that expose the true varying degrees of racial inequality in police brutality across the US. Lastly, we incorporate the third dataset into our analysis to explore potential relationships between a state's political climate and racial disparities in incidents of police violence.

To substantiate and conduct our analysis, we appeal to three additional sources:

1. [*Quick Facts*](https://www.census.gov/quickfacts/fact/table/US/PST045219)
- This *US Census* data provides 2019 racial population estimates in terms of percentages.
– It will be used to evaluate proportions of shootings based on race as they relate to proportions of racial subgroups that make up the total US population.

2. [*Most Republican States*](https://worldpopulationreview.com/state-rankings/most-republican-states) and [*Most Racist States*](https://worldpopulationreview.com/state-rankings/most-racist-states)
- These 2021 articles from the *World Population Review* are used to identify the most "Republican" and "racist" states in the country based on the *Cook Partisan Voting Index (CPVI)*, which “measures how strongly a state leans Republican or Democratic compared to the entire nation”, as well as instances of hate crimes and hate speech that “can also be used to determine where racism is most prevalent”.'
- They will be used to contextualize our findings and be referenced throughout.

3. [*Mapping the US*](https://cran.r-project.org/web/packages/usmap/vignettes/mapping.html)
- This article offers packages and instructions for creating US map visualizations of data in R.
– It will be used to guide our state mappings of racial discrepancies in populations proportions of police shootings.

<h1> III. Preprocessing </h1>

According to the US Census Bureau, Black and white folks made up roughly 13% and 60% of the total US population as of 2019, respectively. Thus, since, 50% of reported shootings in the last five years were white according to our data, it follows that the reported 27% of shootings corresponding to the Black population is more than double what we'd expect if there were no racial gap in police violence between Black and white Americans. However, fairness for all groups would require that police shootings be consistent with their given US populations, eliminating race as a potential risk factor for police brutality.  

<!-- That is, for police shootings to have been consistent with American Black and white populations, Black shootings would've had to make up 11% of all shootings&mdash;approximately 5/6 of the corresponding US racial population. Yet, based on percentages alone, while the proportion of Black folks shot by police is almost double the proportion of Black folks in the US, the proportion of White folks shot by police is about 16% smaller than their US population. -->

To compare the number of Black and white victims of police violence in the US, we first transform our data to account for differences in race populations across states. Specifically, we assign a weight \\(w_{ij}\\) to each state's documented number of race-specific shootings \\(n_{ij}\\) that is proportional to its corresponding racial population. Letting \\(p_{ij}\\) denote the unscaled population parameter, we compute the following statistic \\(x_{ij}\\) for each state \\(j\\) and racial category \\(i\in\{b, w\}\\):

\\[x_{ij} = n_{ij}w_{ij}, \text{ where } w_{ij} = \frac{1}{p_{ij}}\times 10^{-6}.\\]

We note that since \\(p_{ij}\\) is unscaled, we convert populations to millions, scaling \\(p_{ij}\\) by 1/1,000,000 for simplicity and visualization purposes. The figure below shows the change in rates of race-specific shootings when accounting for the corresponding racial populations across states. 

<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig1.png" width="70%" alt=""></center>

Evidently, despite the fact that the actual number of white victims is significantly larger than that of Black victims nation-wide, the number of individuals shot per million in each state is actually far greater with regards to the Black population. This finding confirms the prevalence of racial disparity in police violence across the US, suggesting that Black Americans are in fact at greater risk of being victimized by police compared to whites.

<h1> IV. Analysis </h1>

To observe the discrepancies in the weighted proportions of Black and White folks shot by police in each state (bar graph on the right-hand side above), we refer to the following two metrics:

1. Differences: weighted Black shootings − weighted white shootings \\[d_j = x_{bj}-x_{wj}\\]
- Raw differences in weighted proportions shed light on these discrepancies as it pertains to how many shootings there are in each state.

2. Ratios: weighted Black shootings/weighted white shootings \\[r_j = x_{bj}/x_{wj}\\]
- Weighted proportion ratios tell us how much larger proportions of Black shootings are than in comparison to the proportions of White shootings. These shed light on the relative differences between shootings irrespective of the sheer number of people shot in each state. 

Both metrics provide a way to analyze racial discrepancies in police shootings across the US by giving us an idea of how much police violence is present in each state and the extent to which one group is affected purely in contrast to the other. For each, we construct a bar graph, violin plot, and US map, to better visualize these discrepancies as they relate to the dominant political party in each state

<!--
****** NOTE: Differences allow us to visualize discrepancies in terms of the sheer level of police violence in given states, while ratios provide a sense for how large this gap is irrespective of how many shootings occurred. For example UT exhibits significantly more police violence than RI, but the actual disparity in violence happens to be larger in RI.

graphs 
explanations

<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig2.png" width="70%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig3.png" width="70%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig4.png" width="70%" alt=""></center>

<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig5.png" width="70%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig6.png" width="70%" alt=""></center>
<center><img src="https://antonellabasso.github.io/IMAGES/PS_fig7.png" width="70%" alt=""></center>
-->

<h1> V. Conclusion </h1>


<h1> Code Appendix </h1>


