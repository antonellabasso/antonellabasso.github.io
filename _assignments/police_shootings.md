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

1. Kaggle - "US Police Shootings"
- This dataset provides basic information about individuals shot or killed by police in the US between 2015 and 2020, including their name, age, gender, race, and details about the incidents.
- Moreover, it includes factors such as the location, date, shooting circumstances, whether the person was armed, whether they exhibited signs of mental illness, and whether the incident was recorded.

2. CDC - "Bridged-Race Population Estimates 1990-2020 Results"
- This dataset offers estimated population figures for different racial groups in each state from 1990 to 2020.
- Its purpose in our analysis is to provide contextual information about racial populations in each state.

3. GkGigs - "List of Blue States and Red States"
- This dataset provides a list of states along with their dominant political party affiliation (blue/Democrat or red/Republican).
- We use this information to categorize states based on their political climate.

Upon initial examination, the first dataset alone does not provide a comprehensive understanding of racial disparities in police shootings. A simple observation of the total number of Black individuals compared to white individuals shot or killed by law enforcement over the past five years reveals a significantly larger number of white individuals, which contradicts our expectations. However, relying solely on raw numbers of shootings is insufficient for making valid inferences about targeted racial groups. To mitigate biased results, we incorporate the second dataset, which provides state-level racial population estimates. This allows us to analyze the number of people shot or killed in each racial group relative to the population composition of each racial category in a given area. By examining the data in terms of proportions, we aim to make unbiased comparisons that expose the true varying degrees of racial inequality in police brutality across the US. Lastly, we incorporate the third dataset into our analysis to explore potential relationships between a state's political climate and racial disparities in incidents of police violence.
