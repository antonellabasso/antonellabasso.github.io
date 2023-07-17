---
code: PHP2550
title: High-Risk Serovar Prevalence in Sources of Listeria Monocytogenes: A Clustering Analysis of Food-Borne Illness
mathjax: true
layout: page
categories: media
permalink: /collections/:title
---

#### Antonella Basso and Nathan Provost

<center><b> Abstract </b></center> 

**Relevance:** *Listeria monocytogenes* (hereafter listeria) is a dangerous species of bacteria that poses a direct threat to public health, and the association between severe strains of listeria and their sources of origin has not been studied with sufficient precision and attention to detail.

**Goals:** We endeavor to create two clustering models for data pertaining to the sources, collection times, minimum self-same distances, and genetic information of various strains of listeria, so that we can assess the intensity of particularly problematic strains within clusters. We also aim to compare our new models to a pre-existing genetic cluster model included in the dataset in terms of similarity.

**Methods:** We employ traditional k-means clustering with optimal cluster selection using silhouette coefficients and network clustering with similarity weighting through the `linkcomm` package in R (Kalinka and Tomancak 2011). Data is assessed for the time period between 2017 and late 2022, with each year divided into four quarters. Sources of isolation are grouped coarsely and missing data is not used in our analysis, which allows for 2,290 unique strains of listeria to be analyzed.

**Outcomes of Interest:** Serovar 4b prevalence by cluster, source prevalence by cluster, Rand index values, silhouette coefficient plots, community centrality, and community modularity are all outcomes of interest. Visualizations of cluster interactions and networks are also objects of interest.

**Results:** A strong sense of similarity exists between our sample networks and the SNP model, while a moderate sense of similarity exists between our k-means model and both our network model and the genetic model. Both genetic and non-genetic factors appear to influence serovar 4b prevalence, but specific recommendations cannot be made. Our methods provide a framework for future research and reproductions.

**Conclusion:** It is likely that severe strains of listeria manifest more frequently in specific isolation sources, but our analysis is too limited to provide any further guidance on this matter. Future studies should aim to atomize isolation sources into more precise groups in order to better understand the relationship between strain severity and source. However, our methodology has laid the groundwork for more intensive research.


[**Full Paper**](https://antonellabasso.github.io/HW pdfs/PHP2550_Basso_FinalProject_Report.pdf) <br>
[**Github Repo**](https://github.com/NTProvost/PHP-2550-Project)

![Poster Presentation](https://antonellabasso.github.io/IMAGES/PHP2550_FinalProject_Poster.png) 
