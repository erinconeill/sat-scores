# sat-scores
This Jupyter Notebook script visualizes the relationship between SAT scores and years taken while considering the region. It was a final project for an intermediate programming class, integrating various techniques learned during the course, including KMeans clustering, GeoPandas, and correlation analysis. 

## Description

This script reads data from CSV files and generates visualizations to longitudinally analyze how SAT scores have changed. It includes choropleth graphs, a heat map, and linear regression plots to represent the data.

The average composite ACT score fell to 19.5 out of a possible 36 for the class of 2023, the lowest since 1991. The proportion of what the ACT calls 'COVID cohort seniors' who did not meet any of the benchmark scores considered necessary to succeed in college reached a historic high of 43%. The analysis explores if the same downward trend is seen with other standardized tests, like the SAT. The conclusions drawn from the analysis would provide a basis for shifting away from standardized testing and related funding.

Besides understanding the relationship between scores and years, the analysis also investigates whether region or percent taking (X) impacts scores (Y) and to what extent. Other questions about the dataset naturally arose during the process.

A dataset from the National Center for Education Statistics provided SAT mean scores, standard deviations, and the percentage of the graduating class taking the test from 2017 to 2022. These years were chosen to examine and compare pre- and post-COVID-19 data.

Breakdown of the dataset:
- 12 NE
- 12 SE
- 12 MID
- 11 W (9 excluding Hawaii and Alaska)
- 4 SW

Following the pattern of regions that allocate the most funding to academics, it was hypothesized that SAT scores would be higher on the East and West coasts than in the Midwestern and Southern states. It was assumed that SAT scores would align with the ACT trends and gradually decrease over time.

To answer the initial predictions and the ones that followed, the following data science approaches were used:
- Regression and correlation analysis to determine the relationship between variables and their corresponding strengths
- Clustering and classification of SAT scores across years to determine if the data was differentiated or homogeneous

#### How Do Composite SAT Scores Vary by State? (choropleth map):

A map showing the distribution of nationwide SAT scores. At first glance, it might appear that states in the Midwest have higher scores. However, fewer people take the SAT in the Midwest. Upon further investigation, it was discovered that only ~2.5% of students take the test in these states. An inverse relationship was observed between average composite SAT scores and the percentage of students taking the test; fewer students taking the test resulted in a higher average composite score for the given state.

#### Are Composite SAT Scores Impacted by Percent of Students Taking SAT? (correlation heat map and linear regression):

After seeing how SAT scores and the percentage of students taking the exam are distributed across the states, the relationship between the percentage of students taking the exam and composite SAT scores was evaluated via a correlation heat map and linear regression. The resulting correlation was -.84, indicating a strong negative correlation between the two variables. This means that as the percentage of students taking the exam increases, the composite SAT score decreases, and vice-versa. Regression analysis was used to find the equation of this relationship. By plotting this equation and creating a scatter plot of the data, it was evident that a lower percentage of students taking the exam results in an increase in the average state SAT score.

#### Have Composite Scores or Percent Taking in All States Decreased Over the Past Five Years? (regression analysis):

After finding a strong relationship between SAT scores and the percentage of students taking the exam, the next step was to see if the same held true for the relationships that the year had with the percentage taking and composite scores. As seen with the regression analysis, there is no strong relationship between the year and either variable. The top scores did decrease slightly during 2020 and beyond.

#### Do Top and Bottom Ranking States in Composite Scores Vary Pre- and Post-Pandemic?

After seeing that there was no relationship between year and score, the states contributing to the extremes of the score range were examined. Although the scores remained relatively the same, only Kansas and Wisconsin consistently placed in the top five composite scores. The same pattern occurred for the bottom five, with only Delaware staying consistent. For math, Wisconsin and Minnesota consistently placed in the top five, while Florida and Delaware stayed in the bottom five. For English, Wisconsin stayed in the top five and Delaware continued to be in the bottom five over the six-year stretch. For the percentage of students taking the exam, Delaware is consistently in the top five, while Arkansas and Mississippi are consistently in the bottom five.

#### Clustering and Classification of Scores and Regions:

Two graphs were plotted, demonstrating the clustering of scores and identification of regions. Based on the two plots, it was concluded that the Midwest was most represented in the top quarter of SAT scores for 2017 and 2022, and more states fell into the bottom cluster in 2022 compared to 2017. A similar model could be used for unsupervised ML classification, potentially predicting the region based on a provided score.

#### Analysis/Discussion: 

A negative inverse relationship was found between the composite SAT score and the percentage of students taking the exam. Despite this, no significant evidence was found that SAT scores or the percentage of students taking the exam are decreasing over time. The states responsible for the extreme values of scores and percentage taking change over time. Following cluster analysis, the Midwest represented the top quarter of scores, further supporting the initial conclusions drawn in the correlation analysis.

#### Limitations/Improvements: 

- More quantitative data could provide a better understanding of trends (i.e., the 90s-2017 data).
- Examining more retrospective data would require conversions as the scoring differed from 2006-2016.
- Qualitative data could help better identify covariates (race, socioeconomic status, etc.) and explain why some trends are occurring (e.g., why the percentage is lower in the Midwest).

## Dependencies

- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [os](https://docs.python.org/3/library/os.html)
- [scipy](https://scipy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
