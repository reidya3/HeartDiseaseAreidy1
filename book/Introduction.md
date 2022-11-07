# Introduction

Heart disease, alternatively known as cardiovascular disease, encases various conditions that impact the heart and is the primary basis of death worldwide over the span of the past few decades. Approximately 10,000 people die in Ireland from Cardiovascular Disease each year, accounting for 36% of deaths per annum {cite:p}`laya_healthcare`. That’s despite the fact that 80% of all heart disease is deemed preventable. This investigations aims to explore a [kaggle dataset](https://www.kaggle.com/code/mushfirat/heartdisease-eda-prediction/notebook) and build a model that can predict the likelihood of a patient having heart disease. More specifically, with this dataset, we would like to see if we can develop a good model to predict if a person has heart disease and what *factors* can be attributed to heart disease most directly. We will be tackling this question with the usage of different regression techniques and algorithms.


## Description of Dataset
We uses an existing dataset from [Kaggle](https://www.kaggle.com/code/mushfirat/heartdisease-eda-prediction/notebook). The dataset comprises approx 320,0000 instances and 14 attributes.

:::{note}
No personal identifiable information of the patients are recorded in the dataset.
:::

The table below summarizes the multiple columns used in this investigation.

| Name | Description |
| :--- | :--- |
| HeartDisease | Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI). |
| BMI | Body Mass Index. |
| Smoking | Have you smoked at least 100 cigarettes in your entire life? |
| AlcoholDrinking | Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week |
| Stroke | (Ever told) (you had) a stroke? |
| PhysicalHealth | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days). |
|MentalHealth|  Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days). |
| DiffWalking | Do you have serious difficulty walking or climbing stairs? |
| Sex | Are you male or female? |
| AgeCategory | Fourteen-level age category. (then calculated the mean) |
| Race | Imputed race/ethnicity value. |
| Diabetic | (Ever told) (you had) diabetes? |
| PhysicalActivity |  Adults who reported doing physical activity or exercise during the past 30 days other than their regular job. |
| GenHealth | Would you say that in general your health is... |
| SleepTime | On average, how many hours of sleep do you get in a 24-hour period? |
| Asthma | (Ever told) (you had) asthma? |
| KidneyDisease | Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease? |
| SkinCancer | (Ever told) (you had) skin cancer?  |

 # Methodology
 
Methodology will follows a typical data science project: from understanding the dataset through exploratory data analysis, data preparation, model buildings and finally model evaluation. We seek to build a model that predicts heart disease, a binary outcome
In this investigation, we seek to use the K-means clustering approach to segment the patients into well-defined groups.
To start, we perform an initial data exploration to perform transformations & data sanitization checks; acquire rudimentary statistics of the datasets; perform data augmentation; create exploratory visualizations. Next, we perform cluster analysis and evaluate our clusters using metrics such as Silhouette Coefficient and an Elbow curve. 
These clusters represent participants that exhibit similar risk factors for heart disease and may have similar underlying determinants of health such as their age, BMI, whether the smoke or have asthma. Next, we envision the probability of developing heart disease in the patients. Finally, we conclude with the most important outcomes of our work. 

$$

$$