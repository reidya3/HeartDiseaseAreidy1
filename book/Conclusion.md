# Conclusion 

In this investigation, we investigate the predictability of patients who have or did not have heart disease. Engineered features included a log transformed BMI variable to offset the effect of outliers in our model, a substance abuse binary variable (whether they were a heavy smoker or drinker) and two principal components of the dataset, in an attempt to reduce the dimensionality.  In addition, we augment the existing data by adding participants general BMI classification (underweight, normal weight, overweight obese). This enables us, to examine not only the the risk factors for heart disease but also the risk factors for heart disease in the context of a patient's BMI.  Furthermore, our method of using K-means clustering to segment patients into groups with similar risk factors for heart disease is a our approach to understanding the risk factors for heart disease. We propose that this method of segmenting patients into groups with similar risk factors for heart disease can be used to identify patients that are at higher risk of developing heart disease and thus, we added this feature into our final model

Our most significant results are the following: 
1. Exploratory data analysis contradicts  patients own self-reported health status. The majority of patients who have heart disease report having good or very good health, and the majority of patients who do not have heart disease report having fair or poor health. This suggests that the patients themselves are not aware of their heart disease status.

2. The most significant risk factors for heart disease for the general population are smoking, alcohol drinking, obesity, and diabetes.
   
3. Similarity, our findings suggest that patients who are obese and have a substance abuse problem are at a significantly higher risk of developing heart disease. This is not surprising as obesity is a well known risk factor for heart disease. However, the fact that patients who are obese and have a substance abuse problem are at a significantly higher risk of developing heart disease in the context of their BMI is more interesting. While this is not surprising, it does suggest that there are other factors that impact the risk of heart disease in obese patients.
   
4. Out of all the undersupplying and oversampling techniques we investigated to overcome the class imbalance problem in our dataset, random oversampling seemed to be most fruitful. This is because it oversampled the minority class (patients who have heart disease) to the same number of samples as the majority class (patients who do not have heart disease). This is important because it ensures that the model is not biased towards the majority class.

5. Our final model was serialized using the  pickle library. This allowed us to create a responsive web application. More details on the web app can be found in the [app directory](app).


   

## Future work

As an extension to this work, and some sort of limitation to the work performed here,
different types of classifiers can be included in the analysis and more in depth sensitivity analysis can be performed on these classifiers, also an extension can be made by applying same analysis to other bioinformatics diseases’ datasets, and see the performance of these classifiers to classify and predict these diseases. In addition, we would like to investigate the use of deeper models. Similar endeavors have shown to be fruitful, albeit often decreasing the interoperability of results.  Finally, we are interested in incorporating other features about the subjects such as socio-economic status,  heart disease prevalence in their family (measured on some continuum), their blood pressure and  cholesterol levels, and their dietary habitats. We hope such features might uncover specific genetic components  patterns or behavioural aspects that might increase or decrease the likehood of heart disease. 