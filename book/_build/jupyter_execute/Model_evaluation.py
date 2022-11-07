#!/usr/bin/env python
# coding: utf-8

# # Model Building, Evaluation & Sensitivity Analysis
# 
# In this section, we will present a comparative analysis of the heart disease classification problem using different classification algorithms. We use the 80:20 train-test split rule to evaluate our models.
# **Note**, a small amount of patients were used as  as hold-out set for hyper-parameter sensitivity analysis. 
# 
# 
# ## Models
# We choose numerous shallow predictive methods to predict heart disease. 
# - [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
# - [Gradient Boost Classifier](https://en.wikipedia.org/wiki/Gradient_boosting)
# - [Decision Tree Classifier](https://en.wikipedia.org/wiki/Decision_tree_learning)
# - [Random Forest Classifier](https://en.wikipedia.org/wiki/Random_forest)
# - [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
# 
# For more information on these algorithms, please click on the relevant links 
# 
# ## Evaluation metrics
# One of the key requirements in developing any algorithm is to measure it's effectiveness. Accuracy is the most simple measure. It tells us the he number of correctly classified examples over the total number of examples. More formally, 
# $$
#  Accuracy = \frac{TruePositive + TrueNegative}{TruePositive + TrueNegative + FalsePositive + FalseNegative  }
# $$
# But is accuracy telling the whole picture ?
# Well, let’s consider those two examples:
# - A classifier which, if a person has the heart disease, will always correctly diagnose it, but gets half of the healthy people wrong. You can see that announcing to a healthy person that he or she has the disease could lead to adverse consequences.
# - A classifier that gets the diagnose right for every healthy person, but also miss half of the disease cases. That wouldn’t be a very good algorithm would it?
# 
# Depending on the distribution of sick to healthy patients those two classifiers could have high accuracy while not being considered very good. Therefore, we decide to employ three further metrics
# - *Precision:* dertimnes what proportion of the negative class got correctly classified. 
#   $$
#   \frac{TruePositive}{TruePositive +   FalsePositive    }
#   $$
#  
# - *Recall*: determine what proportion of the actual sick people were correctly detected by the model.
#   $$
#   \frac{TruePositive}{TruePositive +    FalseNegative   }
#   $$
# 

# ## Import libaries
# ### Data Processing 

# In[1]:


import pandas  as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
warnings.filterwarnings('ignore')


# ### Model building and evaluation

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score


from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    NearMiss
)
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
)


# ### Model Serialisation

# In[3]:


import pickle


# ### Data Visualisation

# In[4]:


import matplotlib.pyplot as plt


# In[5]:


dict_classifiers = {
    "Logistic Regression ": LogisticRegression(),
    "Gradient Boost Classifier": GradientBoostingClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
}

heart_disease_dataset_standardized = pd.read_csv("data/standardised_heart_disease.csv")

# Initial Attempt
train_heart_disease_df = heart_disease_dataset_standardized.drop('HeartDisease',axis='columns')
test_heart_diease_df = heart_disease_dataset_standardized.HeartDisease.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(train_heart_disease_df, test_heart_diease_df, test_size=0.2, random_state=15, stratify=test_heart_diease_df)


# In[6]:


print(f"Ratio of classes in training set:\n{y_train.value_counts(normalize=True)}")
print()
print(f"Ratio of classes in test set:\n{y_test.value_counts(normalize=True)}")
      


# We balance the test dataset, to ensure accuracy is a fair measure of model performance.

# In[7]:


test_df = X_test.copy()
test_df['HeartDisease'] = y_test
class_0 = test_df[test_df['HeartDisease'] == 0]
class_1 = test_df[test_df['HeartDisease'] == 1]

class_1 = class_1.sample(len(class_0),replace=True)
test_df = pd.concat([class_0, class_1], axis=0)
print('Data in Test:')
print(test_df['HeartDisease'].value_counts())
X_test = test_df.drop('HeartDisease',axis='columns')
y_test = test_df.HeartDisease.astype(np.float32)


# In[8]:


def run_exps(
             X_train, X_test, y_train, y_test) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: None
    '''
    
    results = pd.DataFrame()
    for model_name, model in dict_classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        class_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease'],output_dict=True)).transpose().reset_index()
        class_report['model'] = [model_name] * class_report.shape[0]
        class_report['auc'] = [roc_auc_score(y_test, y_pred)] * class_report.shape[0]
        class_report = pd.concat([class_report], keys=['Model'], names=['Firstlevel'])
        results = pd.concat([results, class_report], ignore_index=True)
    results['dummy'] = None

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))

    for cls, ax in zip(dict_classifiers.values(), axes.flatten()):
        plot_confusion_matrix(cls, 
                            X_test, 
                            y_test, 
                            ax=ax, 
                            cmap='Blues',
                            display_labels=['No Heart Disease', 'Heart Disease'])
        ax.title.set_text(type(cls).__name__)
    plt.tight_layout()  
    plt.show()

    print(f"""Algorithm with the highest accuracy: {
        results[results['index'] == 'accuracy'].sort_values(
            ['support'], ascending=False).head(1)[['model', 'support']].values[0].tolist()}""")
    
    print(f"""Algorithm with the highest macro recall:
        {results[results['index'] == 'macro avg'].sort_values(
            ['recall'], ascending=False).head(1)[['model', 'recall']].values[0].tolist()}""")
    
    print(f"""Algorithm with the highest macro precision:
        {results[results['index'] == 'macro avg'].sort_values(
            ['precision'], ascending=False).head(1)[['model', 'precision']].values[0].tolist()}""")
    
    print(f"""Algorithm with the highest AUC:
        {results.sort_values(['auc'], ascending=False).head(1)[
            ['model', 'auc']].values[0].tolist()}""")
    
    print(results.groupby(
        ['model', 'index', 'precision', 'recall', 'f1-score', 'support',  'auc']
        )['dummy'].count())
    
    return results


# ## Imbalanced data
# As you can see above, our data is extremely imbalanced. Imbalanced datasets are those where there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class to the majority class.
# 
# This bias in the training dataset can influence many machine learning algorithms, leading some to ignore the minority class entirely. This is a problem as it is typically the minority class on which predictions are most important (i.e. predicting heart disease in our case).
# 
# One approach to addressing the problem of class imbalance is to randomly resample the training dataset. The two main approaches to randomly resampling an imbalanced dataset are to delete examples from the majority class, called undersampling, and to duplicate examples from the minority class, called oversampling.
# 
# 

# ## Undersampling 
# 
# 
# 
# The follwoing undersampling methods were choosen:
# - `RandomUnderSampler`: Random undersampling consists in extracting at random samples from the majority class, until they reach a certain proportion compared to the minority class, typically 50:50.
# - `CondensedNearestNeighbour`: The algorithms works as follows:
#     - Put all minority class observations in a group, typically group O
#     - Add 1 sample (at random) from the majority class to group O
#     - Train a KNN with group O
#     - Take a sample of the majority class that is not in group O yet
#     - Predict its class with the KNN from point 3
#     - If the prediction was correct, go to 4 and repeat
#     - If the prediction was incorrect, add that sample to group O, go to 3 and repeat
#     - Continue until all samples of the majority class were either assigned to O or left out
#     - Final version of Group O is our undersampled dataset
# 
#     This algorithm tends to pick points near the fuzzy boundary between the classes, and transfer those to the group O, in our example. If the classes are similar, group O will contain a fair amount of both classes. If the classes are very different, group O would contain mostly 1 class, the minority class.
# - `TomekLinks`: Tomek links are 2 samples from a different class, which are nearest neighbours to each other. In other words, if 2 observations are nearest neighbours, and from a different class, they are Tomek Links. This procedures removes either the sample from the majority class if it is a Tomek Link, or alternatively, both observations, the one from the majority and the one from the minority class.
# - `OneSidedSelection`: First finds the hardest instances to classify correctly from the majority class. Then removes noisy observations with Tomek Links.
# - `EditedNearestNeighbours`: Train a KNN algorithm on the data (user defines number of neighbours, typically 3)
#   - Find the 3 nearest neighbour to each observation (or the number defined by the user in 1)
#   - Find the label of each of the neighbours (we know it, is the target in the dataset)
#   - if the majority of the neighbours show the same label as the observation, then we keep the observation
#   
# - `RepeatedEditedNearestNeighbours`: Extends Edited Nearest neighbours in that it repeats the procedure over an over, until no further observation is removed from the dataset, or alternatively until a maximum number of iterations is reached.
# - `AllKNN`: Adapts the functionality of Edited Nearest Neighbours in that, at each round, it increases the number of neighbours utilised to exclude or retain the observations.
#           It starts by looking at the 1 closest neighbour.
#           It finishes at a maximum number of neighbours to examine, determined by the user
#           it stops prematurely if the majority class becomes the minority
# - `NeighbourhoodCleaningRule`: The Neighbourhood Cleaning Rule works as follows:
#     1) Remove noisy observations from the majority class with ENN:
#     - explores the 3 closest neighbours\n
#     - uses majority vote of neighbours to retain observations
#     2) Remove observations from the majority class if:,
#     - they are 1 of the 3 closest neighbours to a minority sample, and,
#     - most / all of those 3 closest neighbours are not minority, and,
#     - the majority class has at least half as many observations as those in the minority (this can be regulated)
# - `NearMiss`: This procedures aims to select samples that are somewhat similar to the minority class, using 1 of three alternative procedures:
#   - Select observations closer to the closest minority class
#   - Select observations closer to the farthest minority class
#   - Select observations furthest from their nearest neighbours
# 
# 
# :::{note}
# We train the models on a portion of the data that is under-sampled
# We evaluate the model performance on another portion of the data that was not resampled, and thus contains the original class distribution.
# :::

# :::{note}
# In addition a verbose output of the models performance will be generated. 
# :::

# In[9]:


undersampler_dict = {

    'random': RandomUnderSampler(
        sampling_strategy='auto',
        random_state=0,
        replacement=False),

    # 'cnn': CondensedNearestNeighbour(
    #     sampling_strategy='auto',
    #     random_state=0,
    #     n_neighbors=1,
    #     n_jobs=4),

    'tomek': TomekLinks(
        sampling_strategy='auto',
        n_jobs=4),

    'oss': OneSidedSelection(
        sampling_strategy='auto',
        random_state=0,
        n_neighbors=1,
        n_jobs=4),

    # 'enn': EditedNearestNeighbours(
    #     sampling_strategy='auto',
    #     n_neighbors=3,
    #     kind_sel='all',
    #     n_jobs=4),

    # 'renn': RepeatedEditedNearestNeighbours(
    #     sampling_strategy='auto',
    #     n_neighbors=3,
    #     kind_sel='all',
    #     n_jobs=4,
    #     max_iter=100),

    # 'allknn': AllKNN(
    #     sampling_strategy='auto',
    #     n_neighbors=3,
    #     kind_sel='all',
    #     n_jobs=4),

    # 'ncr': NeighbourhoodCleaningRule(
    #     sampling_strategy='auto',
    #     n_neighbors=3,
    #     kind_sel='all',
    #     n_jobs=4,
    #     threshold_cleaning=0.5),

    'nm1': NearMiss(
        sampling_strategy='auto',
        version=1,
        n_neighbors=3,
        n_jobs=4),

    'nm2': NearMiss(
        sampling_strategy='auto',
        version=2,
        n_neighbors=3,
        n_jobs=4),
}


# In[10]:


# train a model on the original data without under-sampling
# and determine model performance
print("No UnderSampling")
print("-------------------")
run_exps(X_train, X_test, y_train, y_test)

print("UnderSampling Methods")
print("-------------------")
print()

# now, we test the different under-samplers, 1 at a time
for undersampler in undersampler_dict.keys():
    
    print(undersampler)
    print("-------------------")
    
    # resample the train set only
    X_resampled, y_resampled = undersampler_dict[undersampler].fit_resample(X_train, y_train)
    
    # train model and evaluate performance
    
    # Note the performance returned is using the
    # test set, which was not under-sampled
    
    run_exps(X_resampled, X_test, y_resampled, y_test)
    
    print()
    
print()


# In[ ]:


# #No Undersamping 
# Algorithm with the highest accuracy: ['Gradient Boost Classifier', 0.9157898028424459]
# Algorithm with the highest macro recall:
#         ['Naive Bayes', 0.6833829351601921]
# Algorithm with the highest macro precision:
#         ['Gradient Boost Classifier', 0.7359577899388909]
# Algorithm with the highest AUC:
#         ['Naive Bayes', 0.6833829351601921]
# # Random 
# Algorithm with the highest accuracy: ['Naive Bayes', 0.777842055066527]
# Algorithm with the highest macro recall:
#         ['Gradient Boost Classifier', 0.7636522622274398]
# Algorithm with the highest macro precision:
#         ['Logistic Regression ', 0.5987409474695906]
# Algorithm with the highest AUC:
#         ['Gradient Boost Classifier', 0.7636522622274399]
# # Tomek
# Algorithm with the highest accuracy: ['Gradient Boost Classifier', 0.9158679779233572]
# Algorithm with the highest macro recall:
#         ['Naive Bayes', 0.6871222195884508]
# Algorithm with the highest macro precision:
#         ['Gradient Boost Classifier', 0.7313563006184678]

# # OSS 
# Algorithm with the highest accuracy: ['Gradient Boost Classifier', 0.9157428977938992]
# Algorithm with the highest macro recall:
#         ['Naive Bayes', 0.687578840593017]
# Algorithm with the highest macro precision:
#         ['Gradient Boost Classifier', 0.7293058306944802]
# Algorithm with the highest AUC:
#         ['Naive Bayes', 0.6875788405930171]
        
# # NM2     
# Algorithm with the highest accuracy: ['Naive Bayes', 0.45100767679294546]
# Algorithm with the highest macro recall:
#         ['Logistic Regression ', 0.5912552564819664]
# Algorithm with the highest macro precision:
#         ['Logistic Regression ', 0.5313939861738548]
# Algorithm with the highest AUC:
#         ['Logistic Regression ', 0.5912552564819664]


# As we can see in the verbose model output, we achieved the F1 score of 0.88 for the Random Forest Classifier. This is a very good score, and it is the highest score we achieved so far. 

# ## Oversampling
# 
# The following undersampling methods were choosen:
# - `Random Oversampling:` Random over-sampling consists in extracting at random samples from the minority class, until they reach a certain proportion compared to the majority class, typically 50:50, or in other words, a balancing ratio of 1.
# - `SMOTE`: Creates new samples by interpolation of samples of the minority class and any of its k nearest neighbours (also from the minority class). K is typically 5.
# - `ADASYN`: Creates new samples by interpolation of samples of the minority class and its closest neighbours. It creates more samples from samples that are harder to classify.
# - `Borderline SMOTE`: Creates new samples by interpolation between samples of the minority class and their closest neighbours.
#   - It does not use all observations from the minority class as templates, unllike SMOTE.
#   - It selects those observations (from the minority) for which, most of their neighbours belong to a different class (DANGER group)
#     - Variant 1 creates new examples, as SMOTE, between samples in the Danger group and their closest neighbours from the minority
#     - Variant 2 creates new examples between samples in the Danger group and neighbours from minority and majority class
# - `SVM SMOTE`: Creates new samples by interpolation of samples of the support vectors from minority class and its closest neighbours.
# 
# 
# 
# 
# :::{note}
# We train the models on a portion of the data that is over-sampled
# We evaluate the model performance on another portion of the data that was not resampled, and thus contains the original class distribution.
# :::

# In[ ]:


oversampler_dict = {

    'random': RandomOverSampler(
        sampling_strategy='auto',
        random_state=0),

    'smote': SMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        n_jobs=4),

    'adasyn': ADASYN(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        n_neighbors=5,
        n_jobs=4),

    'border1': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-1',
        n_jobs=4),

    'border2': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-2',
        n_jobs=4),

    # 'svm': SVMSMOTE(
    #     sampling_strategy='auto',  # samples only the minority class
    #     random_state=0,  # for reproducibility
    #     k_neighbors=5,
    #     m_neighbors=10,
    #     n_jobs=4,
    #     svm_estimator=SVC(kernel='linear')),
}


# In[ ]:


# train a model on the original data without under-sampling
# and determine model performance
print("No OverSampling")
print("-------------------")
run_exps(X_train, X_test, y_train, y_test)

print()

print("OverSampling Methods")
print("-------------------")

# now, we test the different under-samplers, 1 at a time
for oversampler in oversampler_dict.keys():
    
    print(oversampler)
    print("-------------------")
    
    # resample the train set only
    X_resampled, y_resampled = oversampler_dict[oversampler].fit_resample(X_train, y_train)
    
    # train model and evaluate performance
    
    # Note the performance returned is using the
    # test set, which was not under-sampled
    
    run_exps(X_resampled, X_test, y_resampled, y_test)
    
    print()
    
print()


# In[ ]:


# No oversampling 


# ## Feature Selection
#  The idea of feature selection and extraction is to avoid the curse of
# dimensionality. This refers to the fact that as we move to higher
# dimension input feature spaces the volume of the space grows rapidly
# and we end up with very few instances per unit volume, i.e. we have
# very sparse sampling of the space of possible instances making
# modelling difficult.
# 
# Feature Selection: It is clear from what we have seen that a good feature engineering
# idea might be to choose a subset of the features available to reduce
# the dimension of the feature space. This act is called feature
# selection. One way of doing this is to try out different permutations of
# features increasing the numbers of features involved as you proceed
# and calculate machine learning performance. This is rarely practical
# though. More efficient approaches include wrapper, filter and
# embedded methods.
# 
# We decide to explore the following methods:
# - Perform PCA anaylsis and idenitify the variables that most contribute to the 
# - A simple filter method:
#   - Identify input features having high correlation with target variable.
#   - Identify input features that have a low correlation with other independent variables
#   - Find the information gain or mutual information of the independent variable with respect to a target variable
# - Permutation Based Feature Importance: ill randomly shuffle each feature and compute the change in the model’s performance
# 
# ### PCA
# 
# 

# PCA is mathematically defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
# 
# We decide to get the top 5 features that contribute most to the first principal component and the top 5 features that contribute most to the second principal compoenent.

# In[ ]:


def pca_most_important_features(df: pd.Dataframe) -> list[str]:
    """
    Retrieve the top 10 features that contribute most 
        variation to the top 2 principal components
    """
    model = PCA(n_components=2).fit(df)

    # number of components
    n_pcs: int = model.components_.shape[0]

    most_important_features_indicies: List[int] = []
    for i in range(n_pcs):
        top_5 = np.argpartition(np.abs(model.components_[i]), -5)[-5:].tolist()
        most_important_features_indicies.extend(top_5)
        
    most_important_features_indicies = list(set(most_important_features_indicies))
    initial_feature_names = df.columns
    most_important_names  = [initial_feature_names[i] for i in most_important_features_indicies]
    
    return most_important_names
    


# In[ ]:


pca_df = pca_most_important_features()


# In[ ]:





# ### Filter Methods
# 
# A simple filter method:
#   - `Identify input features having high correlation with target variable`: We want to keep features with only a high correlation with the target variable. This implies that the input feature has a high influence in predicting the target variable. We set the threshold to the absolute value of 0.2. We keep input features only if the correlation of the input feature with the target variable is greater than 0.2. Our analysis reveled most variables have little if all correlation to our target variable
# 
#   - `Find the information gain or mutual information of the independent variable with respect to a target variable`

# In[ ]:


def idenity_high_corr_features(df: pd.DataFrame) -> List[str]:
    importances = df.drop(
        "HeartDisease", axis=1).apply(
            lambda x: x.corr(df.HeartDisease))
    
    indices = np.argsort(importances)
    important_feature_names = []
    for i in range(0, len(indices)):
        if np.abs(importances[i]) > 0.2:
            important_feature_names.append(
                df.columns[i])
    return important_feature_names
    


# In[ ]:


mi = mutual_info_classif(heart_disease_dataset_standardized.drop("HeartDisease", axis=1),
                         heart_disease_dataset_standardized["HeartDisease"])
mi = pd.Series(mi)
mi.index = heart_disease_dataset_standardized.drop("HeartDisease", axis=1).columns
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))


# ## Hyperparameter finetuning
# Our best algorithm was
# 
# Our final step is to fine-tune our model. We will use GridSearch to achieve this. 

# In[ ]:


param_grid = {
    'min_samples_split': [3, 5, 10], 
    'n_estimators' : [100, 300],
    'max_depth': [3, 5, 15, 25],
    'max_features': [3, 5, 10, 20]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


# In[ ]:


def grid_search_wrapper(refit_score='accuracy_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of our final model optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


# In[ ]:





# ## Pickle Model
# Pickle is the standard way of serializing objects in Python.
# 
# You can use the pickle operation to serialize your machine learning algorithms and save the serialized format to a file.
# 
# Later we will load this file to deserialize your model and use it to make new predictions in our web app.

# In[ ]:


model  =  GaussianNB()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


set(X_train.columns)


# In[ ]:


pickle.dump(model, open('../app/model/finalized_model.sav', 'wb'))


# In[ ]:




