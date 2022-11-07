"Module containing the required data processing functionality"
import pandas as pd
from typing import List, Set
import numpy as np
pd.options.mode.chained_assignment = None

class DataLoader():
    """
    Takes in pandas df and performs the 
        necessary feature engineering steps
        required by our best perfoming model
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing attributes of patients
    """
    InputList  = List[str]
    binary_vars: List[str] = ["Smoking",
                            "AlcoholDrinking",
                            "Stroke", 
                            "DiffWalking",
                            "Diabetic",
                            "PhysicalActivity",
                            "Asthma",
                            "KidneyDisease",
                            "SkinCancer"]
    expected_columns: List[str] = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
                            'MentalHealth', 'DiffWalking', 'AgeCategory', 'Diabetic',
                            'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
                            'SkinCancer', 'Female', 'Male', 'American Indian/Alaskan Native',
                            'Asian', 'Black', 'Hispanic', 'Other', 'White', 'Race_freq', 'BMI_Bin',
                            'LOG_BMI']

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()


    def _encode_binary_vars(self) -> None:
        """
        Turn binary yes/no columns to contain [0, 1] instead
        """
        for binary_var in DataLoader.binary_vars:
            self.df[binary_var] = self.df[binary_var].replace({"Yes": 1, "No": 0})
        self.df.Diabetic: pd.Series = self.df.Diabetic.replace({'No, borderline diabetes': 1, 'Yes (during pregnancy)' : 1}).astype(int)
   
    def _onehot_encode_cat_vars(self)-> None:
        """Perform one-hot encoding on the categorical features"""
        
        sex_one_hot = pd.get_dummies(self.df['Sex'])
        self.df = self.df.join(sex_one_hot)
        self.df.drop(columns=['Sex'], inplace=True)

        self.df["AgeCategory"].replace({
            '18-24':1,
            '25-29':2,
            '30-34':3,
            '35-39':4,
            '40-44':5,
            '45-49':6,
            '50-54':7,
            '55-59':8,
            '60-64':9,
            '65-69':10,
            '70-74':11,
            '75-79':12,
            '80 or older':13
        }, inplace=True)
        self.df["GenHealth"].replace({
            'Poor': 1,
            'Fair':2, 
            'Good':3, 
            'Very good':4,  
            'Excellent':5},
            inplace=True)

        bins: List[int] = [0, 18.5, 24.9, 29.9, np.inf]
        names: List[int] = [1, 2, 3, 4]
        self.df['BMI_Bin']: pd.Series = pd.cut(self.df['BMI'], bins, labels=names)
        self.df["LOG_BMI"]: pd.Series = np.log(self.df["BMI"])
        
        race_one_hot: pd.DataFrame = pd.get_dummies(self.df['Race'])
        self.df: pd.DataFrame = self.df.join(race_one_hot)


    def _freq_encode_cat_vars(self):
        # Frequency encoding using value_counts function 
        race_freq: pd.Series = self.df['Race'].value_counts(normalize=True)

        # Mapping the encoded values with original data 
        self.df['Race_freq']: pd.Series = self.df['Race'].apply(lambda x : race_freq[x])
        self.df.drop(columns=['Race'], inplace=True)     
    

    def feature_engineer(self) -> pd.DataFrame:
        self._encode_binary_vars()
        self._onehot_encode_cat_vars()
        self._freq_encode_cat_vars()
        
        missing_columns: Set[str] = set(DataLoader.expected_columns).difference(
            set(self.df.columns)
        )
        for miss_col in list(missing_columns):
             self.df[miss_col] = [0] * self.df.shape[0]
        
        return self.df[DataLoader.expected_columns]

FEATURE_MARKDOWN_TABLE: str = """
| Field | Description |
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
"""
