"""Our web app module"""
from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from processing import DataLoader, FEATURE_MARKDOWN_TABLE
from inference import BestModel
from typing import List, Dict, Union
from exceptions import EmptyFieldError

APP_FILE_PATH: str = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE_PATH: str = os.path.join(APP_FILE_PATH, 'model/finalized_model.sav')

@st.experimental_memo
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar: 
    st.image("../book/images/heart_disease.jpeg")
    st.title("Heart Disease Prediction")
    choice = st.radio("Navigation", ["Individual Upload", "Bulk Upload", "Profiling",  "Modelling & Download"])
    st.info("This application helps you explore your patient's data and predict if patient(s) have heart disease")

if choice == "Individual Upload": 
    st.title("Upload a single instance")
    st.markdown("Please enter the details of your patient. **All attributes must be filled**")
    
    # Features to be inputted by the user
    BMI = st.number_input('Enter BMI', 0)
    Smoking = st.text_input('Have the patient at least 100 cigarettes in your entire life? (Yes or No) ', '')
    AlcoholDrinking = st.text_input('Is your patient a heavy drinker (Yes or No)', '')
    Stroke = st.text_input('Has your patient ever had a stroke (Yes or No)', '')
    PhysicalHealth = st.number_input("Now thinking about your patient's physical health, which includes physical illness and injury, for how many days during the past 30 days was their physical health not good? (0-30)", 0)
    MentalHealth = st.number_input("For how many days during the past 30 days was your patient's mental health not good? (0-30)", 0)
    DiffWalking = st.text_input('Does your patient serious difficulty walking or climbing stairs? (Yes or No)', '')
    Sex = st.text_input('Is your patient male or female?', '')
    AgeCategory = st.text_input('Age of your patient in 5 year spans (18-24, 25-29, ... 75-79, 80 or older)', '')
    Race = st.text_input('Enter Race (White, Black, Asian, American Indian/Alaskan Native, Hispanic or  Other', '')
    Diabetic = st.text_input('Is your patient diabetic? (Yes or No)', '')
    PhysicalActivity = st.text_input('Did your patient perform any type of physical activity in the past 30 days? (Yes or No)', '')
    GenHealth = st.text_input('What is the general health of your patient (Poor, Fair, Good, Very good, Excellent)', '')
    SleepTime = st.number_input(' On average, how many hours of sleep does your patient get in a 24-hour period? ', 0)
    Asthma = st.text_input('Is your patient asthmatic? (Yes or No)', '')
    KidneyDisease = st.text_input('Does your patient have kidney disease? (Yes or No)', '')
    SkinCancer = st.text_input('Does your patient have skin cancer? (Yes or No)', '')
    
    if st.button("Predict"):
        # Create instance dataframe
        data_dict: Dict[str, Union[str, float]] = {'BMI': float(BMI),
                'Smoking': Smoking,
                'AlcoholDrinking': AlcoholDrinking, 
                'Stroke': Stroke,
                'PhysicalHealth': float(PhysicalHealth),
                'MentalHealth':  float(MentalHealth),  
                'DiffWalking': DiffWalking, 
                'Sex': Sex.title(),
                'AgeCategory': AgeCategory, 
                'Race': Race, 
                'Diabetic': Diabetic, 
                'PhysicalActivity': PhysicalActivity, 
                'GenHealth': GenHealth,
                'SleepTime': float(SleepTime),
                'Asthma':  Asthma, 
                'KidneyDisease':  KidneyDisease, 
                'SkinCancer': SkinCancer}
        instance_df =  pd.DataFrame(data_dict, index=[0])

        # If a user does not enter a field, raise an EmptyFieldError
        missing_fields: List[str] = []
        for column in instance_df.columns:
            if instance_df[column].values[0] == '':
                st.error(f'{column} was left blank.')  
                raise EmptyFieldError(f'{column} was left blank.')
        # Perform feature engineering 
        instance_data_loader: pd.DataFrame = DataLoader(instance_df)
        instance_df_processed: pd.DataFrame  = instance_data_loader.feature_engineer()
        
        # instantiate model and display prediction
        instance_model = BestModel(MODEL_FILE_PATH)
        instance_model.predict_single_instance(instance_df_processed)
    
if choice == "Bulk Upload":
    st.title("Bulk Upload Your Dataset")
    file = st.file_uploader("Upload Your CSV file")
    if file: 
        st.session_state.bulk_df = pd.read_csv(file)
        st.dataframe(st.session_state.bulk_df)
          
    st.markdown(FEATURE_MARKDOWN_TABLE)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    if 'bulk_df' not in st.session_state:
        st.error('Please bulk upload patient data to use this page')
    # Generates profile reports from a pandas DataFrame
    # https://pandas-profiling.ydata.ai/docs/master/index.html
    profile_df = st.session_state.bulk_df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling & Download":
    st.title("Run Bulk Predictions")
    st.markdown("**Note**, your dataset should already be uploaded.")
    if st.button('Run Bulk Prediction'):
        if 'bulk_df' not in st.session_state:
            st.error('Please upload your patient data first')
        # Perform feature engineering    
        bulk_data_loader: pd.DataFrame = DataLoader(st.session_state.bulk_df)
        bulk_df_processed: pd.DataFrame = bulk_data_loader.feature_engineer()
        # instantiate model and display prediction
        bulk_model: BestModel = BestModel(MODEL_FILE_PATH)
        results: pd.DataFrame = bulk_model.predict_bulk(bulk_df_processed)
        csv: bytes = convert_df(results)
        
        st.success('Predictions ready', icon="✅")  
        if st.download_button(
            "Press to Download Results",
            csv,
            "results.csv",
            "text/csv",
            key='download-csv'
            ):
            del st.session_state['bulk_df']