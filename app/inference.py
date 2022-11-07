"""
Module contains the BestModel which instantiates 
    our best performing model
"""
from exceptions import InvalidTxtError
import pandas as pd
import numpy as np
import pickle
import sklearn
import streamlit as st
import seaborn as sns
from typing import Dict, List
import matplotlib.pyplot as plt

class BestModel:
    """
    This class
        (a) deserialize the model from a pickle file
        (b) predict whether if an individual patient has heart disease
        (c) predict if a bulk of patients have heart disease
        
    Parameters
    ----------
    input_filepath : str
        The filepath of the pickle file
    """
    target_class: Dict[int, str] = {
        0: "No Heart Disease", 1:"Heart Disease"}
    
    def __init__(self, input_filepath: str) -> None:
        self.model = BestModel._load_model(input_filepath)

    @staticmethod
    def _load_model(input_filepath: str):
        """
        Loads sklearn model
        """
        try:
            with open(input_filepath, 'rb') as f:
                model = pickle.load(f)
        except Exception as err:
            raise InvalidTxtError("Invalid model pickle filename") from err
        return model
    
    def predict_single_instance(self, instance_df: pd.DataFrame) -> None:
        """ 
        Read all the data entered by the user. The predicted result and the probabilities are then retrieved. 
        The result is printed to the webpage. Also, the probabilities of each class are displayed (bar graph).
        
        This function gets called automatically when the user clicks on the “Predict” button
            in the Individual Upload view.
        """
        preds: List[float] = self.model.predict_proba(instance_df)[0]
        predicted_class: str = BestModel.target_class[np.argmax(preds)]
        probs = [np.round(x, 4) for x in preds]
        
        st.write("The predicted class is ", predicted_class)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        ax = sns.barplot(x=probs, y=list(BestModel.target_class.values()), palette="winter", orient='h')
        plt.title("Probabilities of the patient belonging to each class")
        for index, value in enumerate(probs):
            plt.text(value, index,str(value))
        st.pyplot()
        
    
    def predict_bulk(self, bulk_df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Read all the data entered by the user, via the CSV upload button.
        Use the model to predict if the patients have heart disease
        
        Return
        ----------
        results : pd.DataFrame
            The original data supplied by the user, along with a new column called "PredictedHeartDisease" appended
        """
        preds: List[int] = self.model.predict(bulk_df).tolist()
        
        results = bulk_df
        results['PredictedHeartDisease'] = preds
        results['PredictedHeartDisease'] = bulk_df['PredictedHeartDisease'].replace({1:"Yes",0: "No"})
        
        return results