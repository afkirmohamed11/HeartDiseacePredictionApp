# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 2023

@author: Afkir mohamed
"""
from LogisticRegression import LogisticRegressionG 
import streamlit as st 
import pickle
import numpy as np
import pandas as pd
from PIL import Image 
from sklearn.preprocessing import StandardScaler


pickle_in = open("src/classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def predict_heart_diseace(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    scaler = StandardScaler()
    # Here we need the features (data) so that we can fit the scaler:
    data= pd.read_csv("heart.csv")
    features = data.drop(columns = 'target', axis=1)
    scaler.fit(features)  

    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    return prediction




image=Image.open("src/images/OIG_055726.jpg")
st.sidebar.info('The Heart Disease Prediction App utilizes user-provided health data and a logistic regression model to estimate the likelihood of heart disease. To view the dataset used in building the model, please click the option below the predict button.')
st.sidebar.image(image)

html_temp = """
<div style="background-color:red;padding:15px;">
<h2 style="color:white;text-align:center;">Heart Disease Prediction App</h2>
</div>
"""

st.markdown(html_temp,unsafe_allow_html=True) 
html_temp1= """ <div
< h2 style= "text-align:center; font-size: 25px;padding:20px"> Welcome to the Heart Disease Prediction App. Please enter your information!</h2> 
</div> """
st.markdown(html_temp1, unsafe_allow_html=True)

age = st.number_input("**Age**", min_value=0, value=0, format="%d")
sex = st.radio("**Sex**", options=["Male", "Female"])
sex = 1 if sex == "Male" else 0
cp = st.number_input("**Chest Pain (cp)**", min_value=0, value=0, format="%d")
trestbps = st.number_input("**Resting Blood Pressure (trestbps)**", min_value=0, value=0, format="%d")
chol = st.number_input("**Cholesterol (chol)**", min_value=0, value=0, format="%d")
fbs = st.number_input("**Fasting Blood Sugar (fbs)**", min_value=0, value=0, format="%d")
restecg = st.number_input("**Resting Electrocardiographic (restecg)**", min_value=0, value=0, format="%d")
thalach = st.number_input("**Maximum Heart Rate (thalach)**", min_value=0, value=0, format="%d")
exang = st.number_input("**Exercise Induced Angina (exang)**", min_value=0, value=0, format="%d")
oldpeak = st.number_input("**Oldpeak**", min_value=0, value=0, format="%d")
slope = st.number_input("**Slope**", min_value=0, value=0, format="%d")
ca = st.number_input("**Number of Major Vessels (ca)**", min_value=0, value=0, format="%d")
thal = st.number_input("**Thal**", min_value=0, value=0, format="%d")

button_style = """
<style>
.stButton>button {
    background-color: red	;
    color: white;
    padding: 15px;
    font-size: 30px;
}
</style>
"""

st.markdown(button_style, unsafe_allow_html=True)
result_style = """
        <style>
        .result-text {
            color:red;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        </style>
    """
result_style1 = """
        <style>
        .result-text {
            color:green;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        </style>
    """
if st.button("Predict"):
    result=predict_heart_diseace(age,sex,cp, trestbps,chol,fbs, restecg, thalach, exang, oldpeak,slope, ca, thal)
    if result == 1:
        st.markdown(result_style, unsafe_allow_html=True)
        st.markdown('<p class="result-text">You are diagnosed with heart disease</p>', unsafe_allow_html=True)
    else:
        st.markdown(result_style1, unsafe_allow_html=True)
        st.markdown('<p class="result-text">You are not diagnosed with heart disease</p>', unsafe_allow_html=True)

show_dataset = st.checkbox("Show Dataset")
if show_dataset:
    data=pd.read_csv("heart.csv")
    st.write(data)


    
    
    