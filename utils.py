import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_data
def load_data():
    return pd.read_csv('Crop_recommendation.csv')

@st.cache_resource
def load_model():
    with open('RF.pkl', 'rb') as model_file:
        return pickle.load(model_file)

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    model = load_model()
    prediction = model.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction