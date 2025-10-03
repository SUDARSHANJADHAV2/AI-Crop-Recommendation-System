import streamlit as st
import pandas as pd
import numpy as np
import pickle
from config import CROP_RECOMMENDATION_DATA_PATH, MODEL_PATH

@st.cache_data
def load_data():
    return pd.read_csv(CROP_RECOMMENDATION_DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as model_file:
        return pickle.load(model_file)

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    model = load_model()
    prediction = model.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction