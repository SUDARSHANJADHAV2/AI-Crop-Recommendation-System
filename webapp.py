## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="KrushiAI - Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    h1, h2, h3 {
        color: #2E7D32;
    }
    .stSidebar {
        background-color: #E8F5E9;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Display header with logo
col1, col2 = st.columns([1, 3])
with col1:
    # Display Images
    from PIL import Image
    try:
        img = Image.open("crop.png")
        st.image(img, width=150)
    except:
        st.write("🌱")

with col2:
    st.markdown("<h1 style='text-align: left;'>KrushiAI: Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# Load the dataset for reference and display
@st.cache_data
def load_data():
    return pd.read_csv('Crop_recommendation.csv')

# Load the model
@st.cache_resource
def load_model():
    return pickle.load(open('RF.pkl', 'rb'))

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    model = load_model()
    prediction = model.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

# Dictionary with crop information
crop_info = {
    'rice': "Rice thrives in warm, humid conditions with abundant water. Ideal for lowland areas with good irrigation.",
    'maize': "Maize (corn) grows well in well-drained soils with moderate rainfall and warm temperatures.",
    'chickpea': "Chickpeas prefer cool, dry conditions and can tolerate drought. They enrich soil with nitrogen.",
    'kidneybeans': "Kidney beans need warm temperatures and moderate rainfall. They prefer well-drained, fertile soil.",
    'pigeonpeas': "Pigeon peas are drought-resistant and grow well in semi-arid regions with minimal rainfall.",
    'mothbeans': "Moth beans are extremely drought-tolerant and thrive in hot, dry conditions with minimal water.",
    'mungbean': "Mung beans prefer warm temperatures and moderate rainfall. They have a short growing season.",
    'blackgram': "Black gram thrives in warm, humid conditions and can tolerate some drought.",
    'lentil': "Lentils prefer cool growing conditions and moderate rainfall. They're adaptable to various soil types.",
    'pomegranate': "Pomegranates thrive in hot, dry climates and are drought-tolerant once established.",
    'banana': "Bananas need consistent warmth, high humidity, and abundant water. They're sensitive to frost.",
    'mango': "Mangoes require tropical conditions with a distinct dry season for flowering. They're frost-sensitive.",
    'grapes': "Grapes grow best in temperate climates with warm, dry summers and mild winters.",
    'watermelon': "Watermelons need hot temperatures, plenty of sunlight, and moderate water during growth.",
    'muskmelon': "Muskmelons require warm temperatures, full sun, and moderate, consistent moisture.",
    'apple': "Apples need a cold winter period for dormancy and moderate summers. They prefer well-drained soil.",
    'orange': "Oranges thrive in subtropical climates with mild winters and warm summers.",
    'papaya': "Papayas need consistent warmth and moisture. They're very frost-sensitive.",
    'coconut': "Coconuts require tropical conditions with high humidity, warm temperatures, and regular rainfall.",
    'cotton': "Cotton thrives in warm climates with long growing seasons and moderate rainfall.",
    'jute': "Jute needs warm, humid conditions with high rainfall during the growing season.",
    'coffee': "Coffee grows best in tropical highlands with moderate temperatures and regular rainfall."
}

## Streamlit code for the web app interface
def main():
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Info", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Get Your Crop Recommendation")
        st.write("Enter your soil and climate parameters to get a personalized crop recommendation.")
        
        # Create two columns for input and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Soil Parameters")
            nitrogen = st.number_input("🧪 Nitrogen (kg/ha)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
            phosphorus = st.number_input("🧪 Phosphorus (kg/ha)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
            potassium = st.number_input("🧪 Potassium (kg/ha)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)
            ph = st.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            
            st.subheader("Climate Parameters")
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
            humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
            
            predict_button = st.button("🔮 Predict Crop")
        
        with col2:
            st.subheader("Recommendation Results")
            if predict_button:
                inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
                if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
                    st.error("Please fill in all input fields with valid values before predicting.")
                else:
                    with st.spinner('Analyzing your parameters...'):
                        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                        recommended_crop = prediction[0].lower()
                        
                        # Display result with styling
                        st.markdown(f"""
                        <div style="background-color:#E8F5E9; padding:20px; border-radius:10px; margin-bottom:20px;">
                            <h3 style="color:#2E7D32; text-align:center;">Recommended Crop</h3>
                            <h2 style="color:#1B5E20; text-align:center; text-transform:uppercase;">{prediction[0]}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display crop information
                        if recommended_crop in crop_info:
                            st.markdown("### Crop Information")
                            st.info(crop_info[recommended_crop])
                        
                        # Display parameter importance visualization
                        st.markdown("### Parameter Importance")
                        
                        # Create a simple visualization of the input parameters
                        param_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
                        param_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(param_names, param_values, color=['#1976D2', '#388E3C', '#FBC02D', '#D32F2F', '#7B1FA2', '#0097A7', '#1565C0'])
                        ax.set_title('Your Input Parameters')
                        ax.set_ylabel('Value')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.info("Fill in the parameters and click 'Predict Crop' to get your recommendation.")
                
    with tab2:
        st.markdown("### Dataset Information")
        df = load_data()
        st.write("This application uses a dataset with the following characteristics:")
        st.write(f"- **Number of records**: {df.shape[0]}")
        st.write(f"- **Number of features**: {df.shape[1]-1}")
        st.write(f"- **Crop varieties**: {df['label'].nunique()}")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Show distribution of crops in the dataset
        st.subheader("Crop Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        crop_counts = df['label'].value_counts()
        sns.barplot(x=crop_counts.index, y=crop_counts.values, ax=ax)
        plt.xticks(rotation=90)
        plt.title('Distribution of Crops in Dataset')
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab3:
        st.markdown("### About KrushiAI")
        st.write("""
        KrushiAI is an intelligent crop recommendation system that uses machine learning to suggest the most suitable crops based on soil composition and environmental factors.
        
        **How it works:**
        1. The system analyzes your input parameters (N, P, K values, temperature, humidity, pH, and rainfall)
        2. It processes this data through a trained Random Forest model
        3. The model predicts the most suitable crop for your conditions
        
        **Benefits:**
        - Optimize agricultural yield by planting suitable crops
        - Reduce resource wastage by avoiding unsuitable crop selections
        - Make data-driven farming decisions
        
        This application was built using Streamlit and scikit-learn.
        """)

## Running the main function
if __name__ == '__main__':
    main()

