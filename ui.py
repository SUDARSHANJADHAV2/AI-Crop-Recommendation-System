import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import predict_crop, load_data
import json

def main_ui(crop_info):
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Info", "ℹ️ About"])

    with tab1:
        st.markdown("### Get Your Crop Recommendation")
        st.write("Enter your soil and climate parameters to get a personalized crop recommendation.")

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
                inputs = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                if not all(inputs) or any(x < 0 for x in inputs):
                    st.error("Please fill in all input fields with valid values before predicting.")
                else:
                    with st.spinner('Analyzing your parameters...'):
                        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                        recommended_crop = prediction[0].lower()

                        st.markdown(f"""
                        <div style="background-color:#2d5a2d; padding:20px; border-radius:10px; margin-bottom:20px; border: 2px solid #4CAF50; box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);">
                            <h3 style="color:#81C784; text-align:center; margin-bottom:10px; font-weight:bold;">🌱 Recommended Crop</h3>
                            <h2 style="color:#A5D6A7; text-align:center; text-transform:uppercase; font-size:2.5rem; margin:0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{prediction[0]}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        if recommended_crop in crop_info:
                            st.markdown("### Crop Information")
                            st.info(crop_info[recommended_crop])

                        st.markdown("### Parameter Importance")
                        param_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
                        param_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

                        plt.style.use('dark_background')
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a1a')
                        bars = ax.bar(param_names, param_values, color=['#1976D2', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4', '#3F51B5'])
                        ax.set_title('Your Input Parameters', color='white', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Value', color='white', fontweight='bold')
                        ax.set_facecolor('#2d2d2d')
                        ax.tick_params(colors='white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['top'].set_color('white')
                        ax.spines['right'].set_color('white')
                        ax.spines['left'].set_color('white')
                        plt.xticks(rotation=45, color='white')
                        plt.yticks(color='white')
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

        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.subheader("Crop Distribution")
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a1a')
        crop_counts = df['label'].value_counts()
        sns.barplot(x=crop_counts.index, y=crop_counts.values, ax=ax, palette='viridis')
        ax.set_facecolor('#2d2d2d')
        ax.set_title('Distribution of Crops in Dataset', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Crop Type', color='white', fontweight='bold')
        ax.set_ylabel('Number of Records', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.markdown("### About KrushiAI")
        st.markdown("""
        <div style="background-color: #2d2d2d; padding: 25px; border-radius: 12px; border: 2px solid #4CAF50; margin-bottom: 20px;">
            <p style="color: #ffffff; font-size: 18px; line-height: 1.8; margin-bottom: 25px;">
                🌾 <strong>KrushiAI</strong> is an intelligent crop recommendation system that uses machine learning
                to suggest the most suitable crops based on soil composition and environmental factors.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50; margin-bottom: 20px;">
            <h4 style="color: #81C784; margin-top: 0; margin-bottom: 15px;">🔬 How it works:</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 20])
        with col1:
            st.write("")
        with col2:
            st.markdown("**1.** The system analyzes your input parameters (N, P, K values, temperature, humidity, pH, and rainfall)")
            st.markdown("**2.** It processes this data through a trained Random Forest model")
            st.markdown("**3.** The model predicts the most suitable crop for your conditions")

        st.markdown("""
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50; margin: 20px 0;">
            <h4 style="color: #81C784; margin-top: 0; margin-bottom: 15px;">🎯 Benefits:</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 20])
        with col1:
            st.write("")
        with col2:
            st.markdown("• 🚀 **Optimize agricultural yield** by planting suitable crops")
            st.markdown("• 💰 **Reduce resource wastage** by avoiding unsuitable crop selections")
            st.markdown("• 📊 **Make data-driven farming decisions**")
            st.markdown("• 🌱 **Promote sustainable farming practices**")

        st.markdown("""
        <div style="background-color: #1a4a1a; padding: 15px; border-radius: 8px; text-align: center; margin-top: 30px; border: 1px solid #4CAF50;">
            <p style="color: #A5D6A7; font-size: 14px; margin: 0;">
                🛠️ Built with <strong>Streamlit</strong> and <strong>scikit-learn</strong> | 🧠 Powered by <strong>Random Forest</strong> algorithm
            </p>
        </div>
        """, unsafe_allow_html=True)