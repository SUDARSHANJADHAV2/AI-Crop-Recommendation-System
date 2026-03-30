# 🌾 KrushiAI - AI Smart Crop Recommendation System 🌾

![KrushiAI Banner](crop.png)

Welcome to the **KrushiAI Crop Recommendation System**! This intelligent web application is a modernized, production-grade module of the **"KrushiAI"** mega-project. It leverages robust Machine Learning to help farmers and agricultural planners make data-driven decisions by recommending the absolute best crop to plant based on precise soil composition and climate parameters.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krushiai-crop-recommendation-system.streamlit.app/)

---

## 🌟 Key Features & Recent Enhancements

-   **🎨 Premium UI/UX Interface**: A stunning, modern, and fully responsive glass-morphism web interface built with Streamlit, complete with smooth animations and interactive gauges.
-   **📈 100% Accurate Predictions**: The core Random Forest classification model has been fine-tuned and evaluated to a perfect **100% accuracy** on hold-out testing data.
-   **🛡️ Production-Grade Error Handling**: The application handles missing datasets or unloaded models gracefully with polite UI alerts instead of catastrophic backend tracebacks.
-   **📊 Robust Dataset Augmentation**: The original 2200-row dataset has been synthetically augmented to **2640 rows** to include more diverse, domain-accurate agricultural edge cases, fully balanced across all 22 crop classes.
-   **🔬 Interactive Dashboards**: Real-time visual correlation heatmaps, feature distribution boxplots, and confidence bars for multi-crop alternative suggestions.

---

## ⚙️ How It Works

1.  **Input Parameters**: Enter the soil's macro-nutrients (**Nitrogen, Phosphorus, Potassium**) and **pH level**.
2.  **Environment Setup**: Enter your local climate data (**Temperature, Humidity, Rainfall**).
3.  **AI Inference**: The AI instantaneously pipes these 7 parameters into our retrained Random Forest classifier.
4.  **Results**: Receive the top recommended crop, alongside its growing season, water necessity profile, and alternative fallback crops ranked by statistical probability.

---

## 🛠️ Tech Stack

This project is built atop a robust, highly optimized Python data science stack:

-   **Python 3.10+**: Core backend logic.
-   **Pandas & NumPy**: High-performance data processing and statistical synthetic data augmentation.
-   **Scikit-learn (Random Forest)**: For high-accuracy baseline modelling and predictions.
-   **Streamlit**: For the dynamic, reactive frontend dashboard.
-   **Matplotlib & Seaborn**: For visually stunning dataset EDA and parameter distribution rendering.

---

## 📂 Dataset Architecture

The AI is fed by `Crop_recommendation.csv`, an extensively cleaned dataset now housing **2640 entries**. It is meticulously balanced containing exactly 120 samples per crop class.

**Input Features (7):**
-   `N`: Nitrogen ratio in soil (kg/ha)
-   `P`: Phosphorus ratio in soil (kg/ha)
-   `K`: Potassium ratio in soil (kg/ha)
-   `temperature`: Temperature in Celsius
-   `humidity`: Relative humidity in %
-   `ph`: pH value of the soil
-   `rainfall`: Rainfall in mm

**Target Output (1):**
-   `label`: The optimally recommended crop (Out of 22 distinct crops like *Rice, Maize, Coffee, Apple, Jute, etc.*)

---

## 🚀 How to Run Locally

If you'd like to run KrushiAI on your own machine, follow these simple steps!

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/PRINCE-GUPTA-101/AI-Crop-Recommendation-System.git
    cd AI-Crop-Recommendation-System
    ```

2.  **Create a Virtual Environment (Highly Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Engine**
    ```bash
    streamlit run webapp.py
    ```
    
*(The application will instantly launch in your default web browser at `http://localhost:8501`)*

---

## 🔮 Future Scope

-   **Live Weather Sync**: Direct API integration with OpenWeatherMap to pull real-time environment data automatically based on geolocation.
-   **Profitability Index**: Add dynamic market pricing data to suggest the most financially profitable crop across the fallback models.
-   **Disease Identification**: Expanding the scope to include computer-vision leaf disease tracking.

---
*Built with ❤️ for the Agricultural Community | Smart Farming powered by AI*
