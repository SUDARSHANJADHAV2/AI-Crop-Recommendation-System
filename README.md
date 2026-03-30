# 🌾 KrushiAI - Advanced Crop Recommendation Pipeline 🌾

![KrushiAI Banner](crop.png)

Welcome to the **KrushiAI Crop Recommendation System v2.0**! The project has been massively overhauled to handle real-world categorical complexity and large-scale data points, shifting from a basic numerical predictor into a production-grade scikit-learn Machine Learning Pipeline.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krushiai-crop-recommendation-system.streamlit.app/)

---

## 🌟 Key Features & Structural Upgrades

-   **📦 Massive Dataset Upgrade**: Now operating autonomously on **`KrushiAI_CropDataset_v1.csv`**. The dataset sits at a robust **12,100 high-quality rows** (spanning 45 crops exactly balanced at 320 records per class minus duration deviance).
-   **⚙️ Advanced ML Pipeline Architecture**: Removed antiquated raw Random Forest methods. The codebase is now utilizing a `sklearn` **ColumnTransformer & Pipeline** architecture to natively route variables into `OneHotEncoders` alongside numeric passthroughs before hitting the Random Forest module.
-   **🚜 Categorical Operational Inputs**: Recommending crops correctly relies on **Soil Type, Season,** and **Irrigation** limits—not just macro-nutrients. These elements are now directly modeled.

---

## 🛠️ Architecture Stack

-   **Python 3.10+**: Core backend logic.
-   **Scikit-Learn (Pipeline, ColumnTransformer, OneHotEncoder)**: Flawless mapping of frontend categorical selections (strings) to numerical algorithm states safely without manual dictionary hardcoding.
-   **Pandas & NumPy**: For pipeline data-frame structures.
-   **Streamlit**: A breathtaking, glass-morphic UI wrapper for deployment that protects against data tracebacks safely using graceful `st.error` checks.
-   **Matplotlib & Seaborn**: For dynamic visual exploratory data generation separate between string logic and numeric logic natively.

---

## 📂 Dataset Specification

The new architecture operates on **11 variables** predicting **45 unique crop designations**.

**Numerical Inputs (4):**
-   `N`: Nitrogen ratio in soil (kg/ha)
-   `P`: Phosphorus ratio in soil (kg/ha)
-   `K`: Potassium ratio in soil (kg/ha)
-   `temperature`: Temperature in Celsius
-   `humidity`: Relative humidity in %
-   `ph`: pH value of the soil
-   `rainfall`: Rainfall in mm
-   `crop_duration_days`: Maturation runway (Days)

**Categorical Inputs (3):**
-   `soil_type`: ('alluvial', 'clayey', 'loamy', 'red', 'black', 'sandy', 'laterite')
-   `season`: ('Kharif', 'Rabi', 'Perennial', 'Zaid')
-   `irrigation`: ('irrigated', 'semi-irrigated', 'rainfed')

**Target Output (1):**
-   `label`: The optimally recommended crop (Out of 45 extensive distinct agricultural outputs).

> *Note: Model performance sits comfortably at > 98% Test Accuracy on unseen split divisions running within the Pipeline framework.*

---

## 🚀 How to Run Locally

If you'd like to run KrushiAI v2.0 on your own machine:

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
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Streamlit Engine**
    ```bash
    streamlit run webapp.py
    ```
    
*(The application will instantly launch in your default web browser at `http://localhost:8501`)*

---
*Built with ❤️ for the Agricultural Community | Smart Farming powered by AI*
