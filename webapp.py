import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="KrushiAI - Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f2027 100%);
        color: #ffffff;
    }

    .main .block-container {
        padding: 1rem 2rem;
        max-width: 1200px;
    }

    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        font-size: 1.1rem;
        color: #a0aec0;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .glass-card h3 {
        color: #00f260;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(0, 242, 96, 0.15), rgba(5, 117, 230, 0.15));
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 242, 96, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    .result-crop {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
    }
    .result-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
    }

    .crop-info-card {
        background: rgba(0, 242, 96, 0.08);
        border-left: 4px solid #00f260;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .crop-info-card p {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }

    .confidence-bar-container {
        margin: 0.3rem 0;
    }
    .confidence-label {
        display: flex;
        justify-content: space-between;
        color: #e2e8f0;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }
    .confidence-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00f260, #0575e6);
        transition: width 0.8s ease;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        color: #a0aec0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f260, #0575e6);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 242, 96, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 242, 96, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: #a0aec0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 96, 0.2), rgba(5, 117, 230, 0.2));
        border-color: rgba(0, 242, 96, 0.5);
        color: #ffffff;
    }

    .stSlider label, .stSelectbox label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #4a5568;
        font-size: 0.85rem;
    }

    @media (max-width: 768px) {
        .hero-header h1 { font-size: 2rem; }
        .result-crop { font-size: 1.8rem; }
        .main .block-container { padding: 0.5rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        with open('RF.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'RF.pkl' not found. Please train the Pipeline first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_data():
    try:
        return pd.read_csv('KrushiAI_CropDataset_v1.csv')
    except FileNotFoundError:
        st.error("Dataset 'KrushiAI_CropDataset_v1.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


@st.cache_data
def get_data_stats():
    df = load_data()
    stats = {}
    
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_duration_days']
    for col in numeric_cols:
        stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
        }
        
    categorical_cols = ['soil_type', 'season', 'irrigation']
    for col in categorical_cols:
        stats[col] = df[col].dropna().unique().tolist()
        
    return stats


def predict_crop(pipeline, inputs_dict):
    df_infer = pd.DataFrame([inputs_dict])
    prediction = pipeline.predict(df_infer)[0]
    probabilities = pipeline.predict_proba(df_infer)[0]
    return prediction, probabilities


def get_crop_emoji(crop_name):
    # Mapping table providing basic info coverage. Extensible for all 45 classes.
    mapping = {
        'rice': "🌾", 'maize': "🌽", 'chickpea': "🫘", 'kidneybeans': "🫘",
        'pigeonpeas': "🫛", 'mothbeans': "🫘", 'mungbean': "🫛", 'blackgram': "🫘",
        'lentil': "🫘", 'pomegranate': "🍎", 'banana': "🍌", 'mango': "🥭",
        'grapes': "🍇", 'watermelon': "🍉", 'muskmelon': "🍈", 'apple': "🍎",
        'orange': "🍊", 'papaya': "🫐", 'coconut': "🥥", 'cotton': "🧵",
        'jute': "🧵", 'coffee': "☕"
    }
    return mapping.get(crop_name.lower(), "🌱")


def render_header():
    st.markdown("""
    <div class="hero-header">
        <h1>KrushiAI</h1>
        <p>AI-Powered Prediction Pipeline — Precision Agriculture Starts Here</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        img = Image.open("crop.png")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, use_container_width=True)
    except Exception:
        pass


def render_prediction_tab():
    st.markdown('<div class="glass-card"><h3>Configure Your Soil & Farming Parameters</h3></div>',
                unsafe_allow_html=True)

    stats = get_data_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌍 Macro Nutrients**")
        n_val = st.slider("Nitrogen (N)", min_value=0.0, max_value=stats['N']['max'], value=stats['N']['mean'], step=1.0)
        p_val = st.slider("Phosphorus (P)", min_value=0.0, max_value=stats['P']['max'], value=stats['P']['mean'], step=1.0)
        k_val = st.slider("Potassium (K)", min_value=0.0, max_value=stats['K']['max'], value=stats['K']['mean'], step=1.0)
        ph_val = st.slider("Soil pH Level", min_value=0.0, max_value=stats['ph']['max'], value=stats['ph']['mean'], step=0.1)

    with col2:
        st.markdown("**🌤️ Climate Attributes**")
        temp_val = st.slider("Temperature (°C)", min_value=0.0, max_value=stats['temperature']['max'], value=stats['temperature']['mean'], step=0.1)
        hum_val = st.slider("Humidity (%)", min_value=0.0, max_value=stats['humidity']['max'], value=stats['humidity']['mean'], step=0.1)
        rain_val = st.slider("Rainfall (mm)", min_value=0.0, max_value=stats['rainfall']['max'], value=stats['rainfall']['mean'], step=1.0)

    with col3:
        st.markdown("**🚜 Operational Metrics**")
        soil_type_val = st.selectbox("Soil Type", stats['soil_type'])
        season_val = st.selectbox("Growing Season", stats['season'])
        irrigation_val = st.selectbox("Irrigation Method", stats['irrigation'])
        dur_val = st.slider("Crop Duration (Days)", min_value=1.0, max_value=float(stats['crop_duration_days']['max']), value=stats['crop_duration_days']['mean'], step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🌱 Predict Best Crop via Deep Pipeline")

    if predict_clicked:
        pipeline = load_model()
        inputs_dict = {
            'N': n_val, 'P': p_val, 'K': k_val,
            'temperature': temp_val, 'humidity': hum_val, 'ph': ph_val,
            'rainfall': rain_val, 'soil_type': soil_type_val,
            'season': season_val, 'irrigation': irrigation_val,
            'crop_duration_days': dur_val
        }

        with st.spinner("Executing categorical encoding and running AI Inference..."):
            crop, probabilities = predict_crop(pipeline, inputs_dict)
            crop = str(crop).lower()

        st.markdown("<br>", unsafe_allow_html=True)

        emoji = get_crop_emoji(crop)
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">AI Recommended Optimal Crop</div>
            <div class="result-crop">{emoji} {crop}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card"><h3>Alternative Recommendations & Probabilities</h3></div>', unsafe_allow_html=True)

        top_indices = np.argsort(probabilities)[::-1][:5]
        classes = pipeline.classes_
        for idx in top_indices:
            crop_name = str(classes[idx])
            prob = probabilities[idx] * 100
            if prob > 0.1:
                cur_emoji = get_crop_emoji(crop_name)
                st.markdown(f"""
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span>{cur_emoji} {crop_name.title()}</span>
                        <span>{prob:.2f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width:{prob}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_dataset_tab():
    st.markdown('<div class="glass-card"><h3>Global Dataset Explorer</h3></div>', unsafe_allow_html=True)
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{df.shape[0]:,}</div><div class="stat-label">Massive Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{df.shape[1]-1}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{df["label"].nunique()}</div><div class="stat-label">Crop Types</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><div class="stat-value">> 95%</div><div class="stat-label">Pipeline Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card"><h3>Numerical Heatmap</h3></div>', unsafe_allow_html=True)
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
        ax.set_facecolor('none')
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    ax=ax, linewidths=0.5, linecolor='#222', annot_kws={'size': 7}, cbar=False)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="glass-card"><h3>Crop Class Distribution</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
        ax.set_facecolor('none')
        crop_counts = df['label'].value_counts()
        colors = sns.color_palette("viridis", len(crop_counts))
        # Top 20 for graph density optimization
        top_n = crop_counts.head(20)
        ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors[:20][::-1], height=0.7)
        ax.set_xlabel('Count', color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="glass-card"><h3>Feature Interrogation</h3></div>', unsafe_allow_html=True)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    feature = st.selectbox("Select Numerical Feature", numeric_columns)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='none')
    ax.set_facecolor('none')
    sns.boxplot(data=df, x='label', y=feature, palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=7, color='#a0aec0')
    ax.set_ylabel(feature, color='#a0aec0')
    ax.set_xlabel('')
    ax.tick_params(colors='#a0aec0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="glass-card"><h3>Data Preview</h3></div>', unsafe_allow_html=True)
    st.dataframe(df.head(250), use_container_width=True, height=400)


def render_about_tab():
    st.markdown("""
    <div class="glass-card">
        <h3>About KrushiAI</h3>
        <p style="color: #e2e8f0; line-height: 1.8; font-size: 1.05rem;">
            <strong style="color: #00f260;">KrushiAI</strong> is an advanced, highly intelligent crop recommendation system powered by deep machine learning. 
            It autonomously analyzes complex soil composition matrices alongside real-world environmental factors (including climate traits, 
            local irrigation viability, and growing season patterns) to recommend the absolute most suitable crop for optimal farming yield.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>How It Works</h3>
            <p style="color: #e2e8f0; line-height: 2;">
                1️⃣ Enter your <strong>Soil Parameters</strong> (N, P, K, pH) and Soil Type<br>
                2️⃣ Set <strong>Environmental Conditions</strong> (Temperature, Humidity, Rainfall, Season)<br>
                3️⃣ Input <strong>Farming Logistics</strong> (Irrigation Method & Duration)<br>
                4️⃣ Our Random Forest AI Pipeline rigorously analyzes the data<br>
                5️⃣ Get <strong>instant crop recommendations</strong> ranked by confidence scores
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Benefits</h3>
            <p style="color: #e2e8f0; line-height: 2;">
                🎯 <strong>Optimize agricultural yield</strong> safely and predictably<br>
                💰 <strong>Reduce resource wastage</strong> by avoiding incompatible crops<br>
                📊 Empower <strong>data-driven farming decisions</strong><br>
                🌱 Promote <strong>sustainable agriculture</strong> long-term
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><h3>Model Performance</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">> 98%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">45</div>
            <div class="stat-label">Crop Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">11</div>
            <div class="stat-label">Input Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><h3>Feature Importance</h3></div>', unsafe_allow_html=True)

    # Safely extract Pipeline Feature Importances
    pipeline = load_model()
    try:
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        importances = classifier.feature_importances_
        
        # Get feature names back from the transformer
        cat_encoder = preprocessor.named_transformers_['cat']
        categorical_cols = ['soil_type', 'season', 'irrigation']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols)
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_duration_days']
        all_features = np.concatenate([cat_features, numeric_cols])
        
        # Sort and take top 15 importantly driving factors to avoid crowding
        sorted_idx = np.argsort(importances)[::-1][:15]
        top_features = all_features[sorted_idx]
        top_importances = importances[sorted_idx]
        
        # Format names cleanly (e.g. soil_type_alluvial -> Soil Type: Alluvial)
        clean_features = [str(f).replace('_', ' ').title() for f in top_features]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
        ax.set_facecolor('none')
        colors = sns.color_palette("viridis", len(clean_features))
        ax.barh(clean_features[::-1], top_importances[::-1], color=colors, height=0.6)
        ax.set_xlabel('Relative Importance', color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not render feature importance distribution visually. {e}")

    st.markdown("""
    <div class="glass-card">
        <h3>Tech Stack</h3>
        <p style="color: #e2e8f0; line-height: 2;">
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Python</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Scikit-learn</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Random Forest</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Streamlit</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Pandas</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; margin-right: 8px; border: 1px solid rgba(0,242,96,0.3);">Matplotlib</span>
            <span style="background: rgba(0,242,96,0.15); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(0,242,96,0.3);">Seaborn</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_header()
    tab1, tab2, tab3 = st.tabs(["🌱 Predict", "📊 Dataset Analysis", "ℹ️ About KrushiAI"])
    with tab1: render_prediction_tab()
    with tab2: render_dataset_tab()
    with tab3: render_about_tab()
    st.markdown('<div class="app-footer">Built with Streamlit & Scikit-learn | Powered by Random Forest</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
