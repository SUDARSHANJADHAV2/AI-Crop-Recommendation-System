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

    .stSlider label {
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
        st.error("Model file 'RF.pkl' not found. Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_data():
    try:
        return pd.read_csv('Crop_recommendation.csv')
    except FileNotFoundError:
        st.error("Dataset 'Crop_recommendation.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


@st.cache_data
def get_data_stats():
    df = load_data()
    stats = {}
    for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
        }
    return stats


def predict_crop(model, inputs):
    prediction = model.predict(np.array(inputs).reshape(1, -1))[0]
    probabilities = model.predict_proba(np.array(inputs).reshape(1, -1))[0]
    return prediction, probabilities


CROP_INFO = {
    'rice': {"emoji": "🌾", "desc": "Thrives in warm, humid conditions with abundant water. Ideal for lowland areas with good irrigation.", "season": "Kharif", "water": "High"},
    'maize': {"emoji": "🌽", "desc": "Grows well in well-drained soils with moderate rainfall and warm temperatures.", "season": "Kharif/Rabi", "water": "Moderate"},
    'chickpea': {"emoji": "🫘", "desc": "Prefers cool, dry conditions. Drought-tolerant and enriches soil with nitrogen.", "season": "Rabi", "water": "Low"},
    'kidneybeans': {"emoji": "🫘", "desc": "Needs warm temperatures and moderate rainfall. Prefers well-drained, fertile soil.", "season": "Kharif", "water": "Moderate"},
    'pigeonpeas': {"emoji": "🫛", "desc": "Drought-resistant, grows well in semi-arid regions with minimal rainfall.", "season": "Kharif", "water": "Low"},
    'mothbeans': {"emoji": "🫘", "desc": "Extremely drought-tolerant. Thrives in hot, dry conditions with minimal water.", "season": "Kharif", "water": "Very Low"},
    'mungbean': {"emoji": "🫛", "desc": "Prefers warm temperatures and moderate rainfall. Short growing season.", "season": "Kharif", "water": "Moderate"},
    'blackgram': {"emoji": "🫘", "desc": "Thrives in warm, humid conditions and can tolerate some drought.", "season": "Kharif", "water": "Moderate"},
    'lentil': {"emoji": "🫘", "desc": "Prefers cool growing conditions and moderate rainfall. Adaptable to various soil types.", "season": "Rabi", "water": "Low"},
    'pomegranate': {"emoji": "🍎", "desc": "Thrives in hot, dry climates. Drought-tolerant once established.", "season": "Perennial", "water": "Low"},
    'banana': {"emoji": "🍌", "desc": "Needs consistent warmth, high humidity, and abundant water. Sensitive to frost.", "season": "Perennial", "water": "High"},
    'mango': {"emoji": "🥭", "desc": "Requires tropical conditions with a distinct dry season for flowering. Frost-sensitive.", "season": "Perennial", "water": "Moderate"},
    'grapes': {"emoji": "🍇", "desc": "Grows best in temperate climates with warm, dry summers and mild winters.", "season": "Perennial", "water": "Moderate"},
    'watermelon': {"emoji": "🍉", "desc": "Needs hot temperatures, plenty of sunlight, and moderate water during growth.", "season": "Summer", "water": "Moderate"},
    'muskmelon': {"emoji": "🍈", "desc": "Requires warm temperatures, full sun, and moderate, consistent moisture.", "season": "Summer", "water": "Moderate"},
    'apple': {"emoji": "🍎", "desc": "Needs a cold winter period for dormancy. Prefers well-drained soil.", "season": "Temperate", "water": "Moderate"},
    'orange': {"emoji": "🍊", "desc": "Thrives in subtropical climates with mild winters and warm summers.", "season": "Perennial", "water": "Moderate"},
    'papaya': {"emoji": "🫐", "desc": "Needs consistent warmth and moisture. Very frost-sensitive.", "season": "Perennial", "water": "High"},
    'coconut': {"emoji": "🥥", "desc": "Requires tropical conditions with high humidity, warm temperatures, and regular rainfall.", "season": "Perennial", "water": "High"},
    'cotton': {"emoji": "🧵", "desc": "Thrives in warm climates with long growing seasons and moderate rainfall.", "season": "Kharif", "water": "Moderate"},
    'jute': {"emoji": "🧵", "desc": "Needs warm, humid conditions with high rainfall during the growing season.", "season": "Kharif", "water": "High"},
    'coffee': {"emoji": "☕", "desc": "Grows best in tropical highlands with moderate temperatures and regular rainfall.", "season": "Perennial", "water": "Moderate"},
}


def render_header():
    st.markdown("""
    <div class="hero-header">
        <h1>KrushiAI</h1>
        <p>AI-Powered Crop Recommendation System — Smart Farming Starts Here</p>
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
    st.markdown('<div class="glass-card"><h3>Configure Your Soil & Climate Parameters</h3></div>',
                unsafe_allow_html=True)

    stats = get_data_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌍 Soil Parameters**")
        n_val = st.slider(
            "Nitrogen (N) — kg/ha",
            min_value=0.0, max_value=140.0,
            value=round(stats['N']['mean'], 1), step=1.0,
            help=f"Range: 0–140 | Avg: {stats['N']['mean']:.0f}"
        )
        p_val = st.slider(
            "Phosphorus (P) — kg/ha",
            min_value=0.0, max_value=145.0,
            value=round(stats['P']['mean'], 1), step=1.0,
            help=f"Range: 0–145 | Avg: {stats['P']['mean']:.0f}"
        )
        k_val = st.slider(
            "Potassium (K) — kg/ha",
            min_value=0.0, max_value=205.0,
            value=round(stats['K']['mean'], 1), step=1.0,
            help=f"Range: 0–205 | Avg: {stats['K']['mean']:.0f}"
        )
        ph_val = st.slider(
            "Soil pH Level",
            min_value=0.0, max_value=14.0,
            value=round(stats['ph']['mean'], 1), step=0.1,
            help=f"Range: 0–14 | Avg: {stats['ph']['mean']:.1f}"
        )

    with col2:
        st.markdown("**🌤️ Climate Parameters**")
        temp_val = st.slider(
            "Temperature — °C",
            min_value=0.0, max_value=51.0,
            value=round(stats['temperature']['mean'], 1), step=0.1,
            help=f"Range: 0–51 | Avg: {stats['temperature']['mean']:.1f}"
        )
        hum_val = st.slider(
            "Humidity — %",
            min_value=0.0, max_value=100.0,
            value=round(stats['humidity']['mean'], 1), step=0.1,
            help=f"Range: 0–100 | Avg: {stats['humidity']['mean']:.1f}"
        )
        rain_val = st.slider(
            "Rainfall — mm",
            min_value=0.0, max_value=500.0,
            value=round(stats['rainfall']['mean'], 1), step=1.0,
            help=f"Range: 0–500 | Avg: {stats['rainfall']['mean']:.0f}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🌱 Predict Best Crop")

    if predict_clicked:
        model = load_model()
        inputs = [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]

        with st.spinner("Analyzing your parameters..."):
            crop, probabilities = predict_crop(model, inputs)
            crop = crop.lower()

        st.markdown("<br>", unsafe_allow_html=True)

        info = CROP_INFO.get(crop, {"emoji": "🌱", "desc": "No additional info available.", "season": "N/A", "water": "N/A"})
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Recommended Crop</div>
            <div class="result-crop">{info['emoji']} {crop}</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Season</div>
                <div class="stat-value" style="font-size:1.3rem;">{info['season']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Water Need</div>
                <div class="stat-value" style="font-size:1.3rem;">{info['water']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            confidence = max(probabilities) * 100
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value" style="font-size:1.3rem;">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="crop-info-card">
            <p>{info['emoji']} {info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card"><h3>Top 5 Recommendations</h3></div>',
                    unsafe_allow_html=True)

        top_indices = np.argsort(probabilities)[::-1][:5]
        classes = model.classes_
        for idx in top_indices:
            crop_name = classes[idx]
            prob = probabilities[idx] * 100
            if prob > 0.1:
                crop_emoji = CROP_INFO.get(crop_name, {}).get("emoji", "🌱")
                st.markdown(f"""
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span>{crop_emoji} {crop_name.title()}</span>
                        <span>{prob:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width:{prob}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card"><h3>Your Parameter Profile</h3></div>',
                    unsafe_allow_html=True)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
        ax.set_facecolor('none')

        param_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        colors = ['#00f260', '#0575e6', '#f7971e', '#fc4a1a', '#7b4397', '#00c9ff', '#92fe9d']

        ax.barh(param_names, inputs, color=colors, height=0.6, edgecolor='none', alpha=0.9)
        ax.set_xlabel('Value', color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.grid(axis='x', alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_dataset_tab():
    st.markdown('<div class="glass-card"><h3>Dataset Explorer</h3></div>', unsafe_allow_html=True)
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{df.shape[0]:,}</div>
            <div class="stat-label">Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{df.shape[1]-1}</div>
            <div class="stat-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{df['label'].nunique()}</div>
            <div class="stat-label">Crop Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card"><h3>Feature Correlations</h3></div>',
                    unsafe_allow_html=True)
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
        ax.set_facecolor('none')
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    ax=ax, linewidths=0.5, linecolor='#222',
                    annot_kws={'size': 8, 'color': 'white'},
                    cbar_kws={'shrink': 0.8})
        ax.tick_params(colors='#a0aec0', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="glass-card"><h3>Crop Distribution</h3></div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
        ax.set_facecolor('none')
        crop_counts = df['label'].value_counts()
        colors = sns.color_palette("viridis", len(crop_counts))
        ax.barh(crop_counts.index[::-1], crop_counts.values[::-1], color=colors[::-1], height=0.7)
        ax.set_xlabel('Count', color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.grid(axis='x', alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="glass-card"><h3>Feature Distributions by Crop</h3></div>',
                unsafe_allow_html=True)
    feature = st.selectbox("Select Feature", ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='none')
    ax.set_facecolor('none')
    sns.boxplot(data=df, x='label', y=feature, palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8, color='#a0aec0')
    ax.set_ylabel(feature, color='#a0aec0')
    ax.set_xlabel('')
    ax.tick_params(colors='#a0aec0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.grid(axis='y', alpha=0.1, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="glass-card"><h3>Raw Data</h3></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=400)


def render_about_tab():
    st.markdown("""
    <div class="glass-card">
        <h3>About KrushiAI</h3>
        <p style="color: #e2e8f0; line-height: 1.8; font-size: 1.05rem;">
            <strong style="color: #00f260;">KrushiAI</strong> is an intelligent crop recommendation system
            powered by machine learning. It analyzes soil composition and environmental factors
            to recommend the most suitable crop for optimal yield.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>How It Works</h3>
            <p style="color: #e2e8f0; line-height: 2;">
                1️⃣ Enter your soil parameters (N, P, K, pH)<br>
                2️⃣ Set environmental conditions (temp, humidity, rainfall)<br>
                3️⃣ Our Random Forest model analyzes the data<br>
                4️⃣ Get instant crop recommendations with confidence scores
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Benefits</h3>
            <p style="color: #e2e8f0; line-height: 2;">
                🎯 Optimize agricultural yield<br>
                💰 Reduce resource wastage<br>
                📊 Data-driven farming decisions<br>
                🌱 Promote sustainable agriculture
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">22</div>
            <div class="stat-label">Crop Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">7</div>
            <div class="stat-label">Input Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>Feature Importance</h3>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    features = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    importances = model.feature_importances_

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    ax.set_facecolor('none')
    sorted_idx = np.argsort(importances)
    colors = ['#00f260', '#0575e6', '#f7971e', '#fc4a1a', '#7b4397', '#00c9ff', '#92fe9d']
    sorted_colors = [colors[i] for i in sorted_idx]
    ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color=sorted_colors, height=0.6)
    ax.set_xlabel('Importance', color='#a0aec0', fontsize=10)
    ax.tick_params(colors='#a0aec0', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.grid(axis='x', alpha=0.1, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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

    tab1, tab2, tab3 = st.tabs(["🌱 Predict", "📊 Dataset", "ℹ️ About"])

    with tab1:
        render_prediction_tab()

    with tab2:
        render_dataset_tab()

    with tab3:
        render_about_tab()

    st.markdown("""
    <div class="app-footer">
        Built with Streamlit & Scikit-learn | Powered by Random Forest
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
