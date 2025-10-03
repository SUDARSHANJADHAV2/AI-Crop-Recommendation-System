import streamlit as st
import json
from PIL import Image
from ui import main_ui
from config import CROP_INFO_JSON_PATH, CROP_IMAGE_PATH, CSS_PATH

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_crop_info(file_path):
    with open(file_path) as f:
        return json.load(f)

def main():
    st.set_page_config(
        page_title="KrushiAI - Crop Recommendation System",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css(CSS_PATH)

    crop_info = load_crop_info(CROP_INFO_JSON_PATH)

    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            img = Image.open(CROP_IMAGE_PATH)
            st.image(img, width=150)
        except FileNotFoundError:
            st.write("🌱")

    with col2:
        st.markdown("<h1 style='text-align: left;'>KrushiAI: Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

    main_ui(crop_info)

if __name__ == '__main__':
    main()