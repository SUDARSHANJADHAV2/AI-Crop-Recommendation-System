import streamlit as st
import json
from PIL import Image
from ui import main_ui

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

    load_css('style.css')
    crop_info = load_crop_info('crop_info.json')

    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            img = Image.open("crop.png")
            st.image(img, width=150)
        except:
            st.write("🌱")

    with col2:
        st.markdown("<h1 style='text-align: left;'>KrushiAI: Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

    main_ui(crop_info)

if __name__ == '__main__':
    main()