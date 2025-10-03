import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_input_parameters(param_names, param_values):
    """
    Plots a bar chart of the user's input parameters.
    """
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

def plot_crop_distribution(df):
    """
    Plots the distribution of crops in the dataset.
    """
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