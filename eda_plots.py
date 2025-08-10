import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('unibo-powertools-dataset/unibo-powertools-dataset/test_result_trial_end_cleaned_v1.0.csv')
    return df

df = load_data()

st.header('Dataset Visualization')

# Pairplot Section
st.subheader('Pairplot of Key Variables')
pairplot_cols = ['SOC', 'max_temperature', 'voltage', 'current', 'time']
available_cols = [col for col in pairplot_cols if col in df.columns]

if len(available_cols) >= 2:
    sample_df = df[available_cols].sample(n=min(5000, len(df)), random_state=42)
    fig = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha':0.5})
    st.pyplot(fig)
else:
    st.warning('Not enough relevant columns for pairplot.')

# Temperature Distribution Section
st.subheader('Distribution of Max Temperature')
if 'max_temperature' in df.columns:
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.histplot(df['max_temperature'], bins=30, kde=True, ax=ax2)
    ax2.set_xlabel('Max Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Max Temperature')
    st.pyplot(fig2)
else:
    st.warning('max_temperature column not found.')
