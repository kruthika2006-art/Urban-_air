import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AQI Predictor",
    layout="wide"
)

st.title("🌍 AQI Analytics & Prediction")

uploaded_file = st.file_uploader(
    "Upload AQI Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    st.subheader("Columns Found")
    st.write(df.columns.tolist())
