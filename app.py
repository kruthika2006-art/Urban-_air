import streamlit as st
import pandas as pd
import plotly.express as px
from ml_pipeline import train_model

st.set_page_config(page_title="AQI Predictor", layout="wide")

st.title("🌍 Air Quality Analytics & Prediction")

uploaded_file = st.file_uploader(
    "Upload AQI Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("AQI Distribution")

    fig = px.histogram(
        df,
        x="aqi_value"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("State-wise Average AQI")

    state_aqi = (
        df.groupby("state")["aqi_value"]
        .mean()
        .reset_index()
        .sort_values("aqi_value", ascending=False)
    )

    fig2 = px.bar(
        state_aqi,
        x="state",
        y="aqi_value"
    )

    st.plotly_chart(fig2, use_container_width=True)

    score, model = train_model(df)

    st.success(
        f"Model Accuracy (R²): {score:.2f}"
    )

    st.subheader("Predict AQI")

    stations = st.number_input(
        "Number of Monitoring Stations",
        min_value=1,
        value=10
    )

    if st.button("Predict"):

        prediction = model.predict(
            [[stations]]
        )[0]

        st.metric(
            "Predicted AQI",
            round(prediction, 2)
        )
