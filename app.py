import streamlit as st
import pandas as pd
import plotly.express as px
from ml_pipeline import train_model

st.set_page_config(
    page_title="AQI Predictor",
    layout="wide"
)

st.title("🌍 Air Quality Analytics & Prediction")

uploaded_file = st.file_uploader(
    "Upload AQI Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Detected Columns")
        st.write(df.columns.tolist())

        # Check required columns
        required_columns = [
            "state",
            "aqi_value",
            "number_of_monitoring_stations"
        ]

        st.subheader("Actual Columns in Dataset")
st.write(df.columns.tolist())

st.stop()

        # Convert AQI to numeric
        df["aqi_value"] = pd.to_numeric(
            df["aqi_value"],
            errors="coerce"
        )

        df = df.dropna(subset=["aqi_value"])

        # AQI Distribution
        st.subheader("AQI Distribution")

        fig = px.histogram(
            df,
            x="aqi_value",
            nbins=30,
            title="AQI Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # State-wise AQI
        st.subheader("State-wise Average AQI")

        state_aqi = (
            df.groupby("state")["aqi_value"]
            .mean()
            .reset_index()
            .sort_values(
                "aqi_value",
                ascending=False
            )
        )

        fig2 = px.bar(
            state_aqi,
            x="state",
            y="aqi_value",
            title="Average AQI by State"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # Machine Learning
        st.subheader("Machine Learning Model")

        score, model = train_model(df)

        st.success(
            f"Model R² Score: {score:.2f}"
        )

        # Prediction Section
        st.subheader("Predict AQI")

        stations = st.number_input(
            "Number of Monitoring Stations",
            min_value=1,
            value=10
        )

        if st.button("Predict AQI"):

            prediction = model.predict(
                [[stations]]
            )[0]

            st.metric(
                "Predicted AQI",
                round(float(prediction), 2)
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
