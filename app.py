
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("bike_rental_model.pkl")

model = load_model()

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}
div.stButton > button {
    background-color: #16a085;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🚲 Bike Rental Demand Prediction")
st.write(
    "Predict daily bike rentals using environmental "
    "and seasonal information."
)

st.info(
    "Model: Tuned Gradient Boosting Regressor | "
    "Dataset: Bike Sharing Dataset"
)

st.subheader("Enter Daily Conditions")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Winter",
            2: "Spring",
            3: "Summer",
            4: "Fall"
        }[x]
    )

    yr = st.selectbox(
        "Year",
        options=[0, 1],
        format_func=lambda x: {
            0: "2011",
            1: "2012"
        }[x]
    )

    mnth = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(
            year=2024, month=x, day=1
        ).strftime("%B")
    )

    weekday = st.selectbox(
        "Weekday (0 = Sunday)",
        options=list(range(7))
    )

    holiday = st.selectbox(
        "Holiday?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    workingday = st.selectbox(
        "Working Day?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    weathersit = st.selectbox(
        "Weather Situation",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear / Partly Cloudy",
            2: "Mist / Cloudy",
            3: "Light Rain / Snow",
            4: "Heavy Rain / Snow / Fog"
        }[x]
    )

    temp = st.slider(
        "Normalized Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    atemp = st.slider(
        "Normalized Feeling Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    hum = st.slider(
        "Normalized Humidity",
        min_value=0.0,
        max_value=1.0,
        value=0.6
    )

    windspeed = st.slider(
        "Normalized Wind Speed",
        min_value=0.0,
        max_value=1.0,
        value=0.2
    )

st.caption(
    "Weather and temperature values use the dataset's "
    "normalized scale (0 to 1)."
)

if st.button("Predict Bike Rentals"):
    input_data = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(0, round(float(prediction)))

    st.success("Prediction completed!")
    st.metric(
        label="Predicted Daily Bike Rentals",
        value=f"{prediction:,} bikes"
    )

    st.caption(
        "This is a model estimate, not a guaranteed rental count."
    )

st.divider()
st.subheader("About This Project")
st.write(
    "This capstone project analyzes daily bike-sharing demand "
    "using environmental and seasonal features. Multiple "
    "regression models were compared before selecting the "
    "tuned Gradient Boosting model."
)
