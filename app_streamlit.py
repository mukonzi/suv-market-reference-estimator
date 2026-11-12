
import streamlit as st
import pandas as pd
import joblib

from feature_engineering import extract_model_family

st.set_page_config(
    page_title="SUV Market Reference Price Estimator",
    page_icon="🚙",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("customs_suv_xgboost_model.joblib")

model = load_model()

st.title("🚙 SUV Market Reference Price Estimator")

st.write(
    "An XGBoost-based decision-support tool for estimating "
    "South African market reference prices of pre-owned SUVs."
)

st.subheader("Vehicle Characteristics")

brand = st.text_input("Vehicle Brand", "Toyota")

vehicle_model = st.text_input(
    "Vehicle Model and Variant",
    "Fortuner 2.8 GD-6"
)

year = st.number_input(
    "Year of Manufacture",
    min_value=1990,
    max_value=2026,
    value=2020,
    step=1
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=1000000,
    value=73000,
    step=1000
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Diesel", "Petrol", "Hybrid", "Electric"]
)

engine_capacity = st.number_input(
    "Engine Capacity (Litres)",
    min_value=0.1,
    max_value=10.0,
    value=2.8,
    step=0.1
)

if st.button("Estimate Market Reference Price", type="primary"):

    model_family = extract_model_family(vehicle_model)

    input_data = pd.DataFrame([{
        "Brand": brand.strip(),
        "Model": vehicle_model.strip(),
        "Year": int(year),
        "Mileage": int(mileage),
        "Transmission": transmission,
        "Fuel_Type": fuel_type,
        "Model_Family": model_family,
        "Engine_capacity_litre": float(engine_capacity)
    }])

    input_data = input_data[
        model.feature_names_in_.tolist()
    ]

    prediction = float(model.predict(input_data)[0])

    st.success("Market reference price estimated")

    st.metric(
        "Estimated Market Reference Price",
        f"R {prediction:,.2f}"
    )

    st.caption(
        f"Automatically identified model family: {model_family}"
    )

    st.warning(
        "This estimate is based on historical South African "
        "vehicle market data. The model may be less reliable "
        "for newer vehicles or unfamiliar models. "
        "The predicted price is not a legally determined "
        "customs value and does not establish undervaluation."
    )

st.divider()

st.caption(
    "Research prototype | MSc Big Data Analytics | "
    "XGBoost Regression | Beitbridge Border"
)
