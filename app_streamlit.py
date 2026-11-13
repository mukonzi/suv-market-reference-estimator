
import json
import re
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="SUV Market Price Estimator",
    page_icon="🚙",
    layout="centered"
)

@st.cache_resource
def load_resources():
    basic = joblib.load(
        "customs_friendly_suv_xgboost_optimized.pkl"
    )
    advanced = joblib.load(
        "general_suv_xgboost_final_v2.pkl"
    )
    with open("brand_model_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    return basic, advanced, mapping

basic_model, advanced_model, brand_models = load_resources()

def extract_model_family(description, brand):
    description = str(description).strip()
    description = re.sub(
        rf"^{re.escape(brand)}(?:\s+|$)",
        "",
        description,
        flags=re.IGNORECASE
    ).strip()

    for family in sorted(
        brand_models.get(brand, []),
        key=len,
        reverse=True
    ):
        pattern = (
            r"^" + re.escape(family) +
            r"(?=$|[\s,;/()\-])"
        )
        if re.match(pattern, description, flags=re.IGNORECASE):
            return family

    return None

def get_categories(model):
    preprocessor = model.named_steps["preprocessor"]
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            encoder = next(
                step for step in transformer.named_steps.values()
                if isinstance(step, OneHotEncoder)
            )
            return dict(zip(
                columns,
                [list(c) for c in encoder.categories_]
            ))
    raise ValueError("OneHotEncoder not found")

st.title("🚙 SUV Market Price Estimator")
st.write(
    "External market listing price estimation for "
    "pre-owned SUVs using XGBoost regression."
)

mode = st.radio(
    "Prediction mode",
    ["Basic — 8 features", "Advanced — 16 features"],
    horizontal=True
)

advanced = mode.startswith("Advanced")
model = advanced_model if advanced else basic_model
categories = get_categories(model)

st.subheader("Vehicle identification")

brand = st.selectbox(
    "Vehicle brand",
    sorted(brand_models.keys())
)

description = st.text_input(
    "Full vehicle model description",
    placeholder="e.g. Toyota Fortuner 2.8GD-6 4x2 AT"
)

detected_family = (
    extract_model_family(description, brand)
    if description.strip() else None
)

available_families = [
    family for family in brand_models[brand]
    if family in categories["Model_Family"]
]

if detected_family:
    st.success(f"Detected model family: {detected_family}")
elif description.strip():
    st.warning(
        "Model family could not be identified automatically. "
        "Please select it manually."
    )

default_index = (
    available_families.index(detected_family)
    if detected_family in available_families
    else 0
)

model_family = st.selectbox(
    "Model family (verify or correct)",
    available_families,
    index=default_index,
    key=f"family_{brand}_{detected_family}"
)

st.subheader("Vehicle specifications")

year = st.number_input(
    "Year", min_value=1990, max_value=2026, value=2020
)

mileage = st.number_input(
    "Mileage (km)", min_value=0, value=60000, step=1000
)

transmission = st.selectbox(
    "Transmission", categories["Transmission"]
)

fuel_type = st.selectbox(
    "Fuel type", categories["Fuel_Type"]
)

engine_capacity = st.number_input(
    "Engine capacity (litres)",
    min_value=0.1, max_value=10.0,
    value=2.8, step=0.1
)

color = st.selectbox(
    "Colour", categories["Color_Standard"]
)

vehicle = {
    "Brand": brand,
    "Model_Family": model_family,
    "Year": year,
    "Mileage": mileage,
    "Transmission": transmission,
    "Fuel_Type": fuel_type,
    "Engine_capacity_litre": engine_capacity,
    "Color_Standard": color
}

if advanced:
    st.subheader("Additional technical specifications")

    vehicle["Cylinder_layout_and_quantity"] = st.selectbox(
        "Cylinder layout",
        categories["Cylinder_layout_and_quantity"]
    )

    vehicle["Seats_quantity"] = st.number_input(
        "Seats", min_value=1, max_value=20, value=7
    )

    vehicle["No_of_doors"] = st.number_input(
        "Doors", min_value=1, max_value=10, value=5
    )

    vehicle["Driven_wheels"] = st.selectbox(
        "Driven wheels", categories["Driven_wheels"]
    )

    vehicle["Fuel_capacity"] = st.number_input(
        "Fuel capacity (litres)",
        min_value=1.0, value=80.0
    )

    vehicle["Fuel_Consumption"] = st.number_input(
        "Fuel consumption (L/100 km)",
        min_value=0.1, value=8.0
    )

    vehicle["Power"] = st.number_input(
        "Power (kW)", min_value=1.0, value=130.0
    )

    vehicle["Torque"] = st.number_input(
        "Torque (Nm)", min_value=1.0, value=450.0
    )

st.caption(
    "Confirm all vehicle specifications before predicting. "
    "Default values are examples, not automatically extracted data."
)

if st.button("Predict market price", type="primary"):

    input_df = pd.DataFrame([vehicle])

    required_features = list(
        model.named_steps["preprocessor"].feature_names_in_
    )
    input_df = input_df[required_features]

    prediction = float(model.predict(input_df)[0])

    st.metric(
        "Estimated advertised market price",
        f"R{prediction:,.2f}"
    )

    st.caption("Currency: South African rand (ZAR)")

    st.info(
        "This prediction is an external market benchmark "
        "for preliminary undervaluation risk assessment. "
        "It is not an official customs valuation or "
        "evidence of undervaluation or fraud."
    )

st.divider()
st.caption(
    "Research prototype | MSc Big Data Analytics | "
    "AutoTrader South Africa secondary dataset"
)
