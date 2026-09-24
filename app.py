import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ----------------------------- Load assets -----------------------------
@st.cache_resource
def load_assets():
    df = pd.read_csv("car_price_dataset.csv")
    ct = joblib.load("ColumnTransformer.pkl")
    model = tf.keras.models.load_model("model.keras")

    # Recreate the PowerTransformer fit exactly like the training notebook
    # (fit on the full dataset, one column at a time, before the split)
    power_cols = ["Horsepower", "Doors", "Seats", "FuelEfficiency(L/100km)"]
    fitted_pts = {}
    for col in power_cols:
        pt = PowerTransformer()
        pt.fit(df[[col]])
        fitted_pts[col] = pt

    return df, ct, model, fitted_pts

df, ct, model, fitted_pts = load_assets()

BRANDS = sorted(df["Brand"].unique())
MODELS_BY_BRAND = df.groupby("Brand")["Model"].unique().apply(sorted).to_dict()
CONDITIONS = sorted(df["Condition"].unique())
FUEL_TYPES = sorted(df["FuelType"].unique())
TRANSMISSIONS = sorted(df["Transmission"].unique())
DRIVE_TYPES = sorted(df["DriveType"].unique())
BODY_TYPES = sorted(df["BodyType"].unique())
COLORS = sorted(df["Color"].unique())
INTERIORS = sorted(df["Interior"].unique())
CITIES = sorted(df["City"].unique())
OPTION_TAGS = ["Bluetooth", "Cruise Control", "Heated Seats", "Navigation",
               "Parking Sensors", "Rear Camera", "Sunroof", "Touchscreen"]

st.title("🚗 Car Price Predictor")
st.caption("Neural network model trained on the car price dataset (Test R² ≈ 0.983)")

with st.form("car_form"):
    st.subheader("Vehicle details")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", BRANDS)
        model_name = st.selectbox("Model", MODELS_BY_BRAND.get(brand, []))
        year = st.number_input("Year", min_value=2005, max_value=2025, value=2020)
        car_age = st.number_input("Car Age", min_value=0, max_value=20, value=2026 - 2020)
        condition = st.selectbox("Condition", CONDITIONS)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=320262, value=50000, step=1000)
        engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        horsepower = st.number_input("Horsepower", min_value=65, max_value=905, value=150)
        torque = st.number_input("Torque", min_value=16, max_value=850, value=200)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        drive_type = st.selectbox("Drive Type", DRIVE_TYPES)

    with col2:
        body_type = st.selectbox("Body Type", BODY_TYPES)
        doors = st.selectbox("Doors", [2, 3, 4, 5], index=2)
        seats = st.selectbox("Seats", [2, 4, 5, 6, 7], index=2)
        color = st.selectbox("Color", COLORS)
        interior = st.selectbox("Interior", INTERIORS)
        city = st.selectbox("City", CITIES)
        accident_history = st.selectbox("Accident History", ["No", "Yes"])
        insurance = st.selectbox("Insurance", ["Valid", "Expired"])
        registration_status = st.selectbox("Registration Status", ["Complete", "Incomplete"])
        fuel_eff = st.number_input("Fuel Efficiency (L/100km)", min_value=0.0, max_value=21.18, value=8.0, step=0.1)
        selected_options = st.multiselect("Options", OPTION_TAGS, default=["Bluetooth"])

    submitted = st.form_submit_button("Predict Price 💰")

if submitted:
    options_str = ", ".join(selected_options) if selected_options else ""

    row = pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,
        "Year": year,
        "CarAge": car_age,
        "Condition": condition,
        "Mileage(km)": mileage,
        "EngineSize(L)": engine_size,
        "FuelType": fuel_type,
        "Horsepower": horsepower,
        "Torque": torque,
        "Transmission": transmission,
        "DriveType": drive_type,
        "BodyType": body_type,
        "Doors": doors,
        "Seats": seats,
        "Color": color,
        "Interior": interior,
        "Options": options_str,
        "City": city,
        "AccidentHistory": accident_history,
        "Insurance": insurance,
        "RegistrationStatus": registration_status,
        "FuelEfficiency(L/100km)": fuel_eff,
    }])

    # Apply the same PowerTransformer used during training
    for col, pt in fitted_pts.items():
        row[[col]] = pt.transform(row[[col]])

    X_processed = ct.transform(row)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    pred = model.predict(X_processed, verbose=0).flatten()[0]
    pred = max(pred, 0)

    st.success(f"### Estimated Price: **${pred:,.0f}**")
    st.caption("Prediction from the trained neural network. Actual market price may vary.")
