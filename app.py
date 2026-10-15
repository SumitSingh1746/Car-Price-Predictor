import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    with open(BASE_DIR / "LinearRegressionModel.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "cleaned_car_data.csv")
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


model = load_model()
car = load_data()

# ---------- Header ----------
st.title("🚗 Used Car Price Predictor")
st.write(
    "Select the details of a used car and get an estimated resale price. "
    "The model is a Linear Regression trained on Quikr listings."
)

# ---------- Inputs ----------
companies = sorted(car["company"].unique())
company = st.selectbox("Company", companies)

# Only show car models that belong to the chosen company
names = sorted(car.loc[car["company"] == company, "name"].unique())
name = st.selectbox("Car model", names)

col1, col2 = st.columns(2)
with col1:
    year = st.slider(
        "Year of purchase",
        min_value=int(car["year"].min()),
        max_value=int(car["year"].max()),
        value=2015,
    )
with col2:
    fuel_type = st.selectbox("Fuel type", sorted(car["fuel_type"].unique()))

kms_driven = st.number_input(
    "Kilometres driven", min_value=0, max_value=1_000_000, value=30_000, step=1_000
)

# ---------- Prediction ----------
if st.button("Predict price", type="primary", use_container_width=True):
    # Column names and order must match what the pipeline was trained on
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )
    price = float(model.predict(input_df)[0])

    if price <= 0:
        st.warning(
            "The model returned a non-positive value for this combination. "
            "Linear Regression can behave oddly for rare inputs "
            "(for example, a very old car with very high mileage)."
        )
    else:
        st.success(f"Estimated price: ₹ {price:,.0f}")
        st.caption(
            f"{year} {name} · {fuel_type} · {kms_driven:,} km. "
            "This is an estimate, not a valuation."
        )

with st.expander("About this model"):
    st.write(
        f"- Trained on **{len(car)}** cleaned listings across "
        f"**{car['company'].nunique()}** companies.\n"
        "- Features: car model, company, year, kilometres driven, fuel type.\n"
        "- Prices are in Indian Rupees (₹)."
    )