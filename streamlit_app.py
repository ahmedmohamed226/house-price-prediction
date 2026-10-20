import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="House Price Predictor")

st.title("House Price Prediction")
st.write("Enter the house details and get an estimated price.")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load():
    model = joblib.load(os.path.join(BASE, "model_artifacts", "house_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "model_artifacts", "scaler.pkl"))
    features = joblib.load(os.path.join(BASE, "model_artifacts", "features.pkl"))
    return model, scaler, features

try:
    model, scaler, features = load()
except FileNotFoundError:
    st.error("Couldn't find the model files. Run the notebook first to generate model_artifacts.")
    st.stop()

st.sidebar.header("House details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 6.0, 2.0, 0.5)
sqft_living = st.sidebar.number_input("Sqft living", 300, 15000, 1500, step=50)
sqft_lot = st.sidebar.number_input("Sqft lot", 500, 200000, 5000, step=100)
floors = st.sidebar.selectbox("Floors", [1, 1.5, 2, 2.5, 3], index=1)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.slider("View (0-4)", 0, 4, 0)
condition = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
sqft_above = st.sidebar.number_input("Sqft above", 300, 10000, 1300, step=50)
sqft_basement = st.sidebar.number_input("Sqft basement", 0, 5000, 0, step=50)
yr_built = st.sidebar.number_input("Year built", 1900, 2026, 1990)
yr_renovated = st.sidebar.number_input("Year renovated (0 = never)", 0, 2026, 0)

if st.button("Predict"):
    row = {f: 0 for f in features}
    values = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "sqft_above": sqft_above,
        "sqft_basement": sqft_basement, "yr_built": yr_built, "yr_renovated": yr_renovated,
    }
    for k, v in values.items():
        if k in row:
            row[k] = v

    X = np.array([[row[f] for f in features]])
    pred = model.predict(scaler.transform(X))[0]
    st.success(f"Estimated price: ${pred:,.0f}")
