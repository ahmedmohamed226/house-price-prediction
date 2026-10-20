import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("Predict a house price from its features (trained on Kaggle House Data).")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "house_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return model, scaler, features

try:
    model, scaler, features = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `notebooks/house_price_model.ipynb` first "
             "to generate `model_artifacts/` (house_model.pkl, scaler.pkl, features.pkl).")
    st.stop()

st.sidebar.header("House Features")
bedrooms   = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms  = st.sidebar.slider("Bathrooms", 0.5, 6.0, 2.0, 0.5)
sqft_liv   = st.sidebar.number_input("Sqft Living", min_value=300, max_value=15000, value=1500, step=50)
sqft_lot   = st.sidebar.number_input("Sqft Lot", min_value=500, max_value=200000, value=5000, step=100)
floors     = st.sidebar.selectbox("Floors", [1, 1.5, 2, 2.5, 3], index=1)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view       = st.sidebar.slider("View (0-4)", 0, 4, 0)
condition  = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
sqft_above = st.sidebar.number_input("Sqft Above", min_value=300, max_value=10000, value=1300, step=50)
sqft_base  = st.sidebar.number_input("Sqft Basement", min_value=0, max_value=5000, value=0, step=50)
yr_built   = st.sidebar.number_input("Year Built", min_value=1900, max_value=2026, value=1990)
yr_reno    = st.sidebar.number_input("Year Renovated (0 = never)", min_value=0, max_value=2026, value=0)
city       = st.sidebar.text_input("City", "Seattle")
statezip   = st.sidebar.text_input("State ZIP", "WA 98103")

if st.button("Predict Price 💰", type="primary"):
    row = {f: 0 for f in features}
    mapping = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_liv,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "sqft_above": sqft_above,
        "sqft_basement": sqft_base, "yr_built": yr_built, "yr_renovated": yr_reno,
    }
    for k, v in mapping.items():
        if k in row:
            row[k] = v

    # الترميز ده بيتطابق مع اللي في النوتبوك — لو عايز دقة أعلى شغّل النوتبوك
    if "city_encoded" in row:
        row["city_encoded"] = hash(city) % 10000
    if "statezip_encoded" in row:
        row["statezip_encoded"] = hash(statezip) % 10000

    X = np.array([list(row[f] for f in features)]).reshape(1, -1)
    X_s = scaler.transform(X)
    pred = model.predict(X_s)[0]

    st.success(f"💵 Estimated Price: **${pred:,.0f}**")
    st.balloons()
