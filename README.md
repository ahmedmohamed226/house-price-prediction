# 🏠 House Price Prediction

Machine learning project to predict house prices from property features.

## 📊 Dataset
- [Kaggle – House Data (shree1992/housedata)](https://www.kaggle.com/datasets/shree1992/housedata)
- Downloaded at runtime with `kagglehub` (no manual download needed).

## 📁 Project Structure
```
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── house_price_model.ipynb
├── app/
│   └── streamlit_app.py
└── model_artifacts/   (generated after training — not committed)
    ├── house_model.pkl
    ├── scaler.pkl
    └── features.pkl
```

## 🧪 Methodology
1. EDA (distributions, correlations, missing values)
2. Cleaning (fix `sqft_above` dtype, drop irrelevant columns, remove outliers via 1%/99% quantiles)
3. Encoding categorical features (`city`, `statezip`)
4. Training: LinearRegression, RandomForest, GradientBoosting
5. Comparison with R² / MAE / RMSE → best model saved with joblib

## 🚀 How to Run

### 1) Train the model (Google Colab)
- Upload `notebooks/house_price_model.ipynb` to Colab
- Runtime → Run all
- Download the generated `model_artifacts/` folder (Colab: right-click folder → Download) and place it in the repo root

### 2) Run the app (local)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 📦 Artifacts (in `model_artifacts/`)
- `house_model.pkl` – trained model
- `scaler.pkl` – fitted StandardScaler
- `features.pkl` – feature column names
