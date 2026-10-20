# House Price Prediction

A small ML project that predicts house prices based on things like size, number of rooms, condition etc.

The data is the house data from Kaggle: https://www.kaggle.com/datasets/shree1992/housedata
It gets downloaded automatically with kagglehub when you run the notebook, so you don't need to download anything yourself.

## What I did

First I did some EDA and cleaned the data. The `sqft_above` column was loaded as text because of commas in the numbers, so I fixed that. I also dropped columns like `date`, `street` and `country` since they don't really help, and removed outliers by keeping only prices between the 1% and 99% quantiles.

Then I encoded `city` and `statezip` with LabelEncoder and trained 3 models: LinearRegression, RandomForest and GradientBoosting. RandomForest usually ends up being the best one.

## Project structure

```
house-price-prediction/
├── notebooks/
│   └── house_price_model.ipynb   # training notebook
├── app/
│   └── streamlit_app.py          # small web app to try the model
├── requirements.txt
└── .gitignore
```

## How to run

1. Open `notebooks/house_price_model.ipynb` in Google Colab and run all cells. This downloads the dataset, trains the models and saves the best one in a `model_artifacts` folder.
2. Download the `model_artifacts` folder from Colab and put it in the project root.
3. Install the requirements and run the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The `model_artifacts` folder is not pushed to GitHub because of its size, you generate it locally by running the notebook.

## Libraries used

pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, kagglehub, streamlit
