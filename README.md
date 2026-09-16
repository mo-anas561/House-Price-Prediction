# 🏠 House Price Prediction — Linear Regression

An end-to-end regression project that predicts house prices from structural,
locational, and amenity features. Built to demonstrate a complete, production-aware
ML workflow — not just a `.fit()` call — for portfolio and interview purposes.

## Results

| Model | RMSE | MAE | R² | Adj. R² | 5-fold CV R² |
|---|---|---|---|---|---|
| **Linear Regression** | 337,511 | 268,620 | **0.875** | 0.871 | 0.868 ± 0.005 |
| Ridge (α=1.0) | 337,485 | 268,588 | 0.875 | 0.871 | 0.868 ± 0.005 |
| Lasso (α=100) | 337,483 | 268,594 | 0.875 | 0.871 | 0.868 ± 0.005 |

Plain Linear Regression performs essentially identically to its regularized
counterparts — evidence the model isn't overfitting and multicollinearity isn't
a serious issue, so we keep the more interpretable plain model as the final choice.

## Project Structure
```
house_price_project/
├── data/
│   ├── housing.csv              # the dataset (2,200 rows, 15 columns)
│   └── generate_dataset.py      # reproducible data generator (swap for real data easily)
├── notebooks/
│   └── House_Price_Prediction.ipynb   # full narrative walkthrough, ready to present
├── train_model.py                # standalone script version of the full pipeline
├── models/
│   └── house_price_model.pkl     # saved, ready-to-load trained pipeline
├── outputs/                      # all generated charts + result tables (PNG/CSV)
└── README.md
```

## Dataset
14 features + target (`price`) for 2,200 houses:

- **Numeric:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `age`, `dist_city_center_km`
- **Categorical (yes/no):** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
- **Categorical:** `furnishingstatus` (furnished / semi-furnished / unfurnished)

A small amount of missing data is intentionally included (like real datasets)
to force proper imputation handling rather than a silent `dropna()`.

> This is a synthetic-but-realistic dataset generated with a controlled
> price-generating formula plus noise — swap `data/housing.csv` for any real
> housing dataset (e.g. Kaggle's Housing Prices) with the same column names
> and the whole pipeline runs unchanged.

## What this project demonstrates

- **Data cleaning:** explicit missing-value handling via `SimpleImputer`
  (median for numeric, most-frequent for categorical) — inside the pipeline,
  not a manual one-off step.
- **No data leakage:** `ColumnTransformer` + `Pipeline` ensures every
  transformation (imputation, scaling, encoding) is fit **only on the
  training data**.
- **Model comparison:** Linear Regression vs Ridge vs Lasso, to check for
  overfitting/multicollinearity before committing to the simplest model.
- **Full evaluation suite:** RMSE, MAE, R², Adjusted R², and 5-fold
  cross-validation — not just a single train/test split.
- **Assumption checking:** residuals-vs-predicted (homoscedasticity) and
  residual distribution (normality) plots, not just a headline R².
- **Interpretability:** standardized coefficients translated into a
  plain-English "what actually drives price" narrative.
- **Deployability:** the fitted pipeline is serialized with `joblib` so it
  can be dropped straight into a Flask/FastAPI/Streamlit app.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python data/generate_dataset.py   # (optional) regenerate the dataset
python train_model.py             # run the full pipeline, saves model + charts
```

Or open `notebooks/House_Price_Prediction.ipynb` for the narrated walkthrough
with all charts and commentary inline.

## Key insight
`area` is the single strongest driver of price, followed by being in a
preferred/prime area, having air conditioning, and main-road access —
consistent with real-world real estate intuition, which is itself a useful
sanity check that the model learned something sensible rather than noise.

## Possible extensions 
- Compare against tree-based models (Random Forest, XGBoost) for a non-linear performance ceiling.
- Add geospatial features (lat/long, neighborhood clustering) if available.
- Hyperparameter-tune Ridge/Lasso alpha via `GridSearchCV`.
- Wrap the saved `.pkl` in a small Flask/FastAPI service or Streamlit app for a live demo.
- Add SHAP values for more rigorous, non-linear-aware feature attribution.
