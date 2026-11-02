# Real Estate Investment Advisor

Classifies whether a property is a **Good Investment** and predicts its **price after 5 years**, with EDA,
MLflow experiment tracking and a Streamlit app. Dataset: `india_housing_prices.csv` (250,000 listings, 20 states, 42 cities).

## Project structure
```
app.py                       Streamlit app (4 pages)
src/config.py                paths + business assumptions (growth model, label threshold)
src/preprocess.py            cleaning + feature engineering -> data/processed/housing_clean.csv
src/eda.py                   20 EDA questions -> reports/figures/*.png + reports/eda_findings.md
src/train_models.py          trains 6 models, logs to MLflow, saves models/*.joblib + metrics.json
data/raw/                    put india_housing_prices.csv here
data/processed/              cleaned dataset (deliverable)
models/                      deployed pipelines + metrics.json
docs/PROJECT_DOCUMENTATION.md   methodology, findings, use cases, limitations
docs/VIDEO_SCRIPT.md            explanation script for the recording
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: rebuild everything from the raw file (cleaned data + models are already included)
python src/preprocess.py
python src/eda.py
python src/train_models.py

streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db       # after train_models.py
```
All paths are relative to the project folder, so it runs from any location.

## Deploy on Streamlit Cloud
Push the folder to GitHub (include `data/processed/`, `models/`, `requirements.txt`), then create an app pointing to `app.py`.
`housing_clean.csv` is ~70 MB: under GitHub's 100 MB limit, but use Git LFS if you add more data.

## Important caveat
The raw data has no price history and its features are statistically independent of price (all |r| < 0.01),
which is typical of synthetic data. The 5-year price and the Good-Investment label are therefore **rule-based and documented**
in `src/config.py`. Model scores show how well each model recovers those rules, not real-market forecasting skill.
See `docs/PROJECT_DOCUMENTATION.md` for details.
