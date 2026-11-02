# Real Estate Investment Advisor: Project Documentation

## 1. Objective
Help investors decide whether a property is a **Good Investment** (classification) and estimate its **price after 5 years** (regression),
delivered through a Streamlit app with MLflow experiment tracking.

## 2. Data
| Item | Value |
|---|---|
| File | `india_housing_prices.csv` |
| Size | 250,000 listings x 23 columns, 20 states, 42 cities, 500 locality labels |
| Missing values / duplicates | 0 / 0 |
| Price unit | INR lakhs (1 lakh = 100,000) |

### Data-quality findings
- **Floor_No > Total_Floors in 116,304 rows (46.5%)**: physically impossible. Floors were capped at Total_Floors and the issue kept in `Floor_Data_Flag`.
- **Price_per_SqFt is in lakhs and rounded to 2 decimals** (0.05, 0.10 ...), losing precision. It was recomputed in INR from price and size.
- **Locality names repeat across all 42 cities**, so a locality is only unique as City + Locality (about 12 listings each). Locality-level analysis uses the 500 labels (~500 listings each).
- **No price history, no listing dates, no crime-rate column.** Time trends use construction year as a proxy; crime is replaced by an infrastructure score.
- **Features are independent of price.** Every correlation among raw features is below |0.01|, and city/state averages differ by only 2-3%. This pattern is typical of synthetic data.

## 3. Methodology
### 3.1 Preprocessing (`src/preprocess.py`)
Duplicate and missing-value checks, text trimming, floor consistency fix, IQR outlier flags (flagged, not removed because price and size are bounded;
19,723 price-per-sq-ft outliers flagged, all small and expensive listings), one-hot encoding and standard scaling inside the model pipeline.

### 3.2 Feature engineering
Price per sq ft (INR), School / Hospital density scores, Transport score, **Infrastructure score** (mean of the three), Is_Metro, Amenity count and
five amenity flags, Floor ratio, Size per BHK, Age bucket, City+Locality key.

### 3.3 Targets (assumption-based, in `src/config.py`)
Because the dataset has no history or outcomes, both targets are built from transparent rules that can be edited and re-run:
- **Annual growth** = 5% base + up to 3 pts x infrastructure score + 1 pt metro + 1 pt under-construction - 0.05 pt per year of age, clipped to 3-12%.
- **Price after 5 years** = price x (1 + growth)^5.
- **Good Investment** = price per sq ft <= city median **and** growth >= 6.25% (27.4% of listings).

### 3.4 Model inputs
Raw property fields plus engineered features. The two label ingredients (`Annual_Growth_Rate`, `Below_City_Median_PPSF`) are excluded, so models
must work from the listing itself.

### 3.5 Models and tracking
80/20 split (stratified for classification). Logistic/Linear Regression, Random Forest and XGBoost are each logged to MLflow with parameters,
metrics and the fitted pipeline (experiment `Real_Estate_Investment_Advisor`). The app serves the XGBoost pipelines retrained on all rows.

## 4. EDA summary (20 questions; full numbers in `reports/eda_findings.md`, charts in `reports/figures/`)
| Theme | Finding |
|---|---|
| Price and size | Prices 10-500 lakhs and sizes 500-5,000 sq ft, both close to uniform. Size vs price r = -0.003. |
| Outliers | 7.9% price-per-sq-ft outliers, driven by small expensive listings. |
| Location | Highest average price/sq ft: Karnataka; lowest: Punjab (2.5% apart). Priciest city: Bangalore (258.5 L) vs cheapest Cuttack (250.8 L). |
| Top localities | Top-5 locality prices are only ~8% above average and show no consistent trend by construction period. |
| Correlations | No meaningful relationship of schools, hospitals, furnishing, facing, parking or amenities with price or price per sq ft (all within about 1-3%). |
| Ownership and status | Owner types in near-equal thirds; availability 50/50. |
| Transport and investment | Share of Good Investments: 17.1% (low), 27.4% (medium), 37.7% (high transport access). This follows from the growth rule, not from market prices. |

## 5. Model results (hold-out set)
| Task | Model | Key metrics |
|---|---|---|
| Classification | Logistic Regression | Acc 0.900, F1 0.815, AUC 0.964 |
| | Random Forest | Acc 0.973, F1 0.950, AUC 0.996 |
| | **XGBoost** | **Acc 0.999, F1 0.998, AUC 1.000** |
| Regression | Linear Regression | RMSE 8.99 L, R2 0.998 |
| | Random Forest | RMSE 2.19 L, R2 0.9999 |
| | **XGBoost** | **RMSE 1.17 L, MAE 0.93 L, R2 ~1.000** |

**How to read these:** the scores are near-perfect because the targets are deterministic rules the models can learn
(XGBoost's top drivers are availability status, price per sq ft, city/metro status for the classifier, and today's price for the regressor).
A **sanity check** that predicts *today's* price from every other feature gives R2 = -0.006, i.e. no better than guessing the average. The data contains no real price signal,
so these results must not be presented as real-market forecasting accuracy.

## 6. Streamlit app (`app.py`)
1. **Investment Advisor**: property form -> Good/Not Good verdict, probability gauge, model confidence, 5-year price with range and implied growth,
   projection chart, price-per-sq-ft vs city median, feature importance.
2. **Property Explorer**: filter by state, city, price, area, BHK, type, availability; sortable table; CSV download.
3. **Market Insights**: city ranking, state/city heatmaps, investment hotspots by locality, construction-year trends, infrastructure effects.
4. **Model Performance**: metric tables, feature importance, caveat and assumption explanation.

## 7. Business use cases
Investor screening of listings; buyers comparing areas; real-estate platforms adding automated investment scores to listings; showing clients
transparent, adjustable assumptions behind a recommendation.

## 8. Limitations and next steps
- Replace the assumed growth model with real historical price series (e.g. listing price per quarter) and real investment outcomes; then retrain and re-validate.
- Add a crime-rate or safety index and real distance-to-metro/employment features.
- Add cross-validation, hyper-parameter search and SHAP explanations once a real signal exists.
- Fix source-data issues (impossible floor numbers, rounded price per sq ft) at the data-collection stage.
