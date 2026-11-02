# Video Explanation Script (about 6-7 minutes)

*Tip: keep the app open in one window and the figures folder in another. I haven't seen the Capstone Explanation Guideline in your reference links, so match the section order to it if it differs.*

**[0:00 - Intro]**
"Hello, I'm Shubhodip. This project is a Real Estate Investment Advisor. It tells an investor whether a property is a good investment and estimates its price after five years. I'll walk through the data, the analysis, the models, and then the Streamlit app."

**[0:30 - Data]**
"The dataset has 250,000 listings across 20 states and 42 cities. It has no missing values or duplicates, but I found three quality issues. Nearly half the rows have a floor number higher than the building's total floors, which is impossible, so I capped the floor and kept a flag. The price-per-square-foot column is in lakhs and rounded to two decimals, so I recomputed it in rupees. And locality names repeat across every city, so I treat locality as city plus locality."

**[1:30 - Feature engineering and targets]**
"I engineered price per square foot, school, hospital and transport scores, a combined infrastructure score, a metro flag and amenity features. The dataset has no price history, so I defined the targets with documented rules. Annual growth starts at 5 percent and rises with infrastructure, metro status and under-construction stock, and falls with building age. The five-year price is today's price compounded by that growth. A property is a Good Investment if its price per square foot is at or below its city median and its growth is at least 6.25 percent. That labels about 27 percent of listings as good. All of this lives in one config file, so the assumptions can be changed."

**[2:45 - EDA]**
"I answered all twenty EDA questions. Prices and sizes are roughly uniform. Size has almost no relationship with price, correlation about zero. City and state averages are within two to three percent of each other, and furnishing, facing, parking and amenities barely move price. The one clear pattern is transport: the share of good investments rises from 17 percent with low transport access to 38 percent with high access, which comes from the growth rule. The honest conclusion is that this data behaves like synthetic data, so I'm careful about what the models can claim."

**[3:50 - Models and MLflow]**
"I trained logistic or linear regression, random forest and XGBoost for each task and logged every run to MLflow with parameters, metrics and the model. XGBoost won both: about 99.9 percent accuracy for classification, and an RMSE of about 1.2 lakhs for the five-year price. These scores are very high because the targets are rules the model can learn. As a check, predicting today's price from the other features gives an R-squared of about zero, so there is no real market signal. I'm presenting these as proof the pipeline works, not as real-world forecasting accuracy."

**[4:50 - App demo]**
"Here's the app. On Investment Advisor I choose state and city, fill in the property details and click Analyse. I get a verdict, a probability gauge, the model's confidence, the estimated price in five years with a range, and the main drivers. [Demo a strong property, then a weak one.] Property Explorer filters by location, price, area and BHK and lets me download results. Market Insights has city rankings, heatmaps, investment hotspots and trends. Model Performance shows the metrics, feature importance and the assumptions."

**[6:10 - Wrap-up]**
"To make this production-ready, I'd replace the assumed growth model with real historical prices and outcomes, add a crime or safety index, and retrain with cross-validation. Thank you."
