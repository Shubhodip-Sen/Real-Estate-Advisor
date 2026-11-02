"""Real Estate Investment Advisor - Streamlit app.

Run:  streamlit run app.py
Needs: data/processed/housing_clean.csv, models/*.joblib, models/metrics.json
       (create them with: python src/preprocess.py && python src/train_models.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import (AMENITY_LIST, CLEAN_CSV, CURRENT_YEAR, GROWTH, GOOD_INVESTMENT_GROWTH_THRESHOLD,  # noqa: E402
                    HORIZON_YEARS, METRO_CITIES, MODELS_DIR, TRANSPORT_MAP)

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.verdict {border-radius: 14px; padding: 1.1rem 1.4rem; font-size: 1.35rem; font-weight: 700; margin-bottom: .6rem;}
.good {background: #e6f6ec; color: #11623a; border: 1px solid #b7e4c7;}
.bad  {background: #fdeceb; color: #8a1f17; border: 1px solid #f5c2be;}
.price-box {background: #eef4fb; color: #12395f; border: 1px solid #c8dcf2; border-radius: 14px; padding: 1.1rem 1.4rem;}
.price-box .big {font-size: 2rem; font-weight: 800;}
</style>
""", unsafe_allow_html=True)

LOAD_COLS = ["ID", "State", "City", "Locality", "Locality_Key", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
             "Price_per_SqFt_INR", "Year_Built", "Furnished_Status", "Age_of_Property", "Public_Transport_Accessibility",
             "Availability_Status", "Owner_Type", "Facing", "Infrastructure_Score", "Annual_Growth_Rate",
             "Future_Price_5Y_Lakhs", "Good_Investment", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Parking_Space"]


# ----------------------------------------------------------------------------- loaders
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV, usecols=LOAD_COLS)
    for c in ["State", "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
              "Availability_Status", "Owner_Type", "Facing", "Parking_Space"]:
        df[c] = df[c].astype("category")
    return df


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return joblib.load(MODELS_DIR / "clf_pipeline.joblib"), joblib.load(MODELS_DIR / "reg_pipeline.joblib")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def state_city_map(_df) -> dict:
    return {s: sorted(g.astype(str).unique()) for s, g in _df.groupby("State", observed=True)["City"]}


@st.cache_data
def city_stats(_df) -> pd.DataFrame:
    return _df.groupby("City", observed=True).agg(
        median_ppsf=("Price_per_SqFt_INR", "median"), avg_price=("Price_in_Lakhs", "mean"),
        avg_growth=("Annual_Growth_Rate", "mean"), good_share=("Good_Investment", "mean"), n=("ID", "size"))


def inr(lakhs: float) -> str:
    return f"₹{lakhs / 100:,.2f} Cr" if lakhs >= 100 else f"₹{lakhs:,.1f} L"


def build_row(i: dict) -> pd.DataFrame:
    """Same feature engineering as src/preprocess.py, for one user-entered property."""
    ppsf = i["price"] * 100_000 / i["size"]
    infra = (i["schools"] / 10 + i["hospitals"] / 10 + TRANSPORT_MAP[i["transport"]]) / 3
    row = {
        "City": i["city"], "Property_Type": i["ptype"], "Furnished_Status": i["furnished"],
        "Public_Transport_Accessibility": i["transport"], "Facing": i["facing"], "Owner_Type": i["owner"],
        "Availability_Status": i["avail"], "Parking_Space": "Yes" if i["parking"] else "No",
        "Security": "Yes" if i["security"] else "No",
        "BHK": i["bhk"], "Size_in_SqFt": i["size"], "Price_in_Lakhs": i["price"], "Price_per_SqFt_INR": ppsf,
        "Age_of_Property": CURRENT_YEAR - i["year"], "Floor_No": i["floor"], "Total_Floors": i["total_floors"],
        "Nearby_Schools": i["schools"], "Nearby_Hospitals": i["hospitals"], "Amenity_Count": len(i["amenities"]),
        "Infrastructure_Score": infra, "Is_Metro": int(i["city"] in METRO_CITIES),
        "Floor_Ratio": i["floor"] / i["total_floors"], "Size_per_BHK": i["size"] / i["bhk"],
    }
    for a in AMENITY_LIST:
        row[f"Has_{a}"] = int(a in i["amenities"])
    return pd.DataFrame([row])


# ----------------------------------------------------------------------------- pages
def page_advisor(df):
    st.title("🏠 Real Estate Investment Advisor")
    st.caption("Enter a property to get a Good-Investment verdict and an estimated price after "
               f"{HORIZON_YEARS} years.")
    clf, reg = load_models()
    met = load_metrics()
    sc = state_city_map(df)
    cs = city_stats(df)

    c1, c2 = st.columns(2)
    state = c1.selectbox("State", sorted(sc))
    city = c2.selectbox("City", sc[state])

    with st.form("advisor_form"):
        a, b, c = st.columns(3)
        ptype = a.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        bhk = b.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
        size = c.number_input("Size (sq ft)", 300, 6000, 1800, step=50)
        a, b, c = st.columns(3)
        price = a.number_input("Asking price (₹ lakhs)", 5.0, 1000.0, 180.0, step=5.0)
        year = b.number_input("Year built", 1985, CURRENT_YEAR, 2012)
        avail = c.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        a, b, c = st.columns(3)
        furnished = a.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
        floor = b.number_input("Floor number", 0, 40, 5)
        total_floors = c.number_input("Total floors in building", 1, 40, 12)
        a, b, c = st.columns(3)
        schools = a.slider("Nearby schools", 0, 10, 6)
        hospitals = b.slider("Nearby hospitals", 0, 10, 5)
        transport = c.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)
        a, b, c = st.columns(3)
        facing = a.selectbox("Facing", ["North", "South", "East", "West"])
        owner = b.selectbox("Owner type", ["Owner", "Builder", "Broker"])
        amenities = c.multiselect("Amenities", AMENITY_LIST, default=["Gym", "Pool"])
        a, b, _ = st.columns(3)
        parking = a.checkbox("Parking space", value=True)
        security = b.checkbox("Security (gated/CCTV/guard)", value=True)
        go_btn = st.form_submit_button("Analyse investment", type="primary", width="stretch")

    if not go_btn:
        st.info("Fill in the property details and press **Analyse investment**.")
        return
    if floor > total_floors:
        st.error("Floor number cannot be higher than total floors in the building.")
        return

    inp = dict(city=city, ptype=ptype, bhk=bhk, size=size, price=price, year=year, avail=avail, furnished=furnished,
               floor=floor, total_floors=total_floors, schools=schools, hospitals=hospitals, transport=transport,
               facing=facing, owner=owner, amenities=amenities, parking=parking, security=security)
    X = build_row(inp)
    prob = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    cagr = (future / price) ** (1 / HORIZON_YEARS) - 1
    rp = met["regression_residual_pct"]
    lo, hi = future / (1 + rp["p95"] / 100), future / (1 + rp["p05"] / 100)
    conf = prob if prob >= .5 else 1 - prob

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        if prob >= .5:
            st.markdown(f'<div class="verdict good">✅ Good Investment</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict bad">⚠️ Not a Good Investment</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="price-box">Estimated price after {HORIZON_YEARS} years'
            f'<div class="big">{inr(future)}</div>'
            f'{future - price:+,.1f} lakhs vs today &nbsp;|&nbsp; implied growth {cagr:.1%} per year<br>'
            f'<small>Model range (90% of fit errors): {inr(lo)} - {inr(hi)}</small></div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Model confidence", f"{conf:.0%}")
        ppsf = price * 100_000 / size
        med = cs.loc[city, "median_ppsf"]
        m2.metric("Price / sq ft", f"₹{ppsf:,.0f}", f"{(ppsf / med - 1):+.0%} vs {city} median", delta_color="inverse")
        m3.metric(f"{city} Good-Investment share", f"{cs.loc[city, 'good_share']:.0%}")
    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Probability of Good Investment"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2E86AB"},
                   "steps": [{"range": [0, 50], "color": "#fdeceb"}, {"range": [50, 100], "color": "#e6f6ec"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        fig.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig, width="stretch")

    t1, t2 = st.columns(2)
    with t1:
        yrs = np.arange(0, HORIZON_YEARS + 1)
        city_g = cs.loc[city, "avg_growth"]
        proj = pd.DataFrame({"Year": yrs, "This property": price * (1 + cagr) ** yrs,
                             f"At {city} average growth": price * (1 + city_g) ** yrs}).melt("Year", var_name="Scenario", value_name="Value (₹ lakhs)")
        f = px.line(proj, x="Year", y="Value (₹ lakhs)", color="Scenario", markers=True, title="Projected value path")
        f.update_layout(height=330, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(f, width="stretch")
    with t2:
        imp = pd.Series(met["feature_importance_clf"]).head(10)[::-1]
        f = px.bar(imp, orientation="h", title="What drives the Good-Investment call", labels={"value": "Importance", "index": ""})
        f.update_layout(height=330, showlegend=False)
        st.plotly_chart(f, width="stretch")
    st.caption("Confidence = model probability for the predicted class. The price range reflects how closely the model "
               "reproduces the assumed growth model, not real-market uncertainty. This is a decision aid, not financial advice.")


def page_explorer(df):
    st.title("🔎 Property Explorer")
    st.caption("Filter the 250,000 listings by location, price, size and BHK.")
    sc = state_city_map(df)
    c1, c2, c3 = st.columns(3)
    states = c1.multiselect("State", sorted(sc))
    pool = sorted({c for s in (states or sc) for c in sc[s]})
    cities = c2.multiselect("City", pool)
    ptypes = c3.multiselect("Property type", sorted(df.Property_Type.astype(str).unique()))
    c1, c2 = st.columns(2)
    price_rng = c1.slider("Price (₹ lakhs)", 10, 500, (10, 500))
    size_rng = c2.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50)
    c1, c2, c3, c4 = st.columns(4)
    bhks = c1.multiselect("BHK", [1, 2, 3, 4, 5])
    avail = c2.multiselect("Availability", sorted(df.Availability_Status.astype(str).unique()))
    only_good = c3.checkbox("Good Investments only")
    sort_by = c4.selectbox("Sort by", ["Future_Price_5Y_Lakhs", "Price_in_Lakhs", "Price_per_SqFt_INR", "Annual_Growth_Rate", "Size_in_SqFt"])

    f = df[df.Price_in_Lakhs.between(*price_rng) & df.Size_in_SqFt.between(*size_rng)]
    if states: f = f[f.State.isin(states)]
    if cities: f = f[f.City.isin(cities)]
    if ptypes: f = f[f.Property_Type.isin(ptypes)]
    if bhks: f = f[f.BHK.isin(bhks)]
    if avail: f = f[f.Availability_Status.isin(avail)]
    if only_good: f = f[f.Good_Investment == 1]

    m = st.columns(4)
    m[0].metric("Matching listings", f"{len(f):,}")
    if len(f):
        m[1].metric("Avg price", inr(f.Price_in_Lakhs.mean()))
        m[2].metric("Avg price / sq ft", f"₹{f.Price_per_SqFt_INR.mean():,.0f}")
        m[3].metric("Good-Investment share", f"{f.Good_Investment.mean():.0%}")
        show = f.sort_values(sort_by, ascending=False).head(500)[
            ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
             "Price_per_SqFt_INR", "Future_Price_5Y_Lakhs", "Annual_Growth_Rate", "Good_Investment"]]
        st.dataframe(show, width="stretch", hide_index=True, height=420, column_config={
            "Annual_Growth_Rate": st.column_config.NumberColumn("Growth/yr", format="%.1f%%", help="Assumed annual appreciation"),
            "Price_per_SqFt_INR": st.column_config.NumberColumn("₹ / sq ft", format="%,.0f"),
            "Good_Investment": st.column_config.CheckboxColumn("Good inv.")})
        st.caption(f"Showing top {len(show)} of {len(f):,} by {sort_by}.")
        st.download_button("Download filtered results (CSV)", f.head(50_000).to_csv(index=False).encode(),
                           "filtered_properties.csv", "text/csv")
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.histogram(f.sample(min(len(f), 20_000), random_state=1), x="Price_in_Lakhs", nbins=40,
                                     title="Price distribution (filtered)"), width="stretch")
        g2.plotly_chart(px.histogram(f.sample(min(len(f), 20_000), random_state=1), x="Annual_Growth_Rate", nbins=40,
                                     title="Assumed annual growth (filtered)"), width="stretch")
    else:
        st.warning("No listings match these filters.")


def page_insights(df):
    st.title("📊 Market Insights")
    st.caption("Note: the dataset has no listing dates. 'Trends' use construction year as a proxy, and growth rates come from the project's assumption model.")
    t1, t2, t3, t4 = st.tabs(["City ranking", "Heatmaps", "Investment hotspots", "Trends & infrastructure"])

    with t1:
        metric = st.radio("Metric", ["Avg price (₹ lakhs)", "Avg price / sq ft (₹)", "Good-Investment share"], horizontal=True)
        col = {"Avg price (₹ lakhs)": ("Price_in_Lakhs", "mean"), "Avg price / sq ft (₹)": ("Price_per_SqFt_INR", "mean"),
               "Good-Investment share": ("Good_Investment", "mean")}[metric]
        n = st.slider("Cities to show", 5, 42, 15)
        s = df.groupby("City", observed=True)[col[0]].agg(col[1]).sort_values(ascending=False).head(n)[::-1]
        f = px.bar(s, orientation="h", labels={"value": metric, "City": ""}, color=s.values, color_continuous_scale="Blues")
        f.update_layout(height=max(380, 24 * n), coloraxis_showscale=False, showlegend=False)
        f.update_xaxes(range=[s.min() * 0.97, s.max() * 1.01] if col[0] != "Good_Investment" else None)
        st.plotly_chart(f, width="stretch")
        st.caption("Axis starts near the minimum so small differences are visible. City averages differ by only a few percent in this dataset.")

    with t2:
        c1, c2, c3 = st.columns(3)
        rows = c1.selectbox("Rows", ["State", "City"])
        cols_ = c2.selectbox("Columns", ["Property_Type", "BHK", "Furnished_Status", "Availability_Status"])
        val = c3.selectbox("Value", ["Price_per_SqFt_INR", "Price_in_Lakhs", "Good_Investment", "Annual_Growth_Rate"])
        pv = df.pivot_table(index=rows, columns=cols_, values=val, aggfunc="mean", observed=True)
        f = px.imshow(pv, text_auto=".2f" if val in ("Good_Investment", "Annual_Growth_Rate") else ".0f",
                      color_continuous_scale="YlGnBu", aspect="auto", title=f"Average {val} by {rows} x {cols_}")
        f.update_layout(height=max(420, 22 * len(pv)))
        st.plotly_chart(f, width="stretch")

    with t3:
        gran = st.radio("Granularity", ["Locality (across cities)", "City + Locality"], horizontal=True)
        keys = ["Locality"] if gran.startswith("Locality") else ["City", "Locality"]
        default_min, max_min = (300, 500) if len(keys) == 1 else (10, 25)
        min_n = st.slider("Minimum listings per group", 5, max_min, default_min, key=f"minn_{len(keys)}")
        loc = df.groupby(keys, observed=True).agg(
            listings=("ID", "size"), good_share=("Good_Investment", "mean"), avg_ppsf=("Price_per_SqFt_INR", "mean"),
            avg_growth=("Annual_Growth_Rate", "mean"), avg_price=("Price_in_Lakhs", "mean")).reset_index()
        loc = loc[loc.listings >= min_n].sort_values("good_share", ascending=False)
        if loc.empty:
            st.warning("No groups meet that minimum - lower the slider.")
        else:
            top = loc.head(15).copy()
            top["Place"] = top[keys].astype(str).agg(" | ".join, axis=1)
            f = px.bar(top[::-1], x="good_share", y="Place", orientation="h", title="Top 15 by Good-Investment share",
                       labels={"good_share": "Good-Investment share"}, color="avg_growth", color_continuous_scale="Greens")
            f.update_layout(height=480)
            st.plotly_chart(f, width="stretch")
            st.dataframe(loc.head(50).round(3), width="stretch", hide_index=True)
        st.caption("Locality names repeat across cities. City + Locality groups hold only ~12 listings each, so their rankings are noisy; "
                   "prefer the across-cities view for stable results.")

    with t4:
        cities = st.multiselect("Compare cities", sorted(df.City.astype(str).unique()), default=["Mumbai", "Bangalore", "Hyderabad"])
        if cities:
            tr = df[df.City.isin(cities)].groupby(["Year_Built", "City"], observed=True).Price_in_Lakhs.mean().reset_index()
            st.plotly_chart(px.line(tr, x="Year_Built", y="Price_in_Lakhs", color="City", title="Avg price by construction year (proxy for trend)"),
                            width="stretch")
        a, b = st.columns(2)
        bins = pd.cut(df.Infrastructure_Score, [0, .3, .45, .6, .75, 1.0])
        g = df.groupby(bins, observed=True).agg(growth=("Annual_Growth_Rate", "mean"), ppsf=("Price_per_SqFt_INR", "mean"))
        g.index = g.index.astype(str)
        a.plotly_chart(px.bar(g, y="growth", title="Projected growth by infrastructure score", labels={"growth": "Avg annual growth", "index": "Infrastructure score"}),
                       width="stretch")
        b.plotly_chart(px.bar(g, y="ppsf", title="Price / sq ft by infrastructure score", labels={"ppsf": "Avg ₹ / sq ft", "index": "Infrastructure score"}),
                       width="stretch")
        st.caption("Infrastructure score lifts projected growth by design, but has almost no effect on today's price per sq ft.")


def page_models(df):
    st.title("🧠 Model Performance")
    met = load_metrics()
    st.warning(
        "**How to read these scores.** The dataset has no price history or investment outcomes, so both targets are built from "
        "documented rules. High scores show the models recover those rules. They do not prove real-market forecasting skill. "
        f"As a check, predicting *today's* price from all other features gives R² = {met['sanity']['r2']:.3f}: "
        "the raw data carries essentially no price signal.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classification: Good Investment")
        t = pd.DataFrame(met["classification"]).T
        st.dataframe(t.style.format("{:.4f}").highlight_max(axis=0, color="#d8f0e0"), width="stretch")
        st.caption(f"Positive class share: {met['class_balance_good_investment']:.1%}. Best by ROC-AUC: {met['best_classifier_by_auc']}.")
    with c2:
        st.subheader(f"Regression: price after {HORIZON_YEARS} years")
        t = pd.DataFrame(met["regression"]).T
        st.dataframe(t.style.format("{:.4f}").highlight_min(subset=["rmse", "mae"], color="#d8f0e0").highlight_max(subset=["r2"], color="#d8f0e0"),
                     width="stretch")
        st.caption(f"RMSE in ₹ lakhs. Best by RMSE: {met['best_regressor_by_rmse']}.")
    a, b = st.columns(2)
    for col, key, ttl in [(a, "feature_importance_clf", "Feature importance: classifier"), (b, "feature_importance_reg", "Feature importance: regressor")]:
        imp = pd.Series(met[key]).head(12)[::-1]
        f = px.bar(imp, orientation="h", title=ttl, labels={"value": "Share of importance", "index": ""})
        f.update_layout(height=420, showlegend=False)
        col.plotly_chart(f, width="stretch")
    with st.expander("How the targets are defined (assumptions)"):
        g = GROWTH
        st.markdown(f"""
- **Annual growth** = {g['base']:.0%} base + up to {g['infra_weight']:.0%} × infrastructure score (schools, hospitals, transport)
  + {g['metro_bonus']:.0%} if metro city + {g['under_construction']:.0%} if under construction − {g['age_penalty']*100:.2f} pt per year of age, clipped to {g['min']:.0%}-{g['max']:.0%}.
- **Price after {HORIZON_YEARS} years** = today's price × (1 + growth)^{HORIZON_YEARS}.
- **Good Investment** = price per sq ft at or below the city median **and** annual growth ≥ {GOOD_INVESTMENT_GROWTH_THRESHOLD:.2%}.
- Edit `src/config.py` and re-run `preprocess.py` + `train_models.py` to test other assumptions.
""")
    st.info("Experiment tracking: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the project folder to compare every run.")


# ----------------------------------------------------------------------------- router
def main():
    if not CLEAN_CSV.exists() or not (MODELS_DIR / "metrics.json").exists():
        st.error("Processed data or models not found. Run `python src/preprocess.py` then `python src/train_models.py`.")
        st.stop()
    df = load_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Investment Advisor", "Property Explorer", "Market Insights", "Model Performance"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"{len(df):,} listings · {df.City.nunique()} cities · {df.State.nunique()} states")
    {"Investment Advisor": page_advisor, "Property Explorer": page_explorer,
     "Market Insights": page_insights, "Model Performance": page_models}[page](df)


main()
