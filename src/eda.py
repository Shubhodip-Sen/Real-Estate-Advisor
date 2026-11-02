"""Step 2 - Exploratory Data Analysis (20 questions from the project brief).

Run:  python src/eda.py        (after src/preprocess.py)
Outputs: reports/figures/q01..q20*.png  and  reports/eda_findings.md
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: E402,F403

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CLEAN_CSV)
F = []  # findings lines


def save(name, title=None):
    if title:
        plt.title(title, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close()


def section(t):
    F.append(f"\n## {t}\n")


def say(q, a):
    F.append(f"**{q}**  \n{a}\n")


F.append("# EDA Findings - India Housing Prices\n")
F.append(f"Dataset: {len(df):,} listings, {df.State.nunique()} states, {df.City.nunique()} cities. "
         "Prices in INR lakhs; price per sq ft shown in INR.\n")

# ---------------------------------------------------------------- 1-5
section("1-5  Price & Size")
plt.figure(figsize=(8, 4.5)); sns.histplot(df.Price_in_Lakhs, bins=50, kde=True, color="#2E86AB")
plt.xlabel("Price (lakhs)"); save("q01_price_distribution", "Q1. Distribution of property prices")
say("Q1. Distribution of prices",
    f"Prices run {df.Price_in_Lakhs.min():.0f}-{df.Price_in_Lakhs.max():.0f} lakhs, mean {df.Price_in_Lakhs.mean():.0f}, "
    f"median {df.Price_in_Lakhs.median():.0f}, skew {df.Price_in_Lakhs.skew():.2f}. The shape is almost flat (uniform) rather than "
    "the right-skew normally seen in real markets - a hint that the data is synthetic.")

plt.figure(figsize=(8, 4.5)); sns.histplot(df.Size_in_SqFt, bins=50, kde=True, color="#F18F01")
plt.xlabel("Size (sq ft)"); save("q02_size_distribution", "Q2. Distribution of property sizes")
say("Q2. Distribution of sizes",
    f"Sizes span {df.Size_in_SqFt.min():,}-{df.Size_in_SqFt.max():,} sq ft (mean {df.Size_in_SqFt.mean():,.0f}), again roughly uniform. "
    "A 1 BHK averages the same size as a 5 BHK, which does not happen in real listings.")

plt.figure(figsize=(8, 4.5)); sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt_INR", palette="Set2", hue="Property_Type", legend=False)
plt.ylabel("Price per sq ft (INR)"); save("q03_ppsf_by_type", "Q3. Price per sq ft by property type")
g = df.groupby("Property_Type").Price_per_SqFt_INR.agg(["mean", "median"]).round(0)
say("Q3. Price per sq ft by property type",
    "; ".join(f"{i}: mean INR {r['mean']:,.0f}, median INR {r['median']:,.0f}" for i, r in g.iterrows()) +
    ". Differences between types are under 1%, i.e. property type does not drive price here.")

s = df.sample(8000, random_state=1)
plt.figure(figsize=(8, 5)); sns.scatterplot(data=s, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.3, s=12, color="#2E86AB")
save("q04_size_vs_price", "Q4. Size vs price (8k sample)")
r = df[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1]
say("Q4. Size vs price", f"Pearson r = {r:.3f}: effectively **no relationship**. In real markets this is typically strongly positive.")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, c, col in zip(ax, ["Price_per_SqFt_INR", "Size_in_SqFt", "Price_in_Lakhs"], ["#C73E1D", "#F18F01", "#2E86AB"]):
    sns.boxplot(y=df[c], ax=a, color=col); a.set_title(c)
save("q05_outliers", "Q5. Outlier check")
nppsf = int(df.PPSF_Outlier.sum())
say("Q5. Outliers",
    f"IQR rule flags {nppsf:,} listings ({nppsf/len(df):.1%}) as high price-per-sq-ft outliers (max INR {df.Price_per_SqFt_INR.max():,.0f}); "
    "none for price or size, which are bounded. "
    f"The PPSF outliers are small, expensive listings (avg {df[df.PPSF_Outlier==1].Size_in_SqFt.mean():,.0f} sq ft at {df[df.PPSF_Outlier==1].Price_in_Lakhs.mean():.0f} lakhs vs "
    f"{df.Size_in_SqFt.mean():,.0f} sq ft at {df.Price_in_Lakhs.mean():.0f} lakhs overall) - flagged, not removed.")

# ---------------------------------------------------------------- 6-10
section("6-10  Location")
st = df.groupby("State").Price_per_SqFt_INR.mean().sort_values()
plt.figure(figsize=(8, 7)); st.plot.barh(color="#2E86AB"); plt.xlabel("Avg price per sq ft (INR)")
plt.xlim(st.min() * .97, st.max() * 1.02); save("q06_ppsf_by_state", "Q6. Avg price per sq ft by state")
say("Q6. Avg PPSF by state", f"Highest: {st.index[-1]} (INR {st.iloc[-1]:,.0f}); lowest: {st.index[0]} (INR {st.iloc[0]:,.0f}). "
    f"Spread across states is only {(st.max()/st.min()-1):.1%}.")

ct = df.groupby("City").Price_in_Lakhs.mean().sort_values()
plt.figure(figsize=(8, 10)); ct.plot.barh(color="#F18F01"); plt.xlim(ct.min() * .98, ct.max() * 1.01)
plt.xlabel("Avg price (lakhs)"); save("q07_price_by_city", "Q7. Average price by city")
say("Q7. Avg price by city", f"Highest: {ct.index[-1]} ({ct.iloc[-1]:.1f} lakhs); lowest: {ct.index[0]} ({ct.iloc[0]:.1f} lakhs). "
    f"All 42 cities sit within {(ct.max()/ct.min()-1):.1%} of each other.")

lm = df.groupby("Locality").Age_of_Property.median().sort_values()
plt.figure(figsize=(8, 4.5)); sns.histplot(lm, bins=15, color="#6A4C93")
plt.xlabel("Median age of property (years)"); plt.ylabel("Number of localities")
save("q08_median_age_locality", "Q8. Median property age across the 500 localities")
say("Q8. Median age by locality", f"Across the 500 localities the median age ranges {lm.min():.0f}-{lm.max():.0f} years "
    f"(overall median {df.Age_of_Property.median():.0f}); oldest stock: {lm.index[-1]} ({lm.iloc[-1]:.0f} y), newest: {lm.index[0]} ({lm.iloc[0]:.0f} y). "
    "Locality labels repeat across all 42 cities, so they are analysed at label level (~500 listings each); a city-locality pair has only ~12 listings, too few for stable medians.")

bc = pd.crosstab(df.City, df.BHK, normalize="index") * 100
plt.figure(figsize=(8, 11)); sns.heatmap(bc.round(1), annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "% of city listings"})
save("q09_bhk_by_city", "Q9. BHK mix by city (%)")
say("Q9. BHK across cities", f"Every city has a near-even 1-5 BHK split (each ~{bc.values.mean():.0f}% +/- {bc.values.std():.1f}).")

loc = df.groupby("Locality").Price_in_Lakhs.mean().nlargest(5)
sub = df[df.Locality.isin(loc.index)].copy()
sub["Build_Period"] = sub.Year_Built // 5 * 5
piv = sub.groupby(["Build_Period", "Locality"]).Price_in_Lakhs.mean().unstack()[loc.index]
plt.figure(figsize=(9, 5)); piv.plot(marker="o", ax=plt.gca()); plt.ylabel("Avg price (lakhs)"); plt.xlabel("Year-built period (5-year bins)")
plt.legend(fontsize=8, title="Locality"); save("q10_top5_locality_trend", "Q10. Top-5 priciest localities - avg price by construction period")
spread = (piv.max().max() / piv.min().min() - 1)
say("Q10. Top-5 locality price trends",
    "The data has no listing dates, so a true time trend is impossible; construction period is used as a proxy. "
    f"Top 5 by average price: {', '.join(f'{i} ({v:.0f} L)' for i, v in loc.items())}, only {(loc.iloc[0]/df.Price_in_Lakhs.mean()-1):.0%} above the overall mean. "
    f"Lines move up and down with no consistent direction (each point is ~70 listings, so swings are sampling noise).")

# ---------------------------------------------------------------- 11-15
section("11-15  Relationships & Correlation")
numc = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Infrastructure_Score"]
cm = df[numc].corr()
plt.figure(figsize=(9, 7.5)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
save("q11_correlation", "Q11. Correlation matrix")
off = cm.where(~np.eye(len(cm), dtype=bool)).abs().stack()
pairs = [(a, b, v) for (a, b), v in cm.stack().items() if a < b and a != b]
_raw = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Total_Floors", "Amenity_Count"]
raw_max = df[_raw].corr().abs().where(~np.eye(len(_raw), dtype=bool)).stack().max()
top = sorted(pairs, key=lambda x: -abs(x[2]))[:3]
say("Q11. Numeric correlations",
    "Strongest pairs: " + "; ".join(f"{a} vs {b} = {v:.2f}" for a, b, v in top) +
    ". The strong ones are mechanical: Price_per_SqFt is derived from price and size, and Floor_No was capped at Total_Floors during cleaning. "
    f"Among the independent raw features (BHK, size, price, age, schools, hospitals, total floors, amenity count) the largest |r| is {raw_max:.3f}.")

for qn, col, lab in [("q12", "Nearby_Schools", "Nearby schools"), ("q13", "Nearby_Hospitals", "Nearby hospitals")]:
    plt.figure(figsize=(8, 4.5))
    m = df.groupby(col).Price_per_SqFt_INR.mean()
    plt.plot(m.index, m.values, marker="o", color="#2E86AB"); plt.xlabel(lab); plt.ylabel("Avg PPSF (INR)")
    plt.ylim(m.min() * .95, m.max() * 1.05)
    save(f"{qn}_{col.lower()}_vs_ppsf", f"{qn.upper()}. {lab} vs price per sq ft")
    rr = df[[col, "Price_per_SqFt_INR"]].corr().iloc[0, 1]
    say(f"{qn.upper()}. {lab} vs PPSF", f"r = {rr:.3f}. Average PPSF moves by less than {(m.max()/m.min()-1):.1%} from fewest to most - no meaningful link.")

plt.figure(figsize=(8, 4.5)); sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", palette="pastel", hue="Furnished_Status", legend=False)
save("q14_price_by_furnishing", "Q14. Price by furnished status")
m = df.groupby("Furnished_Status").Price_in_Lakhs.mean().round(1)
say("Q14. Price by furnishing", "; ".join(f"{i}: {v} lakhs" for i, v in m.items()) + ". Furnishing carries no price premium in this data.")

plt.figure(figsize=(8, 4.5)); m = df.groupby("Facing").Price_per_SqFt_INR.mean().sort_values()
sns.barplot(x=m.index, y=m.values, palette="viridis", hue=m.index, legend=False); plt.ylim(m.min() * .97, m.max() * 1.02)
plt.ylabel("Avg PPSF (INR)"); save("q15_ppsf_by_facing", "Q15. Price per sq ft by facing")
say("Q15. PPSF by facing", "; ".join(f"{i}: {v:,.0f}" for i, v in m.items()) + f". Spread is {(m.max()/m.min()-1):.1%}.")

# ---------------------------------------------------------------- 16-20
section("16-20  Investment, Amenities & Ownership")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
vc = df.Owner_Type.value_counts(); ax[0].pie(vc, labels=vc.index, autopct="%1.1f%%", colors=sns.color_palette("Set2")); ax[0].set_title("Owner type")
vc2 = df.Availability_Status.value_counts(); ax[1].pie(vc2, labels=vc2.index, autopct="%1.1f%%", colors=sns.color_palette("pastel")); ax[1].set_title("Availability status")
save("q16_17_owner_availability", "Q16-17. Ownership & availability")
say("Q16. Properties per owner type", "; ".join(f"{i}: {v:,}" for i, v in vc.items()) + " (near-equal thirds).")
say("Q17. Properties per availability status", "; ".join(f"{i}: {v:,}" for i, v in vc2.items()) + " (about 50/50).")

plt.figure(figsize=(7, 4.5)); m = df.groupby("Parking_Space").Price_in_Lakhs.mean()
sns.barplot(x=m.index, y=m.values, palette="Set2", hue=m.index, legend=False); plt.ylim(m.min() * .98, m.max() * 1.01)
plt.ylabel("Avg price (lakhs)"); save("q18_parking_vs_price", "Q18. Parking space vs price")
say("Q18. Parking vs price", "; ".join(f"Parking={i}: {v:.1f} lakhs" for i, v in m.items()) + f". Difference {abs(m.diff().iloc[-1]):.1f} lakhs (<1%).")

am = pd.DataFrame({a: [df[df[f"Has_{a}"] == 1].Price_per_SqFt_INR.mean(), df[df[f"Has_{a}"] == 0].Price_per_SqFt_INR.mean()]
                   for a in AMENITY_LIST}, index=["With", "Without"]).T
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
am.plot.bar(ax=ax[0], color=["#2E86AB", "#BBBBBB"]); ax[0].set_ylabel("Avg PPSF (INR)"); ax[0].set_ylim(am.values.min() * .97, am.values.max() * 1.02)
ac = df.groupby("Amenity_Count").Price_per_SqFt_INR.mean(); ax[1].plot(ac.index, ac.values, marker="o", color="#C73E1D")
ax[1].set_xlabel("Number of amenities"); ax[1].set_ylabel("Avg PPSF (INR)"); ax[1].set_ylim(ac.min() * .95, ac.max() * 1.05)
save("q19_amenities_vs_ppsf", "Q19. Amenities vs price per sq ft")
say("Q19. Amenities vs PPSF", "; ".join(f"{a}: {(r.With / r.Without - 1):+.1%}" for a, r in am.iterrows()) +
    ". No amenity moves price per sq ft by more than about +/-1%.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
m = df.groupby("Public_Transport_Accessibility").Price_per_SqFt_INR.mean().reindex(["Low", "Medium", "High"])
sns.barplot(x=m.index, y=m.values, ax=ax[0], palette="Blues", hue=m.index, legend=False); ax[0].set_ylim(m.min() * .97, m.max() * 1.02); ax[0].set_ylabel("Avg PPSF (INR)")
gi = df.groupby("Public_Transport_Accessibility").Good_Investment.mean().reindex(["Low", "Medium", "High"]) * 100
sns.barplot(x=gi.index, y=gi.values, ax=ax[1], palette="Greens", hue=gi.index, legend=False); ax[1].set_ylabel("% rated Good Investment")
save("q20_transport", "Q20. Public transport vs PPSF and investment potential")
say("Q20. Transport vs PPSF / investment potential",
    "PPSF: " + "; ".join(f"{i} {v:,.0f}" for i, v in m.items()) + ". Good-Investment share: " +
    "; ".join(f"{i} {v:.1f}%" for i, v in gi.items()) +
    ". Price is unaffected, but better transport raises the share of Good Investments because it lifts the infrastructure score in the growth model.")

# ---------------------------------------------------------------- extras for the brief's EDA bullets
section("Brief-specific EDA bullets")
plt.figure(figsize=(8, 4.5))
m = df.groupby("City").agg(size=("Size_in_SqFt", "mean"), ret=("Annual_Growth_Rate", "mean"))
sns.regplot(data=m, x="size", y="ret", color="#2E86AB"); plt.xlabel("Avg size (sq ft)"); plt.ylabel("Avg projected annual growth")
save("x1_area_vs_return", "City avg area vs projected return")
say("Area vs investment return", f"r(Size, Annual_Growth_Rate) at listing level = {df[['Size_in_SqFt','Annual_Growth_Rate']].corr().iloc[0,1]:.3f}. "
    "Growth is driven by infrastructure, metro status, availability and age, not size.")
say("Crime rate vs Good Investment", "The dataset has **no crime-rate column**, so this cannot be tested directly. "
    "Infrastructure_Score (schools, hospitals, transport) is used as the area-quality proxy instead.")
plt.figure(figsize=(8, 4.5))
b = df.groupby(pd.cut(df.Infrastructure_Score, [0, .3, .45, .6, .75, 1.0])).agg(growth=("Annual_Growth_Rate", "mean"), ppsf=("Price_per_SqFt_INR", "mean"))
b.index = b.index.astype(str); ax1 = plt.gca(); ax1.bar(b.index, b.ppsf, color="#BBBBBB"); ax1.set_ylabel("Avg PPSF (INR)"); ax1.set_ylim(b.ppsf.min() * .95, b.ppsf.max() * 1.05)
ax2 = ax1.twinx(); ax2.plot(b.index, b.growth * 100, color="#C73E1D", marker="o"); ax2.set_ylabel("Projected growth %")
save("x2_infra_vs_value", "Infrastructure score vs price and projected growth")
say("Infrastructure score vs resale value", f"r(Infrastructure_Score, Price_in_Lakhs) = {df[['Infrastructure_Score','Price_in_Lakhs']].corr().iloc[0,1]:.3f} - no link to price; "
    "it only affects projected growth through the assumed model.")

section("Data-quality notes & caveats")
F.append(f"- **Floor_No > Total_Floors in {(df.Floor_Data_Flag==1).sum():,} rows ({df.Floor_Data_Flag.mean():.1%})** - impossible in reality. Floors were capped at Total_Floors and the original issue is kept in `Floor_Data_Flag`.\n"
         "- **Price_per_SqFt in the raw file is in lakhs and rounded to 2 dp** (0.05, 0.10 ...), so it was recomputed in INR from price and size.\n"
         "- **Locality names (Locality_1 ... Locality_500) repeat across all 42 cities**, so locality is only meaningful as City + Locality.\n"
         "- **Zero missing values and zero duplicates.**\n"
         "- **Features are statistically independent of price** (every correlation with price < 0.01), consistent with synthetic data. "
         "A model cannot learn real market structure from it. The 5-year price and the Good-Investment label are therefore built from documented rules "
         "(see `src/config.py`), and model scores measure how well the rule is recovered, not real-world forecasting skill.\n")

(REPORTS_DIR / "eda_findings.md").write_text("\n".join(F), encoding="utf-8")
print("EDA complete:", len(list(FIG_DIR.glob('*.png'))), "figures ->", FIG_DIR)
