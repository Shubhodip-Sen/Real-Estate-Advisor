"""Central configuration: paths + business assumptions (edit here, re-run pipeline)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "housing_clean.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

CURRENT_YEAR = 2025          # dataset: Age_of_Property = 2025 - Year_Built
HORIZON_YEARS = 5            # forecast horizon required by the brief
LAKH = 100_000               # 1 lakh = 100,000 INR

METRO_CITIES = {"Mumbai", "Delhi", "New Delhi", "Bangalore", "Hyderabad", "Chennai",
                "Kolkata", "Pune", "Gurgaon", "Noida"}

# ---- Assumed appreciation model ---------------------------------------------
# The dataset has no historical prices, so the 5-year target is scenario-based
# and fully documented here. Change the numbers, re-run the pipeline.
GROWTH = {
    "base": 0.05,                # baseline annual appreciation
    "infra_weight": 0.03,        # + up to 3 pts for a top infrastructure score
    "metro_bonus": 0.01,         # + 1 pt for metro cities
    "under_construction": 0.01,  # + 1 pt upside for under-construction stock
    "age_penalty": 0.0005,       # - 0.05 pt per year of building age
    "min": 0.03, "max": 0.12,
}
# "Good Investment" rule: priced at/below its city median per sq ft
# AND projected annual appreciation >= threshold.
GOOD_INVESTMENT_GROWTH_THRESHOLD = 0.0625

TRANSPORT_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
