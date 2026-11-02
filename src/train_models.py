"""Step 3 - Model training, evaluation and MLflow tracking.

Run:  python src/train_models.py          (after src/preprocess.py)
View: mlflow ui --backend-store-uri sqlite:///mlflow.db     (from the project root)

Two tasks
  * Classification : Good_Investment (0/1)
  * Regression     : Future_Price_5Y_Lakhs
Both targets are rule-based (see src/config.py) because the dataset has no price history,
so scores show how well each model recovers the documented rule, not real-market skill.
A third "sanity" run tries to predict today's price from the other features to show how little
real signal the raw data contains.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: E402,F403

SEED = 42
CAT = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility", "Facing",
       "Owner_Type", "Availability_Status", "Parking_Space", "Security"]
NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Floor_No",
       "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score",
       "Is_Metro", "Floor_Ratio", "Size_per_BHK"] + [f"Has_{a}" for a in AMENITY_LIST]
FEATURES = CAT + NUM


def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


def grouped_importance(pipe) -> dict:
    """Map model importances on one-hot columns back to the original input features."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    out = {}
    for n, v in zip(names, imp):
        kind, rest = n.split("__", 1)
        base = next((c for c in CAT if rest.startswith(c + "_")), rest) if kind == "cat" else rest
        out[base] = out.get(base, 0.0) + float(v)
    tot = sum(out.values()) or 1
    return dict(sorted(((k, v / tot) for k, v in out.items()), key=lambda x: -x[1]))


def main():
    df = pd.read_csv(CLEAN_CSV)
    X = df[FEATURES]
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("Real_Estate_Investment_Advisor")

    # ------------------------------------------------------------------ classification
    y = df["Good_Investment"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=500, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss"),
    }
    results = {"classification": {}, "regression": {}, "sanity": {}}
    best_clf, best_auc = None, -1
    for name, m in clf_models.items():
        pipe = Pipeline([("pre", make_pre()), ("model", m)])
        with mlflow.start_run(run_name=f"clf_{name}"):
            mlflow.set_tags({"task": "classification", "target": "Good_Investment"})
            mlflow.log_params({"model": name, "n_features": len(FEATURES), "train_rows": len(Xtr), **{k: v for k, v in m.get_params().items() if k in ("n_estimators", "max_depth", "learning_rate", "C", "min_samples_leaf")}})
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xte); pr = pipe.predict_proba(Xte)[:, 1]
            met = {"accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p), "recall": recall_score(yte, p),
                   "f1": f1_score(yte, p), "roc_auc": roc_auc_score(yte, pr)}
            mlflow.log_metrics(met)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            results["classification"][name] = {k: round(float(v), 4) for k, v in met.items()}
            print(f"[clf] {name:18s}", {k: round(v, 4) for k, v in met.items()})
            if met["roc_auc"] > best_auc:
                best_auc, best_clf, best_clf_name = met["roc_auc"], pipe, name

    # ------------------------------------------------------------------ regression
    y = df["Future_Price_5Y_Lakhs"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    reg_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=10, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                                colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=SEED),
    }
    best_reg, best_rmse = None, 1e18
    for name, m in reg_models.items():
        pipe = Pipeline([("pre", make_pre()), ("model", m)])
        with mlflow.start_run(run_name=f"reg_{name}"):
            mlflow.set_tags({"task": "regression", "target": "Future_Price_5Y_Lakhs"})
            mlflow.log_params({"model": name, "n_features": len(FEATURES), "train_rows": len(Xtr), **{k: v for k, v in m.get_params().items() if k in ("n_estimators", "max_depth", "learning_rate", "min_samples_leaf")}})
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xte)
            met = {"rmse": float(np.sqrt(mean_squared_error(yte, p))), "mae": mean_absolute_error(yte, p), "r2": r2_score(yte, p)}
            mlflow.log_metrics(met)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            results["regression"][name] = {k: round(float(v), 4) for k, v in met.items()}
            print(f"[reg] {name:18s}", {k: round(v, 4) for k, v in met.items()})
            if met["rmse"] < best_rmse:
                best_rmse, best_reg, best_reg_name, best_pred = met["rmse"], pipe, name, p

    resid_pct = (best_pred - yte.values) / yte.values * 100
    results["regression_residual_pct"] = {"p05": round(float(np.percentile(resid_pct, 5)), 3),
                                          "p95": round(float(np.percentile(resid_pct, 95)), 3)}

    # ------------------------------------------------------------------ sanity: can the data predict TODAY's price?
    feats_no_price = [c for c in FEATURES if c not in ("Price_in_Lakhs", "Price_per_SqFt_INR")]
    y = df["Price_in_Lakhs"]
    Xtr, Xte, ytr, yte = train_test_split(df[feats_no_price], y, test_size=0.2, random_state=SEED)
    cat_s = [c for c in CAT]; num_s = [c for c in NUM if c in feats_no_price]
    pre = ColumnTransformer([("num", StandardScaler(), num_s), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_s)])
    pipe = Pipeline([("pre", pre), ("model", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, tree_method="hist", n_jobs=-1, random_state=SEED))])
    with mlflow.start_run(run_name="sanity_predict_current_price"):
        mlflow.set_tags({"task": "sanity_check", "target": "Price_in_Lakhs"})
        pipe.fit(Xtr, ytr); p = pipe.predict(Xte)
        met = {"rmse": float(np.sqrt(mean_squared_error(yte, p))), "r2": r2_score(yte, p)}
        mlflow.log_metrics(met)
        results["sanity"] = {"target": "Price_in_Lakhs (current)", **{k: round(float(v), 4) for k, v in met.items()},
                             "baseline_rmse_predict_mean": round(float(yte.std()), 4)}
        print("[sanity] current-price R2 from other features:", round(met["r2"], 4))

    # ------------------------------------------------------------------ persist for the Streamlit app
    # The Streamlit app always serves XGBoost (compact file, native importances) so the UI is consistent.
    xgb_clf = next(n for n in clf_models if n == "XGBoost")
    final_clf = Pipeline([("pre", make_pre()), ("model", clf_models["XGBoost"])])
    final_reg = Pipeline([("pre", make_pre()), ("model", reg_models["XGBoost"])])
    final_clf.fit(X, df["Good_Investment"]); final_reg.fit(X, df["Future_Price_5Y_Lakhs"])
    joblib.dump(final_clf, MODELS_DIR / "clf_pipeline.joblib", compress=3)
    joblib.dump(final_reg, MODELS_DIR / "reg_pipeline.joblib", compress=3)
    results["best_classifier_by_auc"] = best_clf_name
    results["best_regressor_by_rmse"] = best_reg_name
    results["feature_importance_clf"] = grouped_importance(final_clf)
    results["feature_importance_reg"] = grouped_importance(final_reg)
    results["features"] = {"categorical": CAT, "numeric": NUM}
    results["class_balance_good_investment"] = round(float(df["Good_Investment"].mean()), 4)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved models + metrics ->", MODELS_DIR)
    print("Best classifier:", best_clf_name, "| best regressor:", best_reg_name)


if __name__ == "__main__":
    main()
