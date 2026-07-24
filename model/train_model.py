"""
train_model.py

Run:
    python train_model.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "housing_data.csv")
SAVE_DIR = os.path.join(HERE, "saved")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "stories",
    "garage_spaces", "age_years", "location_score",
    "has_pool", "has_basement", "near_main_road",
]
CATEGORICAL_FEATURES = ["furnishing", "condition"]
TARGET = "price"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("Dataset not found - generating it now...")
        sys.path.insert(0, os.path.join(HERE, "..", "data"))
        from generate_data import generate_housing_data
        df = generate_housing_data()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, model, X_test, y_test, results):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}
    print(f"{name:28s} | R2: {r2:.4f} | MAE: {mae:,.0f} | RMSE: {rmse:,.0f}")
    return r2


def main():
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - ENSEMBLE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    df = load_data()
    print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows\n")

    preprocessor = build_preprocessor()

    base_learners = {
        "Random Forest (bagging)": RandomForestRegressor(
            n_estimators=120, max_depth=12, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees (bagging)": ExtraTreesRegressor(
            n_estimators=120, max_depth=12, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting (boosting)": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE,
        ),
        "AdaBoost (boosting)": AdaBoostRegressor(
            n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        base_learners["XGBoost (boosting)"] = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        print("(xgboost not installed in this environment - skipping; "
              "install it via requirements.txt for the full ensemble)\n")

    results = {}
    fitted_pipelines = {}

    print("--- Training individual base learners ---")
    for name, estimator in base_learners.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        evaluate(name, pipe, X_test, y_test, results)
    print(f"(trained {len(base_learners)} base learners in the section above)\n")

    voting_estimators = [
        (key.split(" ")[0].lower(), est) for key, est in base_learners.items()
    ]
    voting_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", VotingRegressor(estimators=voting_estimators, n_jobs=-1)),
    ])
    print("--- Training VotingRegressor ensemble ---")
    voting_pipe.fit(X_train, y_train)
    evaluate("VOTING ENSEMBLE", voting_pipe, X_test, y_test, results)

    # ---- ENSEMBLE #2: Stacking Regressor (meta-model combines them) --
    stacking_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", StackingRegressor(
            estimators=voting_estimators,
            final_estimator=Ridge(alpha=1.0),
            n_jobs=-1,
            passthrough=False,
        )),
    ])
    print("\n--- Training StackingRegressor ensemble ---")
    stacking_pipe.fit(X_train, y_train)
    evaluate("STACKING ENSEMBLE", stacking_pipe, X_test, y_test, results)

    all_pipelines = dict(fitted_pipelines)
    all_pipelines["VOTING ENSEMBLE"] = voting_pipe
    all_pipelines["STACKING ENSEMBLE"] = stacking_pipe

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipeline = all_pipelines[best_name]
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}  (R2 = {results[best_name]['r2']})")
    print("=" * 70)

    feature_names = (
        NUMERIC_FEATURES
        + list(
            fitted_pipelines["Random Forest (bagging)"]
            .named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
    )
    rf_model = fitted_pipelines["Random Forest (bagging)"].named_steps["model"]
    importances = dict(zip(feature_names, rf_model.feature_importances_.round(4).tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # ---- Persist artifacts ---------------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(SAVE_DIR, "best_model.joblib"))
    joblib.dump(fitted_pipelines, os.path.join(SAVE_DIR, "base_models.joblib"))
    joblib.dump(voting_pipe, os.path.join(SAVE_DIR, "voting_model.joblib"))
    joblib.dump(stacking_pipe, os.path.join(SAVE_DIR, "stacking_model.joblib"))

    metadata = {
        "best_model": best_name,
        "metrics": results,
        "feature_importances": importances,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": {
            "furnishing": sorted(df["furnishing"].unique().tolist()),
            "condition": sorted(df["condition"].unique().tolist()),
        },
        "xgboost_available": XGBOOST_AVAILABLE,
        "target": TARGET,
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
    }
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifacts to: {SAVE_DIR}")
    print(" - best_model.joblib      (deployed to the API)")
    print(" - base_models.joblib     (individual 'appraiser' models)")
    print(" - voting_model.joblib")
    print(" - stacking_model.joblib")
    print(" - metadata.json          (metrics + feature importances)")


if __name__ == "__main__":
    main()
