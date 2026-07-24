"""
app.py - House Price Prediction Backend API

Run:
    python app.py
Serves on:
    http://localhost:5000
"""

import json
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "model", "saved")
MODEL_DIR = os.path.join(HERE, "..", "model")
FRONTEND_DIR = os.path.join(HERE, "..", "frontend")

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


MODEL = None
BASE_MODELS = None
VOTING_MODEL = None
STACKING_MODEL = None
METADATA = None
LOAD_ERROR = None

REQUIRED_NUMERIC = [
    "area_sqft", "bedrooms", "bathrooms", "stories",
    "garage_spaces", "age_years", "location_score",
    "has_pool", "has_basement", "near_main_road",
]
REQUIRED_CATEGORICAL = ["furnishing", "condition"]


def load_artifacts(retrain_if_missing=True):
    global MODEL, BASE_MODELS, VOTING_MODEL, STACKING_MODEL, METADATA, LOAD_ERROR
    try:
        MODEL = joblib.load(os.path.join(SAVED_DIR, "best_model.joblib"))
        BASE_MODELS = joblib.load(os.path.join(SAVED_DIR, "base_models.joblib"))
        VOTING_MODEL = joblib.load(os.path.join(SAVED_DIR, "voting_model.joblib"))
        STACKING_MODEL = joblib.load(os.path.join(SAVED_DIR, "stacking_model.joblib"))
        with open(os.path.join(SAVED_DIR, "metadata.json")) as f:
            METADATA = json.load(f)
        print(f"Loaded model artifacts from {SAVED_DIR}")
        print(f"Best model in production: {METADATA['best_model']}")
        LOAD_ERROR = None
    except Exception as e:
        if retrain_if_missing:

            print("No saved model found - training it now (first boot only)...")
            sys.path.insert(0, MODEL_DIR)
            import train_model as _train_model
            _train_model.main()
            load_artifacts(retrain_if_missing=False)
        else:
            LOAD_ERROR = str(e)
            print("=" * 70)
            print("WARNING: Could not load model artifacts, even after training.")
            print(f"Details: {e}")
            print("=" * 70)


load_artifacts()


def validate_payload(data):
    missing = [f for f in REQUIRED_NUMERIC + REQUIRED_CATEGORICAL if f not in data]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"

    for f in REQUIRED_NUMERIC:
        try:
            float(data[f])
        except (TypeError, ValueError):
            return f"Field '{f}' must be numeric"

    if METADATA:
        for f in REQUIRED_CATEGORICAL:
            allowed = METADATA["categorical_options"].get(f, [])
            if allowed and data[f] not in allowed:
                return f"Field '{f}' must be one of {allowed}"

    return None


def to_dataframe(data):
    row = {f: float(data[f]) for f in REQUIRED_NUMERIC}
    row.update({f: data[f] for f in REQUIRED_CATEGORICAL})
    return pd.DataFrame([row])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
        "error": LOAD_ERROR,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    if METADATA is None:
        return jsonify({"error": "Model metadata not available. Train the model first."}), 503
    return jsonify(METADATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    error = validate_payload(data)
    if error:
        return jsonify({"error": error}), 400

    X = to_dataframe(data)

    final_prediction = float(MODEL.predict(X)[0])

    appraisers = {}
    for name, pipe in BASE_MODELS.items():
        appraisers[name] = round(float(pipe.predict(X)[0]), 2)
    appraisers["Voting Ensemble"] = round(float(VOTING_MODEL.predict(X)[0]), 2)
    appraisers["Stacking Ensemble"] = round(float(STACKING_MODEL.predict(X)[0]), 2)

    values = list(appraisers.values())
    spread = round(max(values) - min(values), 2)

    return jsonify({
        "predicted_price": round(final_prediction, 2),
        "best_model_used": METADATA["best_model"],
        "appraiser_estimates": appraisers,
        "spread": spread,
        "input_echo": data,
    })


@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
