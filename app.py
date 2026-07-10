"""
OptiCrop - Smart Agricultural Production Optimization Engine
---------------------------------------------------------------
Flask web application implementing:
  Scenario 1: Smart Crop Recommendation for Farmers
  Scenario 2: Crop Suitability & Environmental Assessment
  Scenario 3: Agricultural Research & Policy Planning Dashboard
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N": "Nitrogen (kg/ha)",
    "P": "Phosphorous (kg/ha)",
    "K": "Potassium (kg/ha)",
    "temperature": "Temperature (\u00b0C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "crop_stats.json")) as fh:
    CROP_STATS = json.load(fh)

with open(os.path.join(MODEL_DIR, "metrics.json")) as fh:
    METRICS = json.load(fh)

CROP_LIST = sorted(CROP_STATS.keys())


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_form_values(form):
    values = {}
    for f in FEATURES:
        values[f] = float(form.get(f))
    return values


def predict_top_crops(values, top_k=3):
    X = pd.DataFrame([values])[FEATURES]
    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
        return [(c, round(float(p) * 100, 2)) for c, p in ranked[:top_k]]
    else:
        pred = model.predict(X_scaled)[0]
        return [(pred, 100.0)]


def suitability_score(values, crop):
    """
    Compute a 0-100 suitability score for a given crop based on how many
    standard deviations each input parameter is from that crop's ideal
    (training-data) mean. Uses a Gaussian-like scoring curve per feature,
    then averages across all features.
    """
    stats = CROP_STATS[crop]
    feature_scores = {}
    for f in FEATURES:
        mean = stats[f]["mean"]
        std = stats[f]["std"] or 1e-6
        z = (values[f] - mean) / std
        # Gaussian falloff: z=0 -> 100, z=1 -> ~60, z=2 -> ~14
        score = 100 * np.exp(-0.5 * (z ** 2))
        feature_scores[f] = {
            "score": round(float(score), 1),
            "value": values[f],
            "ideal_mean": mean,
            "ideal_std": std,
            "status": "optimal" if abs(z) <= 0.5 else ("acceptable" if abs(z) <= 1.5 else "unsuitable"),
        }
    overall = round(float(np.mean([v["score"] for v in feature_scores.values()])), 1)
    return overall, feature_scores


# ---------------------------------------------------------------------
# Scenario 1: Smart Crop Recommendation
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURES, labels=FEATURE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    values = get_form_values(request.form)
    top_crops = predict_top_crops(values, top_k=3)
    best_crop, best_conf = top_crops[0]

    return render_template(
        "result.html",
        values=values,
        labels=FEATURE_LABELS,
        best_crop=best_crop,
        best_conf=best_conf,
        top_crops=top_crops,
    )


# ---------------------------------------------------------------------
# Scenario 2: Crop Suitability & Environmental Assessment
# ---------------------------------------------------------------------
@app.route("/suitability", methods=["GET"])
def suitability_form():
    return render_template(
        "suitability.html", features=FEATURES, labels=FEATURE_LABELS, crops=CROP_LIST
    )


@app.route("/suitability/check", methods=["POST"])
def suitability_check():
    values = get_form_values(request.form)
    crop = request.form.get("crop")
    overall, feature_scores = suitability_score(values, crop)

    if overall >= 80:
        verdict = "Highly Suitable"
    elif overall >= 55:
        verdict = "Moderately Suitable"
    else:
        verdict = "Not Recommended"

    return render_template(
        "suitability_result.html",
        crop=crop,
        overall=overall,
        verdict=verdict,
        feature_scores=feature_scores,
        labels=FEATURE_LABELS,
    )


# ---------------------------------------------------------------------
# Scenario 3: Agricultural Research & Policy Planning Dashboard
# ---------------------------------------------------------------------
@app.route("/dashboard", methods=["GET"])
def dashboard():
    has_feature_importance = os.path.exists(
        os.path.join(BASE_DIR, "static", "images", "feature_importance.png")
    )
    return render_template(
        "dashboard.html",
        metrics=METRICS,
        crop_stats=CROP_STATS,
        crop_list=CROP_LIST,
        has_feature_importance=has_feature_importance,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
