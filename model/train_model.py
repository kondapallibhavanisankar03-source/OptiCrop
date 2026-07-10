"""
OptiCrop - Model Training Pipeline
-----------------------------------
Trains and evaluates crop-recommendation classifiers, generates EDA /
evaluation visualizations, and persists the best model + supporting
artifacts (scaler, crop statistics) for use by the Flask application.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

os.makedirs(IMG_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def run_eda(df):
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=110)
    plt.close()

    # Distribution of each feature
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#2e7d32")
        ax.set_title(col)
    for ax in axes.flat[len(FEATURES):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_distributions.png"), dpi=110)
    plt.close()

    # Crop counts
    plt.figure(figsize=(10, 6))
    df["label"].value_counts().sort_values().plot(kind="barh", color="#558b2f")
    plt.title("Samples per Crop")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "crop_distribution.png"), dpi=110)
    plt.close()

    # Simple ANOVA (scipy.stats) to show statistical spread of rainfall across crops
    groups = [g["rainfall"].values for _, g in df.groupby("label")]
    f_val, p_val = stats.f_oneway(*groups)
    with open(os.path.join(MODEL_DIR, "anova_rainfall.json"), "w") as fh:
        json.dump({"f_statistic": float(f_val), "p_value": float(p_val)}, fh, indent=2)


def train_and_select_model(df):
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=250, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cv = cross_val_score(clf, scaler.transform(X), y, cv=5).mean()
        results[name] = {"accuracy": acc, "f1_weighted": f1, "cv_mean": cv}
        print(f"{name:20s} acc={acc:.4f}  f1={f1:.4f}  cv={cv:.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}")

    # Final evaluation plots for the best model
    preds = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=110)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURES)
        importances = importances.sort_values()
        plt.figure(figsize=(8, 5))
        importances.plot(kind="barh", color="#33691e")
        plt.title(f"Feature Importance ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=110)
        plt.close()

    report = classification_report(y_test, preds, output_dict=True)

    # Model comparison bar chart
    plt.figure(figsize=(9, 5))
    comp_df = pd.DataFrame(results).T
    comp_df["cv_mean"].sort_values().plot(kind="barh", color="#00695c")
    plt.title("Model Comparison (5-fold CV Accuracy)")
    plt.xlabel("Mean CV Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=110)
    plt.close()

    return best_model, best_name, scaler, results, report


def save_crop_statistics(df):
    """Per-crop mean/std used later for the suitability-assessment feature."""
    stats_df = df.groupby("label")[FEATURES].agg(["mean", "std"])
    stats_dict = {}
    for crop in stats_df.index:
        stats_dict[crop] = {
            feat: {
                "mean": round(float(stats_df.loc[crop, (feat, "mean")]), 2),
                "std": round(float(stats_df.loc[crop, (feat, "std")]), 2),
            }
            for feat in FEATURES
        }
    with open(os.path.join(MODEL_DIR, "crop_stats.json"), "w") as fh:
        json.dump(stats_dict, fh, indent=2)
    return stats_dict


def main():
    df = load_data()
    run_eda(df)
    best_model, best_name, scaler, results, report = train_and_select_model(df)
    save_crop_statistics(df)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "crop_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump({
            "best_model": best_name,
            "model_comparison": results,
            "classification_report": report,
        }, fh, indent=2)

    print("\nSaved model artifacts to /model")
    print("Saved EDA + evaluation plots to /static/images")


if __name__ == "__main__":
    main()
