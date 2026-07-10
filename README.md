# 🌱 OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a data-driven web application that recommends the most suitable
crop for a given plot of land based on soil nutrients (N, P, K) and climate
conditions (temperature, humidity, pH, rainfall). It is built with a
scikit-learn classification pipeline and a Flask front end, and covers three
usage scenarios:

| Scenario | Feature | Route |
|---|---|---|
| 1. Smart Crop Recommendation for Farmers | Enter soil/climate readings → get top-3 recommended crops with confidence scores | `/` , `/predict` |
| 2. Crop Suitability & Environmental Assessment | Check how well current conditions suit a *specific* crop, with parameter-level feedback | `/suitability` , `/suitability/check` |
| 3. Agricultural Research & Policy Planning | Dashboard of model performance, EDA plots, and crop-environment profiles | `/dashboard` |

## Tech Stack

- **NumPy / Pandas** — data generation, wrangling, statistics
- **Scikit-learn** — RandomForest / SVM / KNN / DecisionTree / LogisticRegression classifiers, model selection, cross-validation
- **Matplotlib / Seaborn** — EDA visualizations, confusion matrix, feature importance, model comparison charts
- **SciPy** — one-way ANOVA test on rainfall-by-crop to validate feature relevance
- **Flask** — web application server and templated UI

## Project Structure

```
OptiCrop/
├── app.py                     # Flask application (routes for all 3 scenarios)
├── requirements.txt
├── data/
│   ├── generate_dataset.py    # Synthetic agronomic dataset generator
│   └── crop_data.csv          # Generated dataset (22 crops x ~120 samples)
├── model/
│   ├── train_model.py         # Training pipeline: EDA, model selection, evaluation
│   ├── crop_model.pkl         # Serialized best model (generated)
│   ├── scaler.pkl             # StandardScaler (generated)
│   ├── crop_stats.json        # Per-crop mean/std used for suitability scoring
│   ├── metrics.json           # Model comparison & classification report
│   └── anova_rainfall.json    # SciPy ANOVA result
├── static/
│   ├── css/style.css
│   └── images/                # Auto-generated EDA & evaluation charts
└── templates/
    ├── base.html, index.html, result.html
    ├── suitability.html, suitability_result.html
    └── dashboard.html
```

## Setup & Run

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Re)generate the dataset — already included, but reproducible
python3 data/generate_dataset.py

# 4. Train the model — generates model/*.pkl and static/images/*.png
python3 model/train_model.py

# 5. Run the web app
python3 app.py
```

Then open **http://localhost:5000** in your browser.

## How the Model Was Built

1. **Dataset**: A 22-crop agronomic dataset (rice, maize, pulses, fruits, cotton,
   jute, coffee, etc.) was synthesized from published agronomic parameter ranges
   for N, P, K, temperature, humidity, pH, and rainfall (2,640 rows total).
2. **EDA**: Correlation heatmaps, feature distributions, and class balance were
   visualized with Seaborn/Matplotlib; a SciPy one-way ANOVA confirmed rainfall
   varies significantly across crop labels (supporting its predictive value).
3. **Model Selection**: Five classifiers (RandomForest, DecisionTree, KNN, SVM,
   LogisticRegression) were trained on standardized features and compared via
   5-fold cross-validation; the best performer (~95%+ accuracy) is auto-selected
   and persisted with `joblib`.
4. **Suitability Scoring**: For Scenario 2, each crop's per-feature mean/std from
   the training data is used to compute a Gaussian-based match score (0–100) for
   each input parameter, and an overall verdict (Highly Suitable / Moderately
   Suitable / Not Recommended).

## Notes

- The bundled dataset is **synthetically generated** from agronomic domain
  knowledge (no internet access was available while building this project) —
  it mirrors the structure of well-known crop-recommendation datasets. Swap in
  a real-world sensor/soil-survey dataset with the same column names
  (`N, P, K, temperature, humidity, ph, rainfall, label`) to retrain on
  real observations with zero code changes.
- Re-running `train_model.py` will overwrite the saved model, scaler, stats,
  and chart images with a fresh run.

## Demo & Repository Links

- **Live Demo:** _add your deployed URL here (e.g. Render/Heroku/PythonAnywhere)_
- **GitHub Repository:** _add your repository URL here_

> Update the two links above before submitting for mentor review.
