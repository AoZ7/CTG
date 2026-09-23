import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

# Scikit-Learn Models
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# Specialized Gradient Boosting Libraries (pip install lightgbm xgboost catboost)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. Fetch & Clean Raw CTG Data
cardiotocography = fetch_ucirepo(id=193)
raw_df = pd.concat([cardiotocography.data.features, cardiotocography.data.targets["NSP"]], axis=1)

# Deduplicate
clean_df = raw_df.drop_duplicates().reset_index(drop=True)

# Separate features & zero-index target ({1,2,3} -> {0,1,2})
X = clean_df.drop(columns=["NSP"])
y = clean_df["NSP"].astype(int) - 1

# 2. CTG Feature Engineering
def transform_ctg_features(data: pd.DataFrame) -> pd.DataFrame:
    transformed = data.copy()
    transformed["FHR_range"] = transformed["Max"] - transformed["Min"]
    transformed["FHR_mean_deviation"] = transformed["Mean"] - transformed["LB"]
    transformed["variability_balance"] = transformed["ASTV"] / (transformed["ASTV"] + transformed["ALTV"] + 1e-6)
    transformed["activity_contraction_ratio"] = transformed["AC"] / (transformed["UC"] + 1e-6)
    transformed["deceleration_severity_score"] = (transformed["DL"] * 1.0) + (transformed["DS"] * 3.0) + (transformed["DP"] * 2.0)
    transformed["histogram_asymmetry"] = (transformed["Mean"] - transformed["Mode"]) / (transformed["Width"] + 1e-6)

    count_cols = ["AC", "FM", "UC", "DL", "DS", "DP", "MSTV", "MLTV", "Width", "Variance", "Nmax", "Nzeros"]
    for col in count_cols:
        if col in transformed.columns:
            transformed[f"{col}_log1p"] = np.log1p(transformed[col].clip(lower=0))

    return transformed.replace([np.inf, -np.inf], np.nan)

# Pre-transform feature matrix for uniform evaluation across models
X_transformed = transform_ctg_features(X)

# 3. Define Top 5 Candidate Models
models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, max_iter=300, learning_rate=0.03
    ),
    "LightGBM": LGBMClassifier(
        class_weight="balanced", random_state=42, n_estimators=300, learning_rate=0.03, verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        auto_class_weights="Balanced", random_state=42, iterations=300, learning_rate=0.03, verbose=0
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_estimators=300, learning_rate=0.03
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=42, n_estimators=300
    )
}

# 4. Stratified 5-Fold Cross-Validation Comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("Evaluating Top 5 Models via 5-Fold Stratified CV...\n")

for name, model in models.items():
    # Apply sample weights specifically for XGBoost multi-class imbalance
    fit_params = {}
    if name == "XGBoost":
        sample_weights = compute_sample_weight("balanced", y)
        fit_params = {"classifier__sample_weight": sample_weights}

    pipeline = Pipeline([("classifier", model)])
    
    cv_scores = cross_validate(
        pipeline,
        X_transformed,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_recall": "recall_macro",
            "macro_f1": "f1_macro"
        },
        params=fit_params
    )
    
results.append({
        "Model": name,
        "Accuracy": f"{cv_scores['test_accuracy'].mean():.4f} +/- {cv_scores['test_accuracy'].std():.4f}",
        "Macro Recall": f"{cv_scores['test_macro_recall'].mean():.4f} +/- {cv_scores['test_macro_recall'].std():.4f}",
        "Macro F1": f"{cv_scores['test_macro_f1'].mean():.4f} +/- {cv_scores['test_macro_f1'].std():.4f}"
    })

# Display Leaderboard
leaderboard = pd.DataFrame(results).sort_values(by="Macro Recall", ascending=False)
print("=== TOP 5 MODEL LEADERBOARD ===")
print(leaderboard.to_string(index=False))


####
from pathlib import Path
import joblib

# Points to project root (Healthon2026/)
project_root = Path.cwd().parent if "Notebook" in str(Path.cwd()) else Path.cwd()
model_path = project_root / "lightgbm_fetal_health_model.joblib"

model = joblib.load(model_path)