import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# 1. Fetch & Clean Dataset
cardiotocography = fetch_ucirepo(id=193)
raw_X = cardiotocography.data.features.copy()
raw_y = cardiotocography.data.targets["NSP"].copy()

# Combine, deduplicate, and fix negative count values
df_clean = pd.concat([raw_X, raw_y], axis=1).drop_duplicates().reset_index(drop=True)
count_cols = ["AC", "FM", "UC", "DL", "DS", "DP", "MSTV", "MLTV", "Nmax", "Nzeros"]
for col in count_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(lower=0)

X = df_clean.drop(columns=["NSP"])
y = df_clean["NSP"].astype(int) - 1  # Remap {1, 2, 3} -> zero-indexed {0, 1, 2}

# 2. Leak-Free CTG Feature Engineering
def transform_ctg_features(data: pd.DataFrame) -> pd.DataFrame:
    transformed = data.copy()

    # Clinical ratios & heart-rate spreads
    transformed["FHR_range"] = transformed["Max"] - transformed["Min"]
    transformed["FHR_mean_deviation"] = transformed["Mean"] - transformed["LB"]
    transformed["heart_rate_center_offset"] = transformed["Mode"] - transformed["LB"]
    transformed["variability_balance"] = transformed["ASTV"] / (transformed["ASTV"] + transformed["ALTV"] + 1e-6)
    transformed["activity_contraction_ratio"] = transformed["AC"] / (transformed["UC"] + 1e-6)
    transformed["deceleration_severity_score"] = (transformed["DL"] * 1.0) + (transformed["DS"] * 3.0) + (transformed["DP"] * 2.0)
    transformed["histogram_asymmetry"] = (transformed["Mean"] - transformed["Mode"]) / (transformed["Width"] + 1e-6)

    # Log1p transforms for skewed count features
    for col in count_cols:
        if col in transformed.columns:
            transformed[f"{col}_log1p"] = np.log1p(transformed[col].clip(lower=0))

    return transformed.replace([np.inf, -np.inf], np.nan)

# 3. Construct Model Pipeline
model = Pipeline(
    steps=[
        ("feature_engineering", FunctionTransformer(transform_ctg_features, validate=False)),
        (
            "classifier",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                class_weight="balanced",  # Handles imbalanced class distributions
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
    ]
)

# 4. Train/Test Split & Fit Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Pathological"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Inference Function API
def predict_fetal_health(new_measurements: pd.DataFrame) -> pd.DataFrame:
    """Predict fetal health risk from raw CTG measurements."""
    risk_labels = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    preds = model.predict(new_measurements[X.columns])
    probs = model.predict_proba(new_measurements[X.columns])

    return pd.DataFrame(
        {
            "predicted_nsp_code": preds + 1,  # Return to {1, 2, 3} scale
            "risk_label": [risk_labels[p] for p in preds],
            "confidence": probs.max(axis=1),
            "prob_normal": probs[:, 0],
            "prob_suspect": probs[:, 1],
            "prob_pathological": probs[:, 2],
        },
        index=new_measurements.index,
    )

# 7. Save Pipeline
model_path = Path.cwd() / "lightgbm_fetal_health_model.joblib"
joblib.dump(model, model_path)
print(f"\nSaved trained model to: {model_path}")

####
from pathlib import Path
import joblib

# Points to project root (Healthon2026/)
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "lightgbm_fetal_health_model.joblib"

joblib.dump(model, model_path)