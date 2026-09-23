Fetal Health Classification - Healthon 2026

Classification pipeline and exploratory analysis for the UCI Cardiotocography (CTG) dataset using LightGBM.

Project Overview

This project predicts fetal health categories (1: Normal, 2: Suspect, 3: Pathological) from CTG metrics. It includes a training script that builds a LightGBM model inside a scikit-learn pipeline, exports the model object, and provides a testing script and notebook for evaluation.

Dataset: UCI CTG Dataset (ID 193)

Model: LightGBM (LGBMClassifier with class_weight='balanced')

Artifact: lightgbm_fetal_health_model.joblib (saved at project root)

Directory Structure

Healthon2026/
├── Code/
│   ├── train_model.py          # Trains pipeline and exports joblib model
│   └── testing.py              # Tests model loading and prediction
├── Notebook/
│   └── note.ipynb              # EDA and model development
├── lightgbm_fetal_health_model.joblib
├── requirements.txt
├── .gitignore
└── README.md


Setup & Running

Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS


Install dependencies:

pip install -r requirements.txt


Train the model:

python Code/train_model.py


Run model test to see what methodoldy is better:

python Code/testing.py


When working in Notebook/note.ipynb, select the project's .venv kernel in VS Code.
