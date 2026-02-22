"""
Model training module.

- Imports callable classes from data_gathering.py and eda_feature_eng.py
  so you can orchestrate the full pipeline in Python without shelling out.
- Uses FeatureEngineer inside an sklearn Pipeline (training/inference parity).
- Train/test split, evaluation, and save to /var/data/diabetes_pipeline.joblib.
"""
from __future__ import annotations
import json
import os
import pathlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from data_gathering import DataGatherer
from eda_feature_eng import FeatureEngineer, EDAFeatureEngineer


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    fmt = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")  # Sentry format
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if os.getenv("ENABLE_FILE_LOGS", "0") == "1":
        log_dir = os.getenv("LOG_DIR", "logs")
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(pathlib.Path(log_dir) / f"{__name__}.log", mode="w")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

logger = _setup_logger()


# ----------------- Config -----------------
RAW_CSV = "data/raw/diabetes_prediction_dataset.csv"
MODEL_DIR = "/home/jefft/CST_605_Week6/var/data"
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_pipeline.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.20
USE_DECISION_TREE = False
RUN_EDA = True 
# -----------------------------------------

def ensure_data(csv_path: str = RAW_CSV) -> str:
    """Ensure the raw CSV exists; if not, use DataGatherer to fetch it."""
    if not os.path.exists(csv_path):
        logger.warning("Raw CSV not found at %s; invoking DataGatherer ...", csv_path)
        _ = DataGatherer(output_dir=os.path.dirname(csv_path))()
    else:
        logger.info("Found raw CSV at %s", csv_path)
    return csv_path


def maybe_run_eda(csv_path: str):
    if RUN_EDA:
        runner = EDAFeatureEngineer(report_path="reports/eda_summary.txt", save_processed=True)
        runner(input_csv=csv_path, output_csv="data/processed/diabetes_with_features.csv")


def build_pipeline() -> Pipeline:
    logger.debug("Building training pipeline...")
    feature_engineer = FeatureEngineer()

    numeric_features = [
        "age", "bmi", "HbA1c_level", "blood_glucose_level",
        # engineered numeric/binary added by FeatureEngineer
        "is_senior", "glucose_hba1c_interaction",
        # binary numeric passthrough
        "hypertension", "heart_disease",
    ]
    categorical_features = ["gender", "smoking_history"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    if USE_DECISION_TREE:
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[
        ("fe", feature_engineer),
        ("prep", preprocessor),
        ("clf", model),
    ])
    logger.debug("Pipeline ready.")
    return pipe


def train_and_save(csv_path: str = RAW_CSV):
    pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    csv_path = ensure_data(csv_path)
    maybe_run_eda(csv_path)
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    if "diabetes" not in df.columns:
        logger.error("Target column 'diabetes' missing in dataset.")
        raise ValueError("Target column 'diabetes' is missing from the dataset.")

    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes"])
    logger.info("Train/test split (test_size=%.2f, stratify=y)", TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    logger.info("Fitting pipeline...")
    pipe.fit(X_train, y_train)
    logger.info("Evaluating...")
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "DecisionTreeClassifier" if USE_DECISION_TREE else "LogisticRegression",
    }

    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception as e:
        logger.exception("Error while computing ROC-AUC: %s", e)
    
    logger.info("Test metrics:\n%s", json.dumps(metrics, indent=2))

    joblib.dump(pipe, MODEL_PATH)
    logger.info("Saved trained pipeline to: %s", MODEL_PATH)


if __name__ == "__main__":
    train_and_save()
