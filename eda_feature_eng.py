"""
EDA & Feature Engineering module with callable class + reusable transformer.

Provides:
- FeatureEngineer (sklearn Transformer): adds engineered features used by the model.
- EDAFeatureEngineer (callable): runs lightweight EDA to a report file, and
  optionally writes a processed CSV with engineered features.

Usage as a module:
    from eda_feature_eng import FeatureEngineer, EDAFeatureEngineer

    # 1) Use transformer in a Pipeline (training/inference)
    fe = FeatureEngineer()

    # 2) Run EDA and optionally save engineered CSV
    runner = EDAFeatureEngineer()
    runner(input_csv="data/raw/diabetes_prediction_dataset.csv",
           output_csv="data/processed/diabetes_with_features.csv",
           report_path="reports/eda_summary.txt",
           save_processed=True)

CLI:
    python eda_feature_eng.py --input data/raw/diabetes_prediction_dataset.csv \
                              --output data/processed/diabetes_with_features.csv \
                              --report reports/eda_summary.txt
"""
from __future__ import annotations
import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd
import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    fmt = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.input_features_: Optional[Iterable[str]] = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.input_features_ = list(X.columns)
        logger.debug("FeatureEngineer.fit() with features: %s", self.input_features_)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        logger.debug("Transform input shape: %s", df.shape)
        if "age" in df.columns:
            df["is_senior"] = (df["age"] >= 60).astype(int)
        else:
            df["is_senior"] = 0

        if all(col in df.columns for col in ["HbA1c_level", "blood_glucose_level"]):
            df["glucose_hba1c_interaction"] = df["HbA1c_level"] * df["blood_glucose_level"]
        else:
            df["glucose_hba1c_interaction"] = 0.0
        logger.debug("Transform output shape: %s", df.shape)
        return df

    def get_feature_names_out(self, input_features=None):
        base = (
            list(self.input_features_)
            if self.input_features_ is not None
            else (list(input_features) if input_features is not None else [])
        )
        return np.array(base + ["is_senior", "glucose_hba1c_interaction"])


@dataclass
class EDAFeatureEngineer:
    """Callable EDA runner + optional engineered CSV writer."""
    report_path: str = "reports/eda_summary.txt"
    save_processed: bool = True

    def __call__(self, input_csv: str, output_csv: Optional[str] = None,
                 report_path: Optional[str] = None, save_processed: Optional[bool] = None):
        if report_path is not None:
            self.report_path = report_path
        if save_processed is not None:
            self.save_processed = save_processed

        df = self._run_initial_eda(input_csv, self.report_path)
        if self.save_processed and output_csv is not None:
            self._save_with_features(input_csv, output_csv)
        return df

    @staticmethod
    def _run_initial_eda(input_csv: str, report_path: str):
        path = pathlib.Path(input_csv)
        logger.info("Running EDA for %s", path)
        df = pd.read_csv(path)

        lines = []
        lines.append(f"File: {path}")
        lines.append(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        lines.append("\nColumn dtypes:\n" + df.dtypes.to_string())

        na_counts = df.isna().sum()
        if na_counts.sum() > 0:
            lines.append("\nMissing values per column:\n" + na_counts.to_string())
            logger.warning("Missing values detected in dataset.")
        else:
            lines.append("\nNo missing values detected.")

        lines.append("\nNumeric describe():\n" + df.describe().to_string())

        if "diabetes" in df.columns:
            lines.append("\nTarget value counts (diabetes):\n" + df["diabetes"].value_counts().to_string())

        report_file = pathlib.Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("\n".join(lines))

        logger.info("EDA summary written to %s", report_file)
        return df

    @staticmethod
    def _save_with_features(input_csv: str, output_csv: str):
        df = pd.read_csv(input_csv)
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(df)
        out = pathlib.Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(out, index=False)
        logger.info("Engineered dataset saved: %s", out)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/diabetes_prediction_dataset.csv")
    p.add_argument("--output", default="data/processed/diabetes_with_features.csv")
    p.add_argument("--report", default="reports/eda_summary.txt")
    p.add_argument("--no-save-processed", action="store_true",
                   help="Only run EDA, do not save engineered CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = EDAFeatureEngineer(report_path=args.report, save_processed=(not args.no_save_processed))
    runner(input_csv=args.input, output_csv=args.output)
