"""
Data gathering module (callable class) for the Diabetes Prediction project.

Provides:
- DataGatherer: a callable class that downloads the Kaggle dataset via kagglehub
  and copies the main CSV into a local folder (default: data/raw/). It returns
  the local CSV path when called.

Usage as a module:
    from data_gathering import DataGatherer
    csv_path = DataGatherer()()

CLI:
    python data_gathering.py --output data/raw
"""
from __future__ import annotations
import argparse
import os
import shutil
import pathlib
import logging
from dataclasses import dataclass
import kagglehub

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


@dataclass
class DataGatherer:
    dataset_id: str = "iammustafatz/diabetes-prediction-dataset"
    csv_name: str = "diabetes_prediction_dataset.csv"
    output_dir: str = "data/raw"
    def __call__(self) -> str:
        out_dir = pathlib.Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Requesting dataset via kagglehub: %s", self.dataset_id)
        ds_path = kagglehub.dataset_download(self.dataset_id)
        logger.info("Dataset cached at: %s", ds_path)

        src_csv = pathlib.Path(ds_path) / self.csv_name
        if not src_csv.exists():
            files = os.listdir(ds_path)
            logger.error("CSV %s not found in %s; files: %s", self.csv_name, ds_path, files)
            raise FileNotFoundError(
                f"{self.csv_name} not found under {ds_path}. Found files: {os.listdir(ds_path)}"
            )
        dest_csv = out_dir / self.csv_name
        shutil.copy2(src_csv, dest_csv)
        logger.info("Copied CSV to %s", dest_csv)
        return str(dest_csv)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw", help="Directory to place the CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path = DataGatherer(output_dir=args.output)()
    logger.info("Local CSV ready at: %s", csv_path)
