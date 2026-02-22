import os
import joblib
import pandas as pd
import logging
import pathlib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from contextlib import asynccontextmanager

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

MODEL_PATH = "/home/jefft/CST_605_Week6/var/data/diabetes_pipeline.joblib"
API_TOKEN_ENV = "MODEL_API_TOKEN"

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# --------- Data schema ----------
class PatientFeatures(BaseModel):
    gender: str = Field(..., description="e.g., 'Male', 'Female', 'Other'")
    age: float
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    smoking_history: str = Field(..., description="e.g., 'never', 'former', 'current', etc.")
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    
    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v: str):
        return v.strip().title()

    @field_validator("smoking_history")
    @classmethod
    def normalize_smoking(cls, v: str):
        return v.strip().lower()
# ---------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML pipeline once, before serving requests, and clean up at shutdown.
    This replaces deprecated @app.on_event('startup'/'shutdown').
    """
    logger.info("Starting application... loading model from %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model not found at %s. Please train a model for use.", MODEL_PATH)
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Train and save the model first."
        )
    pipeline = joblib.load(MODEL_PATH)
    app.state.pipeline = pipeline
    logger.info("Model loaded.")
    try:
        yield
    finally:
        app.state.pipeline = None
        logger.info("Application shutdown complete.")
# -------------------------------------

app = FastAPI(
    title="Diabetes Prediction API",
    version="1.1.0",
    lifespan=lifespan
)


def verify_token(token: Optional[str]):
    expected = os.getenv(API_TOKEN_ENV)
    if not expected:
        logger.debug("MODEL_API_TOKEN not set; allowing request (dev).")
        return True
    if token != expected:
        logger.warning("Unauthorized request: bad token.")
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    return True

@app.get("/healthz")
def healthz():
    loaded = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    logger.debug("Health check: model_loaded=%s", loaded)
    return {"status": "ok"}

@app.post("/predict")
def predict(
    features: PatientFeatures,
    x_api_token: Optional[str] = Header(default=None, convert_underscores=False, alias="X-API-Token")
):
    verify_token(x_api_token)
    logger.info("Received prediction request")
    df = pd.DataFrame([features.model_dump()])
    pipe = app.state.pipeline
    pred = int(pipe.predict(df)[0])
    result = {"prediction": pred}
    logger.debug("Predicted class: %s", pred)
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(df)[0, 1])
        result["probability_diabetes_1"] = proba
        logger.debug("Predicted probability (class 1): %.4f", proba)
    return result