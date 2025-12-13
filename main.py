# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from typing import Any

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# set up tracer (you will configure exporter/OTel collector in prod)
trace.set_tracer_provider(SDKTracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

MODEL_PATH = os.getenv("MODEL_PATH", "trained_pipeline.joblib")

app = FastAPI(title="heart-disease-predn")

# input schema - adjust fields to your dataset feature names
class Transaction(BaseModel):
    sno: int
    age: float
    gender: str
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# load model once
try:
    model = joblib.load(MODEL_PATH)
    # optionally print classes for debugging
    try:
        print("Model classes:", getattr(model, "classes_", None))
    except Exception:
        pass
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

def _label_to_binary(label: Any) -> int:
    """Map common label types to 0/1. Specifically handles 'yes'/'no'."""
    if label is None:
        return 0
    # numeric label
    if isinstance(label, (int, np.integer)):
        return int(label)
    s = str(label).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "fraud"):
        return 1
    if s in ("0", "false", "f", "no", "n", "legit", "nonfraud"):
        return 0
    # fallback: try int conversion
    try:
        return int(float(s))
    except Exception:
        return 0

def _positive_class_index(proba_array, model_obj):
    """
    Determine which column index in predict_proba corresponds to the positive class ('yes' -> 1).
    If model.classes_ exists and contains a positive label, use that. Otherwise fall back to last column.
    """
    classes = getattr(model_obj, "classes_", None)
    if classes is not None:
        # try to find a class that maps to positive
        for i, c in enumerate(classes):
            if _label_to_binary(c) == 1:
                return i
    # default: use last column
    return proba_array.shape[1] - 1

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(tx: Transaction):
    # convert to DataFrame so pipeline works with preprocessing
    df = pd.DataFrame([tx.dict()])

    # use an explicit span to measure model.predict()
    with tracer.start_as_current_span("inference_total"):
        with tracer.start_as_current_span("model_predict"):
            try:
                # get raw prediction (could be 'yes'/'no' strings)
                pred = model.predict(df)
                pred_label = pred[0]

                # map label to binary 0/1
                pred_int = _label_to_binary(pred_label)

                # determine probability for the positive class (if available)
                proba = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df)  # shape (n_samples, n_classes)
                    pos_idx = _positive_class_index(probs, model)
                    proba = float(probs[0, pos_idx])

            except Exception as e:
                # in prod, avoid returning raw exceptions — log instead
                raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": int(pred_int), "probability": proba}
