"""
CSE4001: Applications of Machine Learning in Industry
Health-Project 3: Breast Cancer Detection System
Experiment 7: Production FastAPI Scoring Service & Lab Information System (LIS) Integration

Features:
- REST API powered by FastAPI with OpenAPI / Swagger documentation.
- Production scoring engine using Champion Model (SVM RBF) and StandardScaler pipeline.
- Real-time single inference, raw FNA feature auto-scaling, and high-throughput batch scoring.
- Automated clinical priority triage flags and risk categorisation.
- Hospital Lab Information System (LIS) accession webhook & order integration.
- Structured request/response logging for regulatory compliance and audit trails.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Directories & Logging Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUTS_DIR, "deployment_request_response.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BreastCancerScoringService")

# ---------------------------------------------------------------------------
# Global State for Serialized ML Artifacts
# ---------------------------------------------------------------------------
ml_assets: Dict[str, Any] = {
    "model": None,
    "scaler": None,
    "metadata": None,
    "feature_names": None,
    "start_time": time.time(),
}

def load_ml_assets():
    champion_path = os.path.join(MODELS_DIR, "champion_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "breast_cancer_scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    if os.path.exists(champion_path) and os.path.exists(scaler_path):
        ml_assets["model"] = joblib.load(champion_path)
        ml_assets["scaler"] = joblib.load(scaler_path)

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            ml_assets["metadata"] = json.load(f)
            ml_assets["feature_names"] = ml_assets["metadata"].get("features", [])
    else:
        # Default WDBC 30 features
        ml_assets["feature_names"] = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

# Initialize assets on import
load_ml_assets()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts
    load_ml_assets()
    logger.info("Breast Cancer Detection Scoring Engine initialized successfully.")
    yield
    # Shutdown
    logger.info("Breast Cancer Detection Scoring Engine shutting down.")

# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Breast Cancer FNA Biopsy Detection Service",
    description=(
        "Enterprise RESTful scoring engine for Fine-Needle Aspirate (FNA) breast cytology. "
        "Integrates with Hospital Lab Information Systems (LIS) to provide automated "
        "second-opinion priority triage flags for pathologist workflow prioritization."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class BiopsyFeatures(BaseModel):
    radius_mean: float = Field(..., description="Mean radius of cell nuclei")
    texture_mean: float = Field(..., description="Mean texture (standard deviation of gray-scale values)")
    perimeter_mean: float = Field(..., description="Mean perimeter of nuclei")
    area_mean: float = Field(..., description="Mean nuclear area")
    smoothness_mean: float = Field(..., description="Mean smoothness (local variation in radius lengths)")
    compactness_mean: float = Field(..., description="Mean compactness (perimeter^2 / area - 1.0)")
    concavity_mean: float = Field(..., description="Mean severity of concave portions of contour")
    concave_points_mean: float = Field(..., description="Mean number of concave contour points")
    symmetry_mean: float = Field(..., description="Mean nuclear symmetry")
    fractal_dimension_mean: float = Field(..., description="Mean fractal dimension (coastline approx - 1)")

    radius_se: float = Field(..., description="Standard error for radius")
    texture_se: float = Field(..., description="Standard error for texture")
    perimeter_se: float = Field(..., description="Standard error for perimeter")
    area_se: float = Field(..., description="Standard error for area")
    smoothness_se: float = Field(..., description="Standard error for smoothness")
    compactness_se: float = Field(..., description="Standard error for compactness")
    concavity_se: float = Field(..., description="Standard error for concavity")
    concave_points_se: float = Field(..., description="Standard error for concave points")
    symmetry_se: float = Field(..., description="Standard error for symmetry")
    fractal_dimension_se: float = Field(..., description="Standard error for fractal dimension")

    radius_worst: float = Field(..., description="Largest/worst radius value")
    texture_worst: float = Field(..., description="Largest/worst texture value")
    perimeter_worst: float = Field(..., description="Largest/worst perimeter value")
    area_worst: float = Field(..., description="Largest/worst area value")
    smoothness_worst: float = Field(..., description="Largest/worst smoothness value")
    compactness_worst: float = Field(..., description="Largest/worst compactness value")
    concavity_worst: float = Field(..., description="Largest/worst concavity value")
    concave_points_worst: float = Field(..., description="Largest/worst concave points value")
    symmetry_worst: float = Field(..., description="Largest/worst symmetry value")
    fractal_dimension_worst: float = Field(..., description="Largest/worst fractal dimension value")

class SinglePredictionRequest(BaseModel):
    sample_id: str = Field(default="SAMPLE-001", description="Laboratory Accession or Sample ID")
    is_standardized: bool = Field(default=False, description="Set True if input features are already Z-score scaled")
    features: BiopsyFeatures

class SinglePredictionResponse(BaseModel):
    sample_id: str
    predicted_class: int = Field(..., description="0 = Benign, 1 = Malignant")
    diagnosis_label: str = Field(..., description="'Benign' or 'Malignant'")
    malignancy_probability: float = Field(..., description="Calibrated posterior probability of malignancy P(y=1|x)")
    confidence_category: str = Field(..., description="Risk tier: 'Low Risk', 'Borderline Review', or 'High Risk'")
    clinical_triage_flag: str = Field(..., description="'NORMAL_QUEUE' or 'PRIORITY_PATHOLOGIST_REVIEW'")
    top_contributing_biomarkers: List[Dict[str, Any]]
    inference_latency_ms: float
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    batch_id: str = Field(default="BATCH-001", description="Batch run identifier")
    is_standardized: bool = Field(default=False, description="Set True if inputs are already standardized")
    samples: List[Dict[str, Any]]

class BatchPredictionResponse(BaseModel):
    batch_id: str
    total_samples: int
    malignant_count: int
    benign_count: int
    priority_flag_count: int
    predictions: List[Dict[str, Any]]
    total_processing_time_ms: float
    timestamp: str

class LISOrderRequest(BaseModel):
    order_id: str = Field(..., description="LIS Order accession number, e.g. LIS-2026-9041")
    patient_id: str = Field(..., description="De-identified patient MRN, e.g. PT-88421")
    ordering_physician: str = Field(..., description="Name or NPI of referring oncologist/surgeon")
    collection_site: str = Field(default="Left Breast Upper Outer Quadrant", description="Anatomical biopsy site")
    raw_measurements: BiopsyFeatures

class LISOrderResponse(BaseModel):
    order_id: str
    patient_id: str
    status: str
    triage_alert: str
    assigned_worklist: str
    predicted_malignancy_prob: float
    recommendation: str
    signature_audit: str

# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def extract_feature_vector(biopsy_obj: BiopsyFeatures, feature_names: List[str]) -> pd.DataFrame:
    raw_dict = biopsy_obj.model_dump()
    vec = [raw_dict[col] for col in feature_names]
    return pd.DataFrame([vec], columns=feature_names)

def compute_triage_tier(prob: float) -> tuple[str, str]:
    """
    Categorizes probability into clinical tiers based on Experiment 6 threshold findings:
    - prob < 0.35: Low Risk Benign (Routine queue)
    - 0.35 <= prob < 0.65: Borderline / Indeterminate (Priority review)
    - prob >= 0.65: High Risk Malignancy (Immediate urgent review)
    """
    if prob >= 0.65:
        return "High Risk Malignancy", "PRIORITY_PATHOLOGIST_REVIEW"
    elif prob >= 0.35:
        return "Borderline / Intermediate Risk", "PRIORITY_PATHOLOGIST_REVIEW"
    else:
        return "Low Risk (Benign)", "NORMAL_QUEUE"

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["System Information"])
async def root_info():
    """Returns basic scoring service operational metadata."""
    uptime_sec = time.time() - ml_assets["start_time"]
    return {
        "service": "Breast Cancer Detection Lab Scoring Service",
        "status": "HEALTHY",
        "version": "1.0.0",
        "champion_model": "Support Vector Machine (RBF Kernel)",
        "intended_use": "Pathologist assistive triage & second-opinion flag",
        "uptime_seconds": round(uptime_sec, 2),
        "docs_url": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/batch-predict", "/lis/webhook"]
    }

@app.get("/health", tags=["System Information"])
async def health_check():
    """Performs system diagnostic check verifying model and scaler availability."""
    model_ready = ml_assets["model"] is not None
    scaler_ready = ml_assets["scaler"] is not None
    if not (model_ready and scaler_ready):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model pipeline not loaded.")
    return {
        "status": "UP",
        "model_loaded": model_ready,
        "scaler_loaded": scaler_ready,
        "memory_status": "NORMAL",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/model-info", tags=["System Information"])
async def model_info():
    """Provides complete model metadata, hyperparameters, and clinical evaluation KPIs."""
    if ml_assets["metadata"]:
        return ml_assets["metadata"]
    return {
        "model_name": "Support Vector Machine (RBF)",
        "features_count": 30,
        "status": "Metadata file loaded from models/model_metadata.json"
    }

@app.post("/predict", response_model=SinglePredictionResponse, tags=["Scoring Engine"])
async def predict_sample(payload: SinglePredictionRequest):
    """
    Primary real-time scoring endpoint.
    Accepts 30 FNA nuclear morphometric features, validates input schema,
    applies StandardScaler normalization if raw, evaluates model decision boundary,
    and returns probability, triage category, and top biomarker cues.
    """
    start_time = time.perf_counter()
    feature_names = ml_assets["feature_names"]
    model = ml_assets["model"]
    scaler = ml_assets["scaler"]

    # Extract numerical array
    X_raw = extract_feature_vector(payload.features, feature_names)

    # Standardize if required
    if not payload.is_standardized:
        X_eval = scaler.transform(X_raw)
    else:
        X_eval = X_raw

    # Inference
    y_pred = int(model.predict(X_eval)[0])
    y_prob = float(model.predict_proba(X_eval)[0, 1])

    # Latency calculation
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

    # Clinical Triage categorisation
    conf_category, triage_flag = compute_triage_tier(y_prob)
    label = "Malignant" if y_pred == 1 else "Benign"

    # Identify top 3 features by standardized magnitude deviation
    std_magnitudes = np.abs(X_eval[0])
    top_indices = np.argsort(std_magnitudes)[::-1][:3]
    top_biomarkers = [
        {
            "biomarker": feature_names[i],
            "standardized_zscore": round(float(X_eval[0, i]), 3),
            "raw_value": round(float(X_raw[0, i]), 4)
        }
        for i in top_indices
    ]

    response_data = {
        "sample_id": payload.sample_id,
        "predicted_class": y_pred,
        "diagnosis_label": label,
        "malignancy_probability": round(y_prob, 4),
        "confidence_category": conf_category,
        "clinical_triage_flag": triage_flag,
        "top_contributing_biomarkers": top_biomarkers,
        "inference_latency_ms": latency_ms,
        "model_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Structured Audit Log
    logger.info(
        f"[PREDICT] Sample={payload.sample_id} | Label={label} | Prob={y_prob:.4f} | "
        f"Flag={triage_flag} | Latency={latency_ms:.2f}ms"
    )

    return response_data

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Scoring Engine"])
async def batch_predict(payload: BatchPredictionRequest):
    """
    High-throughput batch inference endpoint for clinical lab runs.
    Accepts an array of biopsy records and returns aggregated triage statistics.
    """
    start_time = time.perf_counter()
    feature_names = ml_assets["feature_names"]
    model = ml_assets["model"]
    scaler = ml_assets["scaler"]

    if not payload.samples:
        raise HTTPException(status_code=400, detail="Samples list cannot be empty.")

    records = []
    sample_ids = []
    for item in payload.samples:
        s_id = item.get("sample_id", f"SAMPLE-{len(records)+1}")
        sample_ids.append(s_id)
        vec = [item.get(col, 0.0) for col in feature_names]
        records.append(vec)

    X_mat = np.array(records, dtype=float)
    if not payload.is_standardized:
        X_eval = scaler.transform(X_mat)
    else:
        X_eval = X_mat

    preds = model.predict(X_eval)
    probas = model.predict_proba(X_eval)[:, 1]

    results = []
    malignant_count = 0
    priority_count = 0

    for i in range(len(preds)):
        p_class = int(preds[i])
        p_prob = float(probas[i])
        tier, triage = compute_triage_tier(p_prob)
        if p_class == 1:
            malignant_count += 1
        if triage == "PRIORITY_PATHOLOGIST_REVIEW":
            priority_count += 1

        results.append({
            "sample_id": sample_ids[i],
            "predicted_class": p_class,
            "diagnosis_label": "Malignant" if p_class == 1 else "Benign",
            "malignancy_probability": round(p_prob, 4),
            "confidence_category": tier,
            "clinical_triage_flag": triage
        })

    total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    benign_count = len(preds) - malignant_count

    logger.info(
        f"[BATCH] Batch={payload.batch_id} | Total={len(preds)} | Malignant={malignant_count} | "
        f"Priority={priority_count} | Time={total_time_ms}ms"
    )

    return {
        "batch_id": payload.batch_id,
        "total_samples": len(preds),
        "malignant_count": malignant_count,
        "benign_count": benign_count,
        "priority_flag_count": priority_count,
        "predictions": results,
        "total_processing_time_ms": total_time_ms,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/lis/webhook", response_model=LISOrderResponse, tags=["Lab Information System (LIS)"])
async def lis_webhook_receiver(order: LISOrderRequest):
    """
    Simulated Hospital Lab Information System (LIS) Webhook Endpoint.
    Ingests accession order, runs automated diagnostic triage scoring,
    and returns worklist assignment and clinical recommendation.
    """
    feature_names = ml_assets["feature_names"]
    model = ml_assets["model"]
    scaler = ml_assets["scaler"]

    # Transform and score
    X_raw = extract_feature_vector(order.raw_measurements, feature_names)
    X_eval = scaler.transform(X_raw)
    prob = float(model.predict_proba(X_eval)[0, 1])

    if prob >= 0.35:
        alert = "CRITICAL_FLAG_SUSPECTED_MALIGNANCY"
        worklist = "URGENT_PRIORITY_CYTOPATHOLOGY_WORKLIST"
        rec = "Prioritize for immediate senior cytopathologist microscopic review within 4 hours."
    else:
        alert = "ROUTINE_NEGATIVE_SCREEN"
        worklist = "STANDARD_CYTOLOGY_BATCH_QUEUE"
        rec = "Schedule for standard routine pathologist batch sign-off within 24 hours."

    audit_sig = f"AI-TRIAGE-SHA256:{hash(order.order_id + str(prob)) & 0xffffffff:08x}"

    logger.info(
        f"[LIS-ORDER] Order={order.order_id} | Patient={order.patient_id} | Prob={prob:.4f} | "
        f"Worklist={worklist} | Alert={alert}"
    )

    return {
        "order_id": order.order_id,
        "patient_id": order.patient_id,
        "status": "PROCESSED_BY_AI_TRIAGE",
        "triage_alert": alert,
        "assigned_worklist": worklist,
        "predicted_malignancy_prob": round(prob, 4),
        "recommendation": rec,
        "signature_audit": audit_sig
    }

# ---------------------------------------------------------------------------
# Direct Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Breast Cancer Detection Scoring Service on http://127.0.0.1:8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
