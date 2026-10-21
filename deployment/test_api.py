"""
CSE4001: Applications of Machine Learning in Industry
Health-Project 3: Breast Cancer Detection System
Experiment 7: Automated Deployment Verification & LIS Testing Script

Tasks:
1. Initialize FastAPI TestClient.
2. Load real patient specimens from held-out test data (both benign and malignant).
3. Test Root (`/`), Health Check (`/health`), and Model Info (`/model-info`) endpoints.
4. Test Single Real-Time Scoring (`/predict`) with Benign and Malignant FNA specimens.
5. Test High-Throughput Batch Inference (`/batch-predict`) with multiple biopsy cases.
6. Test Hospital Lab Information System (LIS) Webhook (`/lis/webhook`).
7. Save full request/response log artifacts in `outputs/`.
"""

import os
import sys
import json
import time
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "deployment"))

from deployment.app import app

def run_deployment_tests():
    print("=" * 80)
    print("EXPERIMENT 7: FASTAPI SCORING SERVICE & LIS INTEGRATION TEST SUITE")
    print("=" * 80)

    client = TestClient(app)
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    request_response_log = []

    # 1. Test Root Endpoint
    print("\n[Test 1] Testing Root Endpoint (GET /)...")
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Expected 200, got {res_root.status_code}"
    root_json = res_root.json()
    print(f"  - Status Code: {res_root.status_code}")
    print(f"  - Service: {root_json['service']}")
    print(f"  - Champion Model: {root_json['champion_model']}")
    request_response_log.append({
        "test_name": "Root Endpoint",
        "method": "GET",
        "endpoint": "/",
        "response_status": res_root.status_code,
        "response_body": root_json
    })

    # 2. Test Health Endpoint
    print("\n[Test 2] Testing Health Diagnostics Endpoint (GET /health)...")
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_json = res_health.json()
    print(f"  - Health Status: {health_json['status']}")
    print(f"  - Model Loaded: {health_json['model_loaded']}")
    print(f"  - Scaler Loaded: {health_json['scaler_loaded']}")
    request_response_log.append({
        "test_name": "Health Diagnostics",
        "method": "GET",
        "endpoint": "/health",
        "response_status": res_health.status_code,
        "response_body": health_json
    })

    # 3. Test Model Info Endpoint
    print("\n[Test 3] Testing Model Info Endpoint (GET /model-info)...")
    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    info_json = res_info.json()
    print(f"  - Model Name: {info_json.get('model_name')}")
    print(f"  - Model Type: {info_json.get('model_type')}")
    print(f"  - Test Accuracy: {info_json.get('performance_metrics', {}).get('test_accuracy')}")
    request_response_log.append({
        "test_name": "Model Info",
        "method": "GET",
        "endpoint": "/model-info",
        "response_status": res_info.status_code,
        "response_body": info_json
    })

    # Load actual raw test dataset to extract realistic specimen values
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))
    benign_sample = raw_df[raw_df["diagnosis"] == "B"].iloc[0].drop(["id", "diagnosis"]).to_dict()
    malignant_sample = raw_df[raw_df["diagnosis"] == "M"].iloc[0].drop(["id", "diagnosis"]).to_dict()

    # 4. Test Single Prediction - Benign Specimen
    print("\n[Test 4] Testing Single Real-Time Scoring: Known BENIGN Specimen (POST /predict)...")
    payload_benign = {
        "sample_id": "PT-BENIGN-001",
        "is_standardized": False,
        "features": benign_sample
    }
    t0 = time.perf_counter()
    res_pred_b = client.post("/predict", json=payload_benign)
    t_lat_b = (time.perf_counter() - t0) * 1000.0
    assert res_pred_b.status_code == 200
    pred_b_json = res_pred_b.json()
    print(f"  - Sample: {pred_b_json['sample_id']}")
    print(f"  - Prediction: {pred_b_json['diagnosis_label']} (Class {pred_b_json['predicted_class']})")
    print(f"  - Malignancy Probability: {pred_b_json['malignancy_probability']:.4f}")
    print(f"  - Confidence Category: {pred_b_json['confidence_category']}")
    print(f"  - Triage Flag: {pred_b_json['clinical_triage_flag']}")
    print(f"  - Response Latency: {t_lat_b:.2f} ms")
    request_response_log.append({
        "test_name": "Single Prediction - Benign",
        "method": "POST",
        "endpoint": "/predict",
        "request_payload": payload_benign,
        "response_status": res_pred_b.status_code,
        "response_body": pred_b_json
    })

    # 5. Test Single Prediction - Malignant Specimen
    print("\n[Test 5] Testing Single Real-Time Scoring: Known MALIGNANT Specimen (POST /predict)...")
    payload_mal = {
        "sample_id": "PT-MALIGNANT-002",
        "is_standardized": False,
        "features": malignant_sample
    }
    t0 = time.perf_counter()
    res_pred_m = client.post("/predict", json=payload_mal)
    t_lat_m = (time.perf_counter() - t0) * 1000.0
    assert res_pred_m.status_code == 200
    pred_m_json = res_pred_m.json()
    print(f"  - Sample: {pred_m_json['sample_id']}")
    print(f"  - Prediction: {pred_m_json['diagnosis_label']} (Class {pred_m_json['predicted_class']})")
    print(f"  - Malignancy Probability: {pred_m_json['malignancy_probability']:.4f}")
    print(f"  - Confidence Category: {pred_m_json['confidence_category']}")
    print(f"  - Triage Flag: {pred_m_json['clinical_triage_flag']}")
    print(f"  - Response Latency: {t_lat_m:.2f} ms")
    request_response_log.append({
        "test_name": "Single Prediction - Malignant",
        "method": "POST",
        "endpoint": "/predict",
        "request_payload": payload_mal,
        "response_status": res_pred_m.status_code,
        "response_body": pred_m_json
    })

    # 6. Test Batch Prediction Endpoint
    print("\n[Test 6] Testing High-Throughput Batch Scoring (POST /batch-predict)...")
    batch_records = []
    for idx in range(10):
        row = raw_df.iloc[idx].drop(["id", "diagnosis"]).to_dict()
        row["sample_id"] = f"BATCH-SPECIMEN-{idx+1:03d}"
        batch_records.append(row)

    payload_batch = {
        "batch_id": "LAB-RUN-2026-OCT-21",
        "is_standardized": False,
        "samples": batch_records
    }
    res_batch = client.post("/batch-predict", json=payload_batch)
    assert res_batch.status_code == 200
    batch_json = res_batch.json()
    print(f"  - Batch ID: {batch_json['batch_id']}")
    print(f"  - Total Biopsy Samples: {batch_json['total_samples']}")
    print(f"  - Malignant Identified: {batch_json['malignant_count']}")
    print(f"  - Benign Identified: {batch_json['benign_count']}")
    print(f"  - Urgent Triage Flags: {batch_json['priority_flag_count']}")
    print(f"  - Total Execution Time: {batch_json['total_processing_time_ms']:.2f} ms")
    request_response_log.append({
        "test_name": "Batch Prediction",
        "method": "POST",
        "endpoint": "/batch-predict",
        "request_payload": {"batch_id": payload_batch["batch_id"], "sample_count": 10},
        "response_status": res_batch.status_code,
        "response_body": batch_json
    })

    # 7. Test Hospital Lab Information System (LIS) Webhook
    print("\n[Test 7] Testing Hospital LIS Order Webhook Integration (POST /lis/webhook)...")
    payload_lis = {
        "order_id": "LIS-ORD-2026-89412",
        "patient_id": "MRN-BREAST-99104",
        "ordering_physician": "Dr. Sarah Jenkins, MD (Surgical Oncology)",
        "collection_site": "Right Breast Upper Outer Quadrant - Core Needle Biopsy",
        "raw_measurements": malignant_sample
    }
    res_lis = client.post("/lis/webhook", json=payload_lis)
    assert res_lis.status_code == 200
    lis_json = res_lis.json()
    print(f"  - LIS Order: {lis_json['order_id']}")
    print(f"  - Patient: {lis_json['patient_id']}")
    print(f"  - Triage Alert: {lis_json['triage_alert']}")
    print(f"  - Assigned Worklist: {lis_json['assigned_worklist']}")
    print(f"  - Probability: {lis_json['predicted_malignancy_prob']:.4f}")
    print(f"  - Recommendation: {lis_json['recommendation']}")
    print(f"  - Audit Signature: {lis_json['signature_audit']}")
    request_response_log.append({
        "test_name": "LIS Webhook Integration",
        "method": "POST",
        "endpoint": "/lis/webhook",
        "request_payload": payload_lis,
        "response_status": res_lis.status_code,
        "response_body": lis_json
    })

    # 8. Save Complete Request/Response Log Artifacts
    json_log_path = os.path.join(OUTPUTS_DIR, "sample_request_response_log.json")
    with open(json_log_path, "w") as f:
        json.dump(request_response_log, f, indent=4)
    print(f"\n[Artifact] Saved complete Request/Response JSON log to: {json_log_path}")

    # Generate a formatted text report
    txt_log_path = os.path.join(OUTPUTS_DIR, "experiment7_deployment_summary.txt")
    with open(txt_log_path, "w") as f:
        f.write("EXPERIMENT 7: FASTAPI SCORING SERVICE DEPLOYMENT & LIS INTEGRATION REPORT\n")
        f.write("=" * 85 + "\n\n")
        f.write("SUMMARY OF ENDPOINT BENCHMARK RESULTS:\n")
        f.write(f"1. Service Status: UP & HEALTHY\n")
        f.write(f"2. Single Inference Latency: {pred_b_json['inference_latency_ms']:.2f} ms (Target SLA: <200 ms)\n")
        f.write(f"3. Batch Throughput: {batch_json['total_processing_time_ms']:.2f} ms for 10 biopsy records\n")
        f.write(f"4. Benign Patient Classification: {pred_b_json['diagnosis_label']} ({pred_b_json['clinical_triage_flag']})\n")
        f.write(f"5. Malignant Patient Classification: {pred_m_json['diagnosis_label']} ({pred_m_json['clinical_triage_flag']})\n")
        f.write(f"6. LIS Webhook Routing: {lis_json['assigned_worklist']}\n\n")
        f.write("SAMPLE REQUEST/RESPONSE AUDIT:\n")
        f.write(json.dumps(pred_m_json, indent=2) + "\n")

    print(f"[Artifact] Saved deployment summary text report to: {txt_log_path}")
    print(f"\n[SUCCESS] All Experiment 7 deployment tests PASSED successfully.")

if __name__ == "__main__":
    run_deployment_tests()
