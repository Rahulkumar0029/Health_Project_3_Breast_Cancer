# Experiment 7: Deployment — FastAPI Scoring Service & Lab Information System (LIS) Integration

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Selected Champion Model:** Support Vector Machine (RBF Kernel, $C=5.0, \gamma=0.01$)  
**Status:** Completed  

---

## 1. Aim

To deploy the calibrated production champion model as an enterprise-grade, asynchronous RESTful inference microservice using **FastAPI** and **Uvicorn**, integrate the scoring pipeline with hospital **Lab Information Systems (LIS)**, establish automated clinical triage workflows with priority alerts, benchmark operational latency against strict Service Level Agreements (SLA $<200$ ms), and document the complete OpenAPI/Swagger schema and audit logging system.

---

## 2. Objectives

1. **Production Service Packaging:** Wrap the serialized champion classifier (`models/champion_model.pkl`) and preprocessing pipeline (`models/breast_cancer_scaler.pkl`) inside a modular, asynchronous FastAPI application.
2. **Strict Schema Validation & Type Safety:** Implement robust Pydantic v2 data models for all 30 digitized Fine-Needle Aspirate (FNA) nuclear morphometric features with boundary constraints.
3. **Multi-Modal Scoring Capabilities:**
   - Real-time single-sample inference endpoint (`POST /predict`) with raw feature auto-standardization.
   - High-throughput batch scoring endpoint (`POST /batch-predict`) for hospital laboratory accession runs.
4. **Clinical Triage Routing & Priority Categorization:** Implement rule-based clinical triage flag generation:
   - $\hat{p} < 0.35 \implies$ `NORMAL_QUEUE` (Routine negative screen)
   - $0.35 \le \hat{p} < 0.65 \implies$ `PRIORITY_PATHOLOGIST_REVIEW` (Borderline / indeterminate atypia)
   - $\hat{p} \ge 0.65 \implies$ `PRIORITY_PATHOLOGIST_REVIEW` (High-risk malignancy)
5. **Hospital Lab Information System (LIS) Integration:** Construct a dedicated webhook endpoint (`POST /lis/webhook`) adhering to standard accession order formats, automatically assigning cases to cytopathology worklists with SHA-256 cryptographic audit signatures.
6. **SLA Latency Benchmarking:** Execute automated multi-request stress testing ($N=100$) to evaluate P50, P95, and P99 latency percentiles against clinical operational requirements ($<200$ ms).
7. **Audit Logging & Regulatory Compliance:** Enforce persistent, structured JSON and text transaction logging in `outputs/deployment_request_response.log` for medical auditability and patient de-identification compliance.

---

## 3. System Architecture & Clinical Deployment Topology

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │             Hospital Lab Information System (LIS) / EMR                │
   │      (Epic Beaker / Cerner Millennium / Sunquest Laboratory)          │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                      HTTP / REST (JSON / HL7 FHIR Bridge)
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                   FastAPI Scoring Microservice Engine                  │
   │                                                                        │
   │  ┌───────────────────────┐                  ┌───────────────────────┐  │
   │  │   Input Validation    │                  │  Automated Standard   │  │
   │  │   (Pydantic Schema)   │ ───────────────> │  Scaling Pipeline     │  │
   │  └───────────────────────┘                  └───────────┬───────────┘  │
   │                                                         │              │
   │                                                         ▼              │
   │  ┌───────────────────────┐                  ┌───────────────────────┐  │
   │  │ Clinical Triage Rules │ <─────────────── │ Champion SVM (RBF)    │  │
   │  │ (Priority Flag Engine)│                  │ Calibrated Inference  │  │
   │  └───────────┬───────────┘                  └───────────────────────┘  │
   │              │                                                         │
   └──────────────┼─────────────────────────────────────────────────────────┘
                  │
                  ├───────────────────────────────┐
                  ▼                               ▼
   ┌──────────────────────────────┐┌──────────────────────────────┐
   │ Pathologist Worklist Router  ││  Audit & Compliance Logger   │
   │ • Normal Routine Queue       ││  • outputs/deployment_...log │
   │ • Urgent Priority Review     ││  • Cryptographic SHA-256 Sig │
   └──────────────────────────────┘└──────────────────────────────┘
```

---

## 4. RESTful API Endpoint Specifications

The service exposes five operational endpoints documented via interactive OpenAPI (Swagger UI) at `/docs` and ReDoc at `/redoc`:

| HTTP Method | Route | Description | Auth / Role | Input Payload | Output Response |
|---|---|---|---|---|---|
| `GET` | `/` | Root service metadata & operational status | Public | None | System info, version, active champion model |
| `GET` | `/health` | Liveness and readiness diagnostic probe | Internal / K8s | None | UP/DOWN status, model and scaler memory readiness |
| `GET` | `/model-info` | Production model metadata & provenance | Clinicians / Admins | None | Serialized training parameters, features, test KPIs |
| `POST` | `/predict` | Real-time single biopsy specimen scoring | Lab Technicians | 30 raw FNA biometric features | Probability, class, triage alert, top biomarkers |
| `POST` | `/batch-predict` | High-throughput batch inference | Lab Batch Run | Array of biopsy specimen objects | Aggregated counts, case predictions, total time |
| `POST` | `/lis/webhook` | Hospital LIS accession order receiver | LIS Middleware | Accession ID, MRN, raw FNA features | Worklist routing, clinical recommendation, audit signature |

---

## 5. Request and Response Schemas

### 5.1 Real-Time Single Biopsy Scoring (`POST /predict`)

#### Sample Request Payload:
```json
{
  "sample_id": "PT-MALIGNANT-002",
  "is_standardized": false,
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

#### Sample Response Payload:
```json
{
  "sample_id": "PT-MALIGNANT-002",
  "predicted_class": 1,
  "diagnosis_label": "Malignant",
  "malignancy_probability": 1.0000,
  "confidence_category": "High Risk Malignancy",
  "clinical_triage_flag": "PRIORITY_PATHOLOGIST_REVIEW",
  "top_contributing_biomarkers": [
    {
      "biomarker": "area_worst",
      "standardized_zscore": 2.518,
      "raw_value": 2019.0
    },
    {
      "biomarker": "perimeter_worst",
      "standardized_zscore": 2.378,
      "raw_value": 184.6
    },
    {
      "biomarker": "concave_points_worst",
      "standardized_zscore": 2.342,
      "raw_value": 0.2654
    }
  ],
  "inference_latency_ms": 5.04,
  "model_version": "1.0.0",
  "timestamp": "2026-10-21T04:28:40.129381Z"
}
```

---

## 6. Hospital Lab Information System (LIS) Webhook Integration

In clinical operations, the AI system does not function in isolation; it interfaces directly with hospital LIS order accession queues.

### 6.1 LIS Order Message Exchange (`POST /lis/webhook`)

#### Input Accession Order Payload:
```json
{
  "order_id": "LIS-ORD-2026-89412",
  "patient_id": "MRN-BREAST-99104",
  "ordering_physician": "Dr. Sarah Jenkins, MD (Surgical Oncology)",
  "collection_site": "Right Breast Upper Outer Quadrant - Core Needle Biopsy",
  "raw_measurements": { ... }
}
```

#### Output Triage Routing Response:
```json
{
  "order_id": "LIS-ORD-2026-89412",
  "patient_id": "MRN-BREAST-99104",
  "status": "PROCESSED_BY_AI_TRIAGE",
  "triage_alert": "CRITICAL_FLAG_SUSPECTED_MALIGNANCY",
  "assigned_worklist": "URGENT_PRIORITY_CYTOPATHOLOGY_WORKLIST",
  "predicted_malignancy_prob": 1.0000,
  "recommendation": "Prioritize for immediate senior cytopathologist microscopic review within 4 hours.",
  "signature_audit": "AI-TRIAGE-SHA256:298a847b"
}
```

### 6.2 Pathologist Worklist Queue Allocation Logic:
- **Routine Negative Screen Queue:** Benign specimens ($\hat{p} < 0.35$) are routed to the standard batch review queue with a standard 24-hour turnaround time.
- **Urgent Priority Cytopathology Worklist:** High-risk ($\hat{p} \ge 0.65$) and Borderline ($\hat{p} \in [0.35, 0.65)$) cases trigger an automated pop-up notification on the attending pathologist's digital pathology workstation, enforcing a mandated 4-hour review SLA.

---

## 7. Performance Benchmarking & SLA Compliance

To validate production readiness, the microservice was subjected to automated latency benchmarking across $N=100$ consecutive HTTP requests:

| Benchmark Metric | Measured Performance | Operational SLA Requirement | SLA Status |
|---|---|---|---|
| **P50 Median Latency** | **6.32 ms** | $< 100$ ms | **PASS (15x faster)** |
| **P95 Latency** | **9.75 ms** | $< 150$ ms | **PASS (15x faster)** |
| **P99 Worst-Case Latency** | **10.50 ms** | $< 200$ ms | **PASS (19x faster)** |
| **Mean Latency** | **6.58 ms** | $< 100$ ms | **PASS** |
| **Batch Throughput (N=10)** | **0.72 ms total** | $< 500$ ms | **PASS (690x faster)** |
| **Service Availability** | **100.00%** | $\ge 99.9\%$ | **PASS** |

The measured sub-10 ms latency confirms that the model adds negligible overhead to clinical laboratory workflows, enabling seamless integration into real-time digital slide scanners.

---

## 8. Audit Logging & Regulatory Compliance

Every transaction processed by the scoring engine is persisted with a strict audit trail in `outputs/deployment_request_response.log`:

```
2026-10-21 09:59:10 [INFO] Breast Cancer Detection Scoring Engine initialized successfully.
2026-10-21 09:59:11 [INFO] [PREDICT] Sample=PT-BENIGN-001 | Label=Benign | Prob=0.0254 | Flag=NORMAL_QUEUE | Latency=41.14ms
2026-10-21 09:59:11 [INFO] [PREDICT] Sample=PT-MALIGNANT-002 | Label=Malignant | Prob=1.0000 | Flag=PRIORITY_PATHOLOGIST_REVIEW | Latency=5.04ms
2026-10-21 09:59:11 [INFO] [BATCH] Batch=LAB-RUN-2026-OCT-21 | Total=10 | Malignant=10 | Priority=10 | Time=0.72ms
2026-10-21 09:59:11 [INFO] [LIS-ORDER] Order=LIS-ORD-2026-89412 | Patient=MRN-BREAST-99104 | Prob=1.0000 | Worklist=URGENT_PRIORITY_CYTOPATHOLOGY_WORKLIST | Alert=CRITICAL_FLAG_SUSPECTED_MALIGNANCY
```

### Regulatory Compliance Highlights:
- **HIPAA Privacy Rule:** No Protected Health Information (PHI) such as patient names, phone numbers, or dates of birth are processed by the scoring engine. De-identified Medical Record Numbers (MRN) and Accession IDs are used exclusively.
- **FDA Software as a Medical Device (SaMD):** Classified as an **Investigational Clinical Decision Support Tool**. The system outputs a recommendation and triage priority flag; final diagnostic sign-off remains under the licensed cytopathologist's clinical discretion.
- **Cryptographic Audit Hash:** Each prediction response includes a verifiable SHA-256 digital signature to guarantee tamper-proof auditability in clinical records.

---

## 9. Result

The Breast Cancer Detection Scoring Service was successfully implemented in FastAPI, validated with comprehensive automated test suites, and integrated with simulated hospital Lab Information Systems. The service demonstrated flawless functional accuracy, robust error handling, and exceptional latency benchmarks (P95 of 9.75 ms vs 200 ms SLA). All request/response transactions and triage worklist routings were logged to `outputs/sample_request_response_log.json` and `outputs/deployment_request_response.log`.

---

## 10. Conclusion

Experiment 7 bridges the gap between machine learning development and real-world clinical pathology operations. By deploying the calibrated SVM RBF model as an enterprise microservice with automated hospital LIS triage routing, digitized FNA biopsy samples can now be screened and prioritized in milliseconds. This infrastructure forms the backbone for continuous performance monitoring and drift surveillance addressed in Experiment 8.
