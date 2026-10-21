# Experiment 8: Continuous Update, Model Monitoring & Lifecycle Maintenance

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Deployed Champion:** Support Vector Machine (RBF Kernel)  
**Status:** Completed  

---

## 1. Aim

To design, implement, and validate a continuous post-deployment monitoring and maintenance framework for the Breast Cancer Detection scoring service, incorporating statistical **Data Drift** detection (Kolmogorov-Smirnov test and Population Stability Index), **Concept Drift** tracking against pathologist-confirmed ground truth, automated retraining triggers, champion-challenger shadow evaluation, instant rollback protocols, and clinical governance SLA documentation.

---

## 2. Objectives

1. **Establish Statistical Drift Detection Surveillance:** Implement quantitative drift surveillance across all 30 digitized Fine-Needle Aspirate (FNA) nuclear features using the non-parametric **Kolmogorov-Smirnov (KS) Two-Sample Test** and the **Population Stability Index (PSI)**.
2. **Longitudinal Concept & Performance Monitoring:** Design an asynchronous feedback loop to log pathologist microscopic confirmations and compute rolling Malignant Recall (Sensitivity) and Brier calibration scores.
3. **Formalize Automated Retraining Triggers:** Establish explicit, multi-tiered retraining threshold triggers:
   - *Data Drift Trigger:* $\text{PSI} \ge 0.25$ or KS test $p$-value $< 0.01$ across top diagnostic biomarkers (`concave_points_worst`, `radius_worst`, `area_worst`).
   - *Performance Trigger:* Rolling Malignant Recall drops below the mandated clinical threshold of $95.0\%$ over $N=50$ confirmed biopsy cases.
   - *Volume Trigger:* Ingestion of $N=250$ newly validated pathologist biopsy samples.
   - *Calendar Cadence:* Mandatory quarterly re-validation audit.
4. **Champion-Challenger Shadow Evaluation:** Construct a zero-downtime evaluation protocol comparing the incumbent production model (Champion) against newly retrained architectures (Challengers) under live shadow traffic.
5. **Instant Fallback & Rollback Governance:** Design automated and manual rollback mechanisms to restore the previously validated model checkpoint (`models/champion_model.pkl`) upon any detectable regression.
6. **Clinical Governance & SLA Ownership:** Formulate a RACI matrix defining roles between Clinical Cytopathologists, Laboratory Directors, and MLOps Engineers, backed by operational Service Level Agreements (SLAs).

---

## 3. Mathematical Foundations of Drift Detection

```
                       ┌──────────────────────────────────────────────┐
                       │   Live Laboratory Biopsy Stream (LIS Orders) │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │   Data Drift Engine     │                     │ Pathologist Confirmation│
         │  (Unsupervised Inputs)  │                     │     (Ground Truth)      │
         └────────────┬────────────┘                     └────────────┬────────────┘
                      │                                               │
             KS-Test & PSI Audit                             Rolling Recall & Brier
                      │                                               │
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │ Data Drift Alert (PSI)  │                     │ Performance Alert (<95%)│
         └────────────┬────────────┘                     └────────────┬────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Retraining Trigger Engine    │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Champion-Challenger Validation  │
                             │ (Shadow Deployment / Rollback)  │
                             └─────────────────────────────────┘
```

### 3.1 Kolmogorov-Smirnov (KS) Two-Sample Test

The two-sample KS test evaluates whether the incoming clinical feature distribution $F_{\text{live}}(x)$ diverges significantly from the reference training distribution $F_{\text{ref}}(x)$:

$$D_{n, m} = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{live}}(x) \right|$$

- **Null Hypothesis ($H_0$):** Both patient cohorts originate from identical continuous distributions ($F_{\text{ref}} = F_{\text{live}}$).
- **Rejection Criterion:** If asymptotic $p$-value $< \alpha = 0.05$, reject $H_0$; a statistically significant distributional shift has occurred.

### 3.2 Population Stability Index (PSI)

The Population Stability Index quantifies the total divergence between baseline reference frequencies ($P_b$) and current cohort frequencies ($Q_b$) across $B$ quantile buckets:

$$\text{PSI} = \sum_{b=1}^{B} \left( Q_b - P_b \right) \times \ln\left( \frac{Q_b}{P_b} \right)$$

#### Clinical Drift Threshold Guidelines:
- $\text{PSI} < 0.10$: **Stable / Negligible Drift** $\implies$ Normal operations; no action required.
- $0.10 \le \text{PSI} < 0.25$: **Moderate Drift / Warning** $\implies$ Log alert; increase monitoring frequency to weekly.
- $\text{PSI} \ge 0.25$: **Significant Shift / Action Required** $\implies$ Trigger automated model retraining and audit laboratory staining/scanner calibration.

---

## 4. Empirical Drift Simulation & Audit Results

The monitoring pipeline simulated two real-world clinical data streams:
1. **Batch 1 (Stable Cohort, $N=57$):** Routine FNA biopsies with standard biological variability.
2. **Batch 2 (Scanner/Staining Shift Cohort, $N=57$):** Laboratory scanner optical recalibration and hematoxylin staining variation shifting nuclear texture, area, and concavity.

### 4.1 Statistical Drift Audit Metrics Table

| Morphometric Biomarker | Batch 1 KS Stat | Batch 1 $p$-val | Batch 1 PSI | Batch 1 Status | Batch 2 KS Stat | Batch 2 $p$-val | Batch 2 PSI | Batch 2 Status |
|---|---|---|---|---|---|---|---|---|
| `concave_points_worst` | 0.0817 | 0.8576 | 0.1311 | Warning | 0.2487 | 0.0031 | **0.3199** | **DRIFT ALERT** |
| `radius_worst` | 0.0731 | 0.9294 | 0.0732 | **STABLE** | 0.3082 | 0.0001 | **1.6560** | **DRIFT ALERT** |
| `perimeter_worst` | 0.0928 | 0.7372 | 0.1060 | Warning | 0.3279 | 0.0000 | **1.5212** | **DRIFT ALERT** |
| `area_worst` | 0.0689 | 0.9545 | 0.0877 | **STABLE** | 0.4025 | 0.0000 | **2.7757** | **DRIFT ALERT** |
| `texture_worst` | 0.0600 | 0.9873 | 0.0973 | **STABLE** | 0.2295 | 0.0078 | **0.4608** | **DRIFT ALERT** |
| `concavity_mean` | 0.0912 | 0.7556 | 0.1165 | Warning | 0.0908 | 0.7605 | 0.1924 | Warning |

### 4.2 Key Findings:
- **Batch 1:** Maintained high stability across core morphometric features (`radius_worst`, `area_worst`, `texture_worst` all $\text{PSI} < 0.10$), yielding **100.00% Malignant Recall**.
- **Batch 2:** Exhibited severe optical distribution shift on nuclear area ($\text{PSI} = 2.7757, p < 0.0001$) and radius ($\text{PSI} = 1.6560$), successfully triggering the automated **Retraining Mandate**.

---

## 5. Continuous Retraining Triggers

Retraining is strictly governed by four formalized triggers to prevent unnecessary pipeline churn while guaranteeing diagnostic safety:

| Trigger Type | Operational Metric / Condition | Action Required | Responsible Party |
|---|---|---|---|
| **Data Drift Trigger** | $\text{PSI} \ge 0.25$ or KS test $p < 0.01$ on $\ge 2$ top biomarkers | Automated retrain workflow triggered; alert MLOps team | Lead MLOps Engineer |
| **Performance Trigger** | Rolling Malignant Recall $< 95.0\%$ over 50 confirmed cases | **CRITICAL ALERT:** Immediate clinical review + fallback | Head of Cytopathology |
| **Data Volume Trigger** | Ingestion of 250 newly confirmed pathologist cases | Incremental model fine-tuning with updated standard scaler | MLOps Pipeline |
| **Calendar Trigger** | 90 days elapsed since last model validation (Quarterly) | Full re-benchmarking against cumulative clinical archive | Quality Assurance Lead |

---

## 6. Champion-Challenger Governance & Automated Rollback Plan

### 6.1 Champion-Challenger Shadow Validation Protocol:
1. **Candidate Training:** When a retraining trigger fires, candidate models (Random Forest, SVM RBF, Gradient Boosting) are automatically trained on the expanded historical corpus.
2. **Offline Benchmark Evaluation:** The Challenger model must satisfy the following gating criteria on held-out validation:
   - Malignant Recall $\ge 98.0\%$.
   - Specificity $\ge 95.0\%$.
   - Brier Score $\le 0.010$.
   - Inference Latency $< 20$ ms.
3. **Canary Shadow Routing:** The Challenger runs in shadow mode (evaluating 100% of live traffic without displaying predictions to pathologists) for 14 days. If agreement with pathologist consensus exceeds 98%, the Challenger is promoted to Champion via blue/green DNS switch.

### 6.2 Zero-Downtime Rollback Plan:
- If at any point the active production model drops below $95.0\%$ recall, encounters runtime exceptions, or demonstrates latency degradation ($>200$ ms P95):
  1. The API gateway automatically redirects traffic to the previous immutable model snapshot (`models/champion_model.pkl`).
  2. The scoring service transitions to **Assisted Fallback Mode** (highlighting raw metric percentiles to pathologists rather than issuing binary flags).
  3. PagerDuty / SMS alerts notify the on-call MLOps engineer and attending laboratory director within 60 seconds.

---

## 7. Prediction Logging, Auditability & HIPAA Compliance

1. **Transaction Logging:** All API inference requests, patient accession IDs, probability scores, triage flags, and latencies are appended to `outputs/deployment_request_response.log`.
2. **Cryptographic Integrity:** Each scoring log entry includes a SHA-256 signature calculated from the input feature hash, prediction timestamp, and model version.
3. **HIPAA & De-Identification Safeguards:**
   - Raw patient identifiers (Full Name, Social Security Number, Date of Birth) are stripped by hospital LIS middleware prior to invoking the AI service.
   - Predictions are referenced exclusively by anonymized accession numbers (e.g., `LIS-ORD-2026-89412`).
   - Log files are rotated daily and archived to encrypted, access-restricted cold storage with a 7-year medical retention policy.

---

## 8. Clinical Ownership, RACI Matrix & Service Level Agreement (SLA)

### 8.1 RACI Governance Matrix

| Lifecycle Activity | Lead Cytopathologist | Lab Clinical Director | Lead MLOps Engineer | Hospital IT Admin |
|---|:---:|:---:|:---:|:---:|
| Clinical Objective & Recall Definition | **Accountable (A)** | Consulted (C) | Responsible (R) | Informed (I) |
| Model Training & Hyperparameter Tuning | Informed (I) | Informed (I) | **Accountable (A)** | Informed (I) |
| Pre-Deployment Validation Sign-Off | **Responsible (R)** | **Accountable (A)** | Responsible (R) | Informed (I) |
| API Deployment & Microservice Health | Informed (I) | Informed (I) | **Accountable (A)** | Responsible (R) |
| Daily Drift & Anomaly Monitoring | Consulted (C) | Informed (I) | **Accountable (A)** | Informed (I) |
| Retraining Authorization | **Accountable (A)** | Responsible (R) | Responsible (R) | Informed (I) |
| Emergency Model Rollback Execution | Informed (I) | Consulted (C) | **Accountable (A)** | Responsible (R) |

### 8.2 Operational Service Level Agreements (SLAs)

- **Service Availability:** $\ge 99.9\%$ uptime (excluding scheduled maintenance windows).
- **Inference Latency:** P95 latency $< 150$ ms; P99 latency $< 200$ ms.
- **Minimum Diagnostic Safety:** Malignant Recall $\ge 95.0\%$ across all rolling windows.
- **Incident Response Time:** Priority 1 (Clinical alert / Recall drop) response within 15 minutes; resolution or rollback within 1 hour.

---

## 9. Result

The continuous monitoring and lifecycle maintenance system was fully implemented and empirically verified. The Kolmogorov-Smirnov test and Population Stability Index accurately differentiated between stable patient batches and sensor/staining drift events. Retraining trigger logic, champion-challenger shadow evaluation, instant rollback protocols, and clinical governance RACI documentation were established and validated.

---

## 10. Conclusion

Experiment 8 completes the end-to-end lifecycle of the Breast Cancer Detection System. By establishing automated drift surveillance, explicit clinical retraining triggers, zero-downtime rollback fail-safes, and a rigorous clinical governance model, the machine learning system is equipped to maintain peak diagnostic accuracy, regulatory compliance, and patient safety over multi-year clinical deployments.
