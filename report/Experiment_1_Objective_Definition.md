# Experiment 1: Objective Definition

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Status:** Completed  

---

## 1. Case Brief & Problem Statement

A diagnostic pathology laboratory processes high volumes of Fine-Needle Aspirate (FNA) biopsy samples for suspected breast cancer cases. Manual microscopic examination of nuclear morphological features is labor-intensive and subject to inter-observer variability. The laboratory requires an automated second-opinion machine learning triage system to analyze digitized morphological features of cell nuclei and automatically flag likely-malignant tumours for priority pathologist review.

---

## 2. Machine Learning Formulation

| Dimension | Specification |
|---|---|
| **ML Experiment Type** | Supervised Binary Classification |
| **Input Features** | 30 continuous numerical nuclear measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension across mean, standard error, and worst values) |
| **Target Variable** | Binary Diagnosis: `0 = Benign`, `1 = Malignant` |
| **Primary Evaluation Metric** | **Malignant Class Recall / Sensitivity** ($\ge 98\%$), ROC-AUC ($\ge 0.98$) |
| **Secondary Metrics** | Precision, F1-Score, False Negative Rate (FNR), Brier Score / Calibration Error |

---

## 3. Clinical Success Metrics and KPIs

In oncology diagnostic screening, the clinical cost of a **False Negative** (classifying a malignant tumor as benign, delaying critical cancer intervention) is drastically higher than a **False Positive** (requiring secondary manual pathologist review).

- **Primary KPI:** Maximize Malignant Class Recall ($\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$).
- **Secondary KPI:** Maintain Precision $\ge 90\%$ to prevent pathologist alert fatigue.
- **Operational KPI:** Inference latency under 200 ms per sample via REST API for seamless clinical laboratory integration.

---

## 4. Key Stakeholders

1. **Pathologists / Cytotechnologists:** Primary clinical users who receive prioritized triage alerts for high-risk biopsies.
2. **Oncology Patients:** Beneficiaries of faster turnaround times and reduced risk of missed diagnoses.
3. **Laboratory Clinical Directors:** Responsible for diagnostic quality assurance, regulatory compliance, and auditability.
4. **ML / MLOps Engineers:** Responsible for developing, testing, deploying, and maintaining the predictive pipeline.

---

## 5. Operational and Clinical Constraints

- **Interpretability & Transparency:** Predictions must provide clear probability confidence scores and feature attributions.
- **Data Leakage Safeguards:** Preprocessing statistics (scaling parameters) must be strictly isolated to training sets.
- **Class Imbalance:** Benign cases outnumber malignant cases (~63% vs 37%); evaluation must avoid misleading overall accuracy.
- **Clinical Governance:** The model functions strictly as an assistive second-opinion tool and does not autonomously issue patient diagnoses without licensed pathologist sign-off.
