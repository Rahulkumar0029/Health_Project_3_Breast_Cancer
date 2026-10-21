# 💻 Terminal Commands & Live Demonstration Guide
**Project:** Health-Project 3: Breast Cancer Detection System  
**Purpose:** Quick-reference cheat sheet for running any experiment on demand during Faculty Review / Viva.

---

## ⚡ Step 0: Initial Environment Activation (Do this first in PowerShell)

Open PowerShell in the project directory:
```powershell
cd "c:\Users\rahul\OneDrive\Desktop\7th sem\Health_Project_3_Breast_Cancer"
.\.venv\Scripts\activate
```
*(You will see `(.venv)` appear at the beginning of your command prompt)*

---

## 🎯 1. Experiment-by-Experiment Run Commands

### 🔹 Experiment 2: Dataset Provenance & Integrity Audit
If Sir says: *"Data audit run karke dikhao, missing values aur dimensions check karo."*
```powershell
python notebooks/experiment_2_dataset_audit.py
```
* **Output on Terminal:** 569 instances, 32 columns, 0 nulls, 0 duplicates, Benign (357, 62.7%), Malignant (212, 37.3%).
* **Files to show Sir:** `outputs/experiment2_class_distribution.png` and `outputs/experiment2_audit_summary.txt`.

---

### 🔹 Experiment 3: Preprocessing, Zero-Leakage Scaling & Splitting
If Sir says: *"Preprocessing run karo aur dikhao train/test split aur scaling kaise hui."*
```powershell
python notebooks/experiment_3_data_preprocessing.py
```
* **Output on Terminal:** Train set: 455 samples, Validation set: 57 samples, Test set: 57 samples. StandardScaler fit strictly on training set.
* **Files to show Sir:** `data/processed/` folder, `models/breast_cancer_scaler.pkl`, and `outputs/experiment3_scaling_comparison.png`.

---

### 🔹 Experiment 4: Exploratory Data Analysis (EDA) & Correlations
If Sir says: *"EDA run karo aur top cancer biomarkers aur correlation matrix dikhao."*
```powershell
python notebooks/experiment_4_data_exploration.py
```
* **Output on Terminal:** Top correlated features (`concave_points_worst` $r=0.7936$, `radius_worst` $r=0.7765$), 21 pairs with multicollinearity $|r|>0.90$.
* **Files to show Sir:** `outputs/experiment4_correlation_heatmap.png`, `outputs/experiment4_target_correlations.png`, and `outputs/experiment4_boxplots_by_diagnosis.png`.

---

### 🔹 Experiment 5: Model Training, Stratified 5-Fold CV & Tuning
If Sir says: *"Models dobara train karke cross-validation scores dikhao."*
```powershell
python notebooks/experiment_5_model_training.py
```
* **Output on Terminal:**
  - 5-Fold Stratified CV Scores:
    - **SVM (RBF Kernel):** CV ROC-AUC = **0.9960**, Accuracy = **97.58%**, Precision = **98.89%**
    - **Logistic Regression:** CV ROC-AUC = **0.9958**, Accuracy = **97.36%**
    - **Random Forest:** CV ROC-AUC = **0.9913**, Accuracy = **95.82%**
  - Validation holdout metrics and hyperparameter tuning heatmaps.
* **Files to show Sir:** `outputs/experiment5_cv_comparison.png`, `outputs/experiment5_confusion_matrices.png`, and `outputs/experiment5_roc_curves.png`.

---

### 🔹 Experiment 6: Held-Out Test Set Evaluation & Error Analysis
If Sir says: *"Unseen test set par evaluation run karo aur confusion matrix dikhao."*
```powershell
python notebooks/experiment_6_model_evaluation.py
```
* **Output on Terminal:**
  - Test Set ($N=57$) Metrics Table:
    - **SVM (RBF):** **100.00% Accuracy, 100.00% Recall (FN=0), 100.00% Specificity (FP=0), 1.0000 ROC-AUC, 0.0056 Brier Score**
    - **Random Forest:** 96.49% Accuracy, 90.48% Recall at $\tau=0.50$ (100% at $\tau^*=0.35$)
  - Discrepant test case manifest (`#2`, `#12`, `#41`) with morphometric Z-scores.
  - Champion model serialized to `models/champion_model.pkl`.
* **Files to show Sir:** `outputs/experiment6_confusion_matrices.png`, `outputs/experiment6_test_roc_curves.png`, and `outputs/experiment6_error_analysis.png`.

---

### 🔹 Experiment 7: Deployment Verification & Latency Benchmarks
If Sir says: *"Deployment service test karo aur latency benchmarks dikhao."*
```powershell
python notebooks/experiment_7_deployment.py
```
* **Output on Terminal:**
  - Single Benign Sample: Predicted Benign (P=0.0254, Flag: `NORMAL_QUEUE`)
  - Single Malignant Sample: Predicted Malignant (P=1.0000, Flag: `PRIORITY_PATHOLOGIST_REVIEW`)
  - Batch of 10 samples: 0.72 ms execution time
  - Hospital LIS Webhook: Automated routing to `URGENT_PRIORITY_CYTOPATHOLOGY_WORKLIST` with SHA-256 signature
  - Latency Benchmarks (100 requests): **P50: 6.32 ms, P95: 9.75 ms** (SLA: $<200\text{ ms}$ — PASS!)
* **Files to show Sir:** `outputs/experiment7_deployment_benchmarks.png`, `outputs/sample_request_response_log.json`, and `outputs/deployment_request_response.log`.

---

### 🔹 Experiment 8: Data Drift Surveillance & Retraining Trigger
If Sir says: *"Continuous monitoring aur data drift detection run karke dikhao."*
```powershell
python notebooks/experiment_8_monitoring_drift.py
```
* **Output on Terminal:**
  - Kolmogorov-Smirnov Test and Population Stability Index (PSI) table across top features.
  - Batch 1 (Stable): $\text{PSI} < 0.10 \implies$ `STABLE` (Recall 100%).
  - Batch 2 (Scanner/Stain Shift): `area_worst` $\text{PSI} = 2.7757$, `radius_worst` $\text{PSI} = 1.6560 \implies$ `DRIFT ALERT` $\implies$ **Automated Retraining Trigger Activated!**
  - Champion-Challenger validation and rollback plan execution.
* **Files to show Sir:** `outputs/experiment8_drift_analysis.png` and `outputs/experiment8_monitoring_summary.json`.

---

## 🌐 2. Live Interactive FastAPI Server & Swagger UI Demo

If Sir says: *"Mujhe live API chalakar browser me dikhao ya koi request bhejkar dikhao."*

### Step 1: Start the server
In your terminal, run:
```powershell
python -m uvicorn deployment.app:app --host 127.0.0.1 --port 8000 --reload
```
Terminal will show:
`INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

### Step 2: Open Browser
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/docs`**

### What to show Sir in the Browser:
1. **Interactive Swagger UI:** Show the clean documentation with endpoints `/health`, `/model-info`, `/predict`, `/batch-predict`, and `/lis/webhook`.
2. **Click on `/health` -> "Try it out" -> "Execute":** Shows `{"status": "UP", "model_loaded": true, "scaler_loaded": true}`.
3. **Click on `/predict` -> "Try it out" -> "Execute":** Runs a real-time prediction and returns probability, diagnosis class, triage alert (`NORMAL_QUEUE` or `PRIORITY_PATHOLOGIST_REVIEW`), and top contributing biomarkers.
4. **Click on `/lis/webhook` -> "Try it out" -> "Execute":** Simulates hospital LIS accession order routing to the cytopathology priority worklist with cryptographic SHA-256 signature.

*(To stop the server when done, simply press `Ctrl + C` in the terminal)*

---

## 📄 3. Re-Generate All 8 Lab PDFs on Demand

If Sir says: *"Sare experiments ke lab manual format wale PDFs generate karke dikhao."*
```powershell
python notebooks/generate_experiment_pdf.py
```
* Generates all 8 PDFs in `experiment_pdfs/`:
  - `Experiment_1_Objective_Definition.pdf`
  - `Experiment_2_Dataset_Details.pdf`
  - `Experiment_3_Data_Preprocessing.pdf`
  - `Experiment_4_Data_Exploration_EDA.pdf`
  - `Experiment_5_Model_Training.pdf`
  - `Experiment_6_Model_Evaluation.pdf`
  - `Experiment_7_Deployment.pdf`
  - `Experiment_8_Continuous_Update_Maintenance.pdf`

---

## ⚡ 4. Master One-Liner (Run Entire Project End-to-End)

If Sir says: *"Poora project start se end tak ek baar me run karke dikhao."*
```powershell
python notebooks/experiment_2_dataset_audit.py; python notebooks/experiment_3_data_preprocessing.py; python notebooks/experiment_4_data_exploration.py; python notebooks/experiment_5_model_training.py; python notebooks/experiment_6_model_evaluation.py; python notebooks/experiment_7_deployment.py; python notebooks/experiment_8_monitoring_drift.py; python notebooks/generate_experiment_pdf.py
```
