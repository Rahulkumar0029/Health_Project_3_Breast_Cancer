# Project Summary & Complete File Index: Breast Cancer Detection System
**Course:** CSE4001 — Applications of Machine Learning in Industry  
**Domain:** Health-Project 3 (Breast Cancer Wisconsin Diagnostic)  
**Scope:** Complete End-to-End Documentation, Pipelines, Visuals & Deployment (Experiments 1 to 8)

---

## 📁 1. Master Repository Directory Structure & File Map

```
Health_Project_3_Breast_Cancer/
│
├── data/                                 <-- Raw & Processed Datasets
│   ├── breast_cancer.csv                 <-- Original WDBC dataset (569 rows, 32 columns)
│   ├── wdbc.data                         <-- Raw UCI repository source file
│   └── processed/                        <-- Preprocessed & Partitioned datasets (Exp 3)
│       ├── breast_cancer_preprocessed.csv<-- Cleaned, target-encoded dataset (no ID, no nulls)
│       ├── train_preprocessed.csv        <-- 80% Training partition (455 samples)
│       ├── val_preprocessed.csv          <-- 10% Validation partition (57 samples)
│       ├── test_preprocessed.csv         <-- 10% Test partition (57 samples)
│       ├── X_train_scaled.csv            <-- Standardized (Z-score) training features
│       ├── X_val_scaled.csv              <-- Scaled validation features
│       ├── X_test_scaled.csv             <-- Scaled test features
│       ├── y_train.csv                   <-- Training labels (0=Benign, 1=Malignant)
│       ├── y_val.csv                     <-- Validation labels
│       └── y_test.csv                    <-- Test labels
│
├── models/                               <-- Trained Model Artifacts & Checkpoints
│   ├── champion_model.pkl                <-- Production Champion: Support Vector Machine (RBF)
│   ├── voting_ensemble_model.pkl         <-- Soft-Voting Ensemble (LR + SVM + RF)
│   ├── svm_rbf_best.pkl                  <-- Serialized Support Vector Classifier (RBF Kernel)
│   ├── random_forest_best.pkl            <-- Serialized Random Forest Classifier (50 Trees)
│   ├── logistic_regression_best.pkl      <-- Serialized L2-Regularized Logistic Regression
│   ├── dummy_baseline.pkl                <-- Zero-Rule majority baseline heuristic
│   ├── breast_cancer_scaler.pkl          <-- Saved StandardScaler object (fit on train)
│   ├── model_metadata.json               <-- Complete production model metadata & provenance
│   └── model_training_summary.json       <-- Exp 5 Cross-validation training logs
│
├── deployment/                           <-- Enterprise Deployment & LIS Integration (Exp 7)
│   ├── app.py                            <-- Production FastAPI scoring service & LIS webhook
│   └── test_api.py                       <-- Automated API verification & test suite
│
├── notebooks/                            <-- Python Executable Pipelines for All Experiments
│   ├── experiment_2_dataset_audit.py     <-- Exp 2 data audit & provenance script
│   ├── experiment_3_data_preprocessing.py<-- Exp 3 cleaning, scaling & stratified splitting
│   ├── experiment_4_data_exploration.py  <-- Exp 4 full 18-step EDA & figure generation
│   ├── experiment_5_model_training.py    <-- Exp 5 training, Stratified 5-Fold CV & tuning
│   ├── experiment_6_model_evaluation.py  <-- Exp 6 test set evaluation & error analysis
│   ├── experiment_7_deployment.py        <-- Exp 7 deployment benchmarking & test runner
│   ├── experiment_8_monitoring_drift.py  <-- Exp 8 data drift monitoring & maintenance pipeline
│   ├── generate_experiment_pdf.py        <-- ReportLab PDF compiler for all Experiments 1-8
│   └── generate_word_doc.py              <-- Word document generator
│
├── outputs/                              <-- High-Resolution Figures, Logs & Artifacts
│   ├── experiment2_audit_summary.txt     <-- Exp 2 data audit findings text log
│   ├── experiment2_class_distribution.png<-- Exp 2 raw target class distribution chart
│   ├── experiment3_class_distribution.png<-- Exp 3 stratified train/val/test split verification
│   ├── experiment3_scaling_comparison.png<-- Exp 3 Before vs After Z-score standardization
│   ├── experiment3_preprocessing_summary.txt <-- Exp 3 split sizes & class ratio logs
│   ├── experiment4_class_distribution.png<-- Exp 4 Benign (357) vs Malignant (212) bar chart
│   ├── experiment4_feature_distributions.png <-- Exp 4 10-feature histograms with KDE curves
│   ├── experiment4_boxplots_by_diagnosis.png <-- Exp 4 Boxplots comparing Benign vs Malignant
│   ├── experiment4_correlation_heatmap.png   <-- Exp 4 Full 30-feature Pearson correlation matrix
│   ├── experiment4_mean_correlation_matrix.png<-- Exp 4 10-mean features correlation with numbers
│   ├── experiment4_outlier_detection.png     <-- Exp 4 Outlier identification boxplots
│   ├── experiment4_classwise_means.png       <-- Exp 4 Average feature values by diagnosis bar chart
│   ├── experiment4_radius_vs_area_scatter.png<-- Exp 4 Bivariate scatter plot showing decision boundary
│   ├── experiment4_target_correlations.png   <-- Exp 4 Ranking of 30 features correlated with cancer
│   ├── experiment4_eda_summary.txt           <-- Exp 4 Statistical summary of correlations & metrics
│   ├── experiment5_cv_comparison.png         <-- Exp 5 5-Fold Stratified CV benchmark across models
│   ├── experiment5_confusion_matrices.png    <-- Exp 5 Confusion Matrices on Validation set (N=57)
│   ├── experiment5_roc_curves.png            <-- Exp 5 Validation ROC curves (All >0.99 AUC)
│   ├── experiment5_precision_recall_curves.png <-- Exp 5 Validation Precision-Recall curves
│   ├── experiment5_feature_importance.png    <-- Exp 5 RF Gini Importance vs LR Weights
│   ├── experiment5_learning_curves.png       <-- Exp 5 Learning Curves (Diagnosing bias/variance)
│   ├── experiment5_hyperparameter_heatmaps.png <-- Exp 5 SVM (C vs gamma) Grid Search heatmaps
│   ├── experiment5_training_summary.txt      <-- Exp 5 Complete text summary of CV and Val results
│   ├── experiment6_test_roc_curves.png       <-- Exp 6 Unseen test set ROC curves (AUC=1.000)
│   ├── experiment6_test_pr_curves.png        <-- Exp 6 Unseen test set Precision-Recall curves
│   ├── experiment6_confusion_matrices.png    <-- Exp 6 Test set 2x2 confusion matrices (Zero FN!)
│   ├── experiment6_threshold_tuning.png      <-- Exp 6 Decision threshold sensitivity sweep curve
│   ├── experiment6_error_analysis.png        <-- Exp 6 Discrepant case morphometric Z-score profiles
│   ├── experiment6_model_comparison.png      <-- Exp 6 Multi-metric bar comparison across all models
│   ├── experiment6_evaluation_summary.json   <-- Exp 6 Complete JSON evaluation manifest
│   ├── experiment6_evaluation_summary.txt    <-- Exp 6 Formatted text evaluation summary
│   ├── experiment7_deployment_benchmarks.png <-- Exp 7 Latency histogram (P95=9.75ms) & LIS queue
│   ├── sample_request_response_log.json      <-- Exp 7 Audited API request/response JSON payload log
│   ├── deployment_request_response.log       <-- Exp 7 Live transaction log with timestamps & SHA-256
│   ├── experiment7_deployment_summary.txt    <-- Exp 7 Formatted deployment report
│   ├── experiment8_drift_analysis.png        <-- Exp 8 Statistical drift plots (KS-test & PSI charts)
│   ├── experiment8_monitoring_summary.json   <-- Exp 8 Monitoring & drift audit JSON manifest
│   └── experiment8_monitoring_summary.txt    <-- Exp 8 Continuous update & maintenance text summary
│
├── report/                               <-- Comprehensive Markdown Academic Reports
│   ├── Experiment_1_Objective_Definition.md  <-- 1-page Problem Definition & KPI Document
│   ├── Experiment_2_Dataset_Details.md       <-- Dataset Description & Provenance Report
│   ├── Experiment_3_Data_Preprocessing.md    <-- Preprocessing & Zero-Leakage Pipeline Report
│   ├── Experiment_4_Data_Exploration_EDA.md  <-- Comprehensive 18-step EDA Report
│   ├── Experiment_5_Model_Training.md        <-- Model Training, 5-Fold CV & Tuning Report
│   ├── Experiment_6_Model_Evaluation.md      <-- Test Evaluation, Error Analysis & Selection Report
│   ├── Experiment_7_Deployment.md            <-- FastAPI Scoring Service & LIS Integration Report
│   └── Experiment_8_Continuous_Update_Maintenance.md <-- Monitoring, Drift & Maintenance Plan
│
├── experiment_pdfs/                      <-- Publication-Quality ReportLab Generated PDFs (All 8 Exps)
│   ├── Experiment_1_Objective_Definition.pdf
│   ├── Experiment_2_Dataset_Details.pdf
│   ├── Experiment_3_Data_Preprocessing.pdf
│   ├── Experiment_4_Data_Exploration_EDA.pdf
│   ├── Experiment_5_Model_Training.pdf
│   ├── Experiment_6_Model_Evaluation.pdf
│   ├── Experiment_7_Deployment.pdf
│   └── Experiment_8_Continuous_Update_Maintenance.pdf
│
└── summary/                              <-- Master Presentation & Viva Resources
    ├── FULL_VIVA_COMPLETE_MASTER_GUIDE.md    <-- Master Comprehensive Viva Guide (Exps 1 to 8 + Top 35 Q&A)
    ├── VIVA_AND_EXPERIMENT_EXPLANATION_GUIDE.md <-- Quick viva revision guide
    └── PROJECT_FILES_AND_VISUALS_GUIDE.md    <-- This master file index
```

---

## 🔬 2. Visual Figures & Charts Catalog

| Figure Filename | Experiment | Clinical & Technical Purpose |
|---|:---:|---|
| `experiment2_class_distribution.png` | Exp 2 | Shows raw dataset class imbalance (357 Benign vs 212 Malignant). |
| `experiment3_class_distribution.png` | Exp 3 | Verifies identical 62.7% : 37.3% ratio across Train, Val, and Test splits. |
| `experiment3_scaling_comparison.png` | Exp 3 | Visualizes raw vs standardized feature distributions ($\mu=0, \sigma=1$). |
| `experiment4_correlation_heatmap.png`| Exp 4 | Full 30-feature Pearson heatmap demonstrating $|r|>0.90$ multicollinearity. |
| `experiment4_boxplots_by_diagnosis.png`| Exp 4 | Shows distinct separation between Benign and Malignant nuclear contours. |
| `experiment4_target_correlations.png`| Exp 4 | Bar chart ranking the top predictive biomarkers (`concave_points_worst`). |
| `experiment5_cv_comparison.png`      | Exp 5 | 5-Fold Stratified CV benchmark (SVM peak: 0.9960 ROC-AUC). |
| `experiment5_learning_curves.png`    | Exp 5 | Learning curves proving high generalization without overfitting. |
| `experiment5_hyperparameter_heatmaps.png`| Exp 5| SVM $C$ vs $\gamma$ surface tuning surface. |
| `experiment6_test_roc_curves.png`    | Exp 6 | Held-out test set ROC curves (SVM & Ensemble: AUC = 1.0000). |
| `experiment6_test_pr_curves.png`     | Exp 6 | Precision-Recall curves illustrating positive predictive power. |
| `experiment6_confusion_matrices.png` | Exp 6 | 2x2 confusion matrix grid proving zero False Negatives on SVM. |
| `experiment6_threshold_tuning.png`   | Exp 6 | Decision threshold sweep showing $\tau^*=0.35$ eliminates tree false negatives. |
| `experiment6_error_analysis.png`     | Exp 6 | Cytomorphological Z-score profiles of borderline test specimens. |
| `experiment6_model_comparison.png`   | Exp 6 | Multi-metric comparison bar chart establishing SVM as champion. |
| `experiment7_deployment_benchmarks.png`| Exp 7| FastAPI latency distribution (P95: 9.75 ms) and LIS queue allocation. |
| `experiment8_drift_analysis.png`     | Exp 8 | KS-test $p$-values, PSI scores, and rolling recall longitudinal recovery. |

---

## 🏆 3. Quick Verification Checklist for Project Submission

- [x] **Experiment 1:** Completed (Report + PDF).
- [x] **Experiment 2:** Completed (Script + Report + PDF + Visuals).
- [x] **Experiment 3:** Completed (Script + Processed CSVs + Scaler .pkl + Report + PDF).
- [x] **Experiment 4:** Completed (Script + 12 Visuals + Report + PDF).
- [x] **Experiment 5:** Completed (Script + Trained .pkl Checkpoints + Report + PDF).
- [x] **Experiment 6:** Completed (`experiment_6_model_evaluation.py` + Champion `champion_model.pkl` + Report + PDF + Visuals).
- [x] **Experiment 7:** Completed (`deployment/app.py` + `deployment/test_api.py` + `experiment_7_deployment.py` + Audit Logs + Report + PDF + Visuals).
- [x] **Experiment 8:** Completed (`experiment_8_monitoring_drift.py` + Retraining Triggers + Rollback Plan + Report + PDF + Visuals).
- [x] **Master Viva Guide:** Completed (`summary/FULL_VIVA_COMPLETE_MASTER_GUIDE.md` with 35+ Q&A and 2-min pitch).
