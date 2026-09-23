# Project Summary & File Index: Breast Cancer Detection System
**Course:** CSE4001 — Applications of Machine Learning in Industry  
**Domain:** Health-Project 3 (Breast Cancer Wisconsin Diagnostic)  
**Scope:** Experiments 1 to 5 Complete Documentation & Visuals

---

## 📁 1. Project Directory Structure & Purpose of Every File

```
Health_Project_3_Breast_Cancer/
│
├── data/                                 <-- Raw & Processed Datasets
│   ├── breast_cancer.csv                 <-- Original WDBC dataset (569 rows, 32 columns)
│   ├── wdbc.data                         <-- Raw UCI repository file
│   └── processed/                        <-- Preprocessed & Partitioned datasets (from Exp 3)
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
├── models/                               <-- Trained Model Artifacts & Metadata (from Exp 5)
│   ├── logistic_regression_best.pkl      <-- Serialized L2-Regularized Logistic Regression model
│   ├── svm_rbf_best.pkl                  <-- Serialized Support Vector Classifier (RBF Kernel)
│   ├── random_forest_best.pkl            <-- Serialized Random Forest Classifier (50 Trees)
│   ├── dummy_baseline.pkl                <-- Zero-Rule majority baseline heuristic
│   ├── breast_cancer_scaler.pkl          <-- Saved StandardScaler object (fit on train)
│   └── model_training_summary.json       <-- Machine-readable JSON training metrics & best params
│
├── notebooks/                            <-- Python Executable Scripts for Experiments
│   ├── experiment_2_dataset_audit.py     <-- Script for Exp 2 data audit & validation
│   ├── experiment_3_data_preprocessing.py<-- Script for Exp 3 preprocessing, scaling & splitting
│   ├── experiment_4_data_exploration.py  <-- Script for Exp 4 full EDA & figure generation
│   ├── experiment_5_model_training.py    <-- Script for Exp 5 model training, CV & tuning
│   └── generate_experiment_pdf.py        <-- Automated ReportLab PDF generator for Exps 1-5
│
├── outputs/                              <-- High-Resolution Generated Figures & Artifacts
│   ├── experiment2_audit_summary.txt     <-- Exp 2 data audit findings text log
│   ├── experiment2_class_distribution.png<-- Exp 2 raw target class distribution chart
│   ├── experiment3_class_distribution.png<-- Exp 3 stratified train/val/test split verification
│   ├── experiment3_scaling_comparison.png<-- Exp 3 Before vs After Z-score standardization densities
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
│   ├── experiment5_roc_curves.png            <-- Exp 5 Validation ROC curves (AUC comparison)
│   ├── experiment5_precision_recall_curves.png<-- Exp 5 Validation PR curves vs prevalence baseline
│   ├── experiment5_feature_importance.png    <-- Exp 5 RF Gini Importance vs LR feature weights
│   ├── experiment5_learning_curves.png       <-- Exp 5 Sample size vs ROC-AUC learning curves
│   ├── experiment5_hyperparameter_heatmaps.png<-- Exp 5 SVM C vs Gamma hyperparameter 2D surface
│   └── experiment5_training_summary.txt      <-- Exp 5 Complete CV & validation metrics text log
│
├── report/                               <-- Full Markdown Lab Reports
│   ├── Experiment_1_Objective_Definition.md
│   ├── Experiment_2_Dataset_Details.md
│   ├── Experiment_3_Data_Preprocessing.md
│   ├── Experiment_4_Data_Exploration_EDA.md
│   └── Experiment_5_Model_Training.md
│
├── experiment_pdfs/                      <-- Ready-to-Submit Academic PDFs
│   ├── Experiment_1_Objective_Definition.pdf
│   ├── Experiment_2_Dataset_Details.pdf
│   ├── Experiment_3_Data_Preprocessing.pdf
│   ├── Experiment_4_Data_Exploration_EDA.pdf
│   └── Experiment_5_Model_Training.pdf
│
└── summary/                              <-- Viva & Lab Guide (This Folder)
    ├── VIVA_AND_EXPERIMENT_EXPLANATION_GUIDE.md
    └── PROJECT_FILES_AND_VISUALS_GUIDE.md
```

---

## 🖼️ 2. Detailed Explanation of Experiment 5 Images & Charts

### 📊 Image 1: `experiment5_cv_comparison.png`
* **Kya h (What it is):** Grouped bar chart with error bars comparing 5-Fold Cross-Validation Accuracy, Malignant Recall, Precision, F1-Score, and ROC-AUC for Logistic Regression, SVM, and Random Forest.
* **Interpretation for Sir:** "Sir, is chart me 5-Fold Cross-Validation ka benchmark hai. SVM (RBF) ne highest overall score achieve kiya ($0.9960\text{ ROC-AUC}, 97.58\%\text{ Accuracy}, 98.89\%\text{ Precision}$). Logistic Regression ne $95.29\%$ CV recall achieve kiya. Sabhi models $95\%$ target benchmark line se upar perform kar rahe hain."

---

### 📊 Image 2: `experiment5_confusion_matrices.png`
* **Kya h (What it is):** Side-by-side Confusion Matrices on the independent validation holdout set ($N=57$: 36 Benign, 21 Malignant) displaying exact counts and percentages.
* **Interpretation for Sir:** "Sir, clinical oncology me False Negatives (FN) sabse dangerous hote hain kyunki cancer patient ko healthy ghoshit kar diya jata hai. Random Forest ne sabse kam False Negatives diye ($\text{FN}=1$, Recall $= 95.24\%$). SVM aur Logistic Regression ne zero False Positives diye ($\text{FP}=0$, Specificity $= 100\%$), jisse unnecessary biopsies avoid hoti hain."

---

### 📊 Image 3: `experiment5_roc_curves.png`
* **Kya h (What it is):** Receiver Operating Characteristic (ROC) curves plotting True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across all discrimination thresholds.
* **Interpretation for Sir:** "Sir, teeno models ki ROC curves top-left corner me hug kar rahi hain with AUC $> 0.9934$. Baseline diagonal line (No-Skill AUC = 0.500) se massive improvement show karti hai, proving high threshold discriminability."

---

### 📊 Image 4: `experiment5_precision_recall_curves.png`
* **Kya h (What it is):** Precision-Recall curves showing trade-offs between precision and recall against the dataset prevalence baseline ($36.8\%$).
* **Interpretation for Sir:** "Sir, imbalanced datasets me PR curve accuracy se behtar evaluation deti hai. Random Forest ne highest PR-AUC ($0.9918$) achieve kiya, jo prove karta hai ki high recall maintain karte hue bhi precision drop nahi hota."

---

### 📊 Image 5: `experiment5_feature_importance.png`
* **Kya h (What it is):** Random Forest Mean Decrease in Impurity (Gini) Top-12 features alongside Logistic Regression standardized weight coefficients.
* **Interpretation for Sir:** "Sir, is plot se model interpretability prove hoti hai. `concave_points_worst`, `perimeter_worst`, aur `radius_worst` dono models me top predictive drivers hain, jo Experiment 4 ke EDA findings ($r > 0.78$) ke sath $100\%$ align karte hain."

---

### 📊 Image 6: `experiment5_learning_curves.png`
* **Kya h (What it is):** Learning curves plotting Training ROC-AUC vs 5-Fold CV ROC-AUC over increasing training sample sizes (from 45 to 455 samples).
* **Interpretation for Sir:** "Sir, learning curves se confirm hota hai ki model overfit ya underfit nahi ho raha. Training aur validation curves tightly converge ho rahi hain ($>0.99$), showing high sample efficiency and generalization."

---

### 📊 Image 7: `experiment5_hyperparameter_heatmaps.png`
* **Kya h (What it is):** 2D Heatmap showing SVM cross-validation ROC-AUC score surface across different combinations of regularization parameter $C$ and kernel coefficient $\gamma$.
* **Interpretation for Sir:** "Sir, heatmap se dikhta hai ki $C=5.0$ aur $\gamma=0.01$ par SVM peak performance ($0.9960$) deta hai. Agar gamma zyada high ho jaye ($>0.1$), to model overfit karne lagta hai."
