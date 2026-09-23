# 🩺 Breast Cancer Wisconsin (Diagnostic) Detection System
> **Course:** CSE4001 — Applications of Machine Learning in Industry  
> **Student:** Rahul Kumar (Registration No: 230160223017)  
> **Domain:** Healthcare AI & Diagnostic Clinical Decision Support  
> **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC - 569 Samples, 30 Morphometric Features)

---

## 📌 1. Project Overview & Clinical Problem Statement

Breast cancer is one of the leading causes of cancer-related mortality among women worldwide. Early and accurate detection significantly increases survival rates. This project develops an end-to-end clinical machine learning classification pipeline using cell nuclear morphometric characteristics extracted from digitized Fine-Needle Aspirate (FNA) biopsy images.

### Clinical Objective:
Formulate an automated triage system that classifies breast masses as **Benign (0)** or **Malignant (1)**. In clinical screening, **False Negatives (missing a malignant case)** carry severe prognosis penalties. The models are optimized for **Malignant Recall ($\ge 95-98\%$)** and **ROC-AUC** while maintaining high specificity to prevent unnecessary surgical biopsies.

```
  Digitized FNA Image
          │
          ▼
  Nuclear Feature Extraction (30 Morphological Features)
  [Radius, Texture, Perimeter, Area, Smoothness, Concavity, etc.]
          │
          ▼
  StandardScaler Pipeline (Zero Data Leakage Fit on Train)
          │
          ▼
  Stratified 5-Fold GridSearchCV Tuning
          │
  ┌───────┴──────────────────────┬──────────────────────┐
  ▼                              ▼                      ▼
Logistic Regression (L2)     SVM (RBF Kernel)      Random Forest
[AUC: 0.9958 | Acc: 97.36%]  [AUC: 0.9960 | Acc: 97.58%]  [AUC: 0.9913 | Recall: 95.24%]
          │                              │                      │
          └──────────────────────┬──────────────────────┘
                                 ▼
                    Optimal Model Checkpoints (.pkl)
```

---

## 📊 2. Dataset Overview & Key Statistics

- **Total Cohort ($N$):** 569 patient biopsy samples
- **Feature Space:** 30 continuous real-valued nuclear attributes (10 base geometric characteristics across 3 statistical moments: *Mean*, *Standard Error (SE)*, and *Worst*):
  1. `radius` (mean distances from center to points on contour)
  2. `texture` (gray-scale variation standard deviation)
  3. `perimeter` (nuclear boundary length)
  4. `area` (cross-sectional area)
  5. `smoothness` (local radius variations)
  6. `compactness` ($\frac{\text{perimeter}^2}{\text{area}} - 1.0$)
  7. `concavity` (severity of contour concavities)
  8. `concave_points` (number of contour creases)
  9. `symmetry` (nuclear contour symmetry)
  10. `fractal_dimension` ("coastline" approximation $- 1.0$)
- **Target Variable (`diagnosis`):**
  - **Benign ($0$):** 357 cases ($62.74\%$)
  - **Malignant ($1$):** 212 cases ($37.26\%$)
- **Data Quality:** Verified 0 null/missing values, 0 duplicates, and zero synthetic artifact leakage.

---

## 🧪 3. Lab Experiments Progression (1 to 5)

| Experiment | Title | Key Activities & Methodology | Deliverables |
|---|---|---|---|
| **Exp 1** | **Objective Definition** | Clinical problem framing, target variable encoding, stakeholder alignment, metric formulation. | Lab Report & Submission PDF |
| **Exp 2** | **Dataset Details & Audit** | Schema inspection, provenance audit, class balance verification, data hygiene checks. | `data/`, Audit Log & PDF |
| **Exp 3** | **Data Preprocessing** | Dropping non-predictive `id`, stratified 80/10/10 train-val-test split, `StandardScaler` pipeline. | Scaled data splits & `scaler.pkl` |
| **Exp 4** | **Exploratory Data Analysis (EDA)** | Univariate distributions, comparative boxplots, Pearson correlation heatmaps, multicollinearity ($r > 0.90$), outliers. | 10 High-Res Charts & EDA Report |
| **Exp 5** | **Model Training & Tuning** | Baseline majority model, Logistic Regression (L2), SVM (RBF), Random Forest, 5-Fold Stratified GridSearchCV. | Checkpoints (`.pkl`), Charts, Docx & PDF |

---

## 📈 4. Experimental Results & Model Benchmark

### A. 5-Fold Stratified Cross-Validation (Training Partition $N=455$)
| Model Candidate | Best Hyperparameters | CV Accuracy | CV Recall (Malignant) | CV Precision | CV F1-Score | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Majority Baseline** | `strategy='most_frequent'` | $62.64\%$ | $0.00\%$ | $0.00\%$ | $0.000$ | $0.5000$ |
| **Logistic Regression** | `C=1.0, penalty='l2', solver='lbfgs'` | $97.36\%$ | **$95.29\%$** | $97.71\%$ | $0.9640$ | **$0.9958$** |
| **SVM (RBF Kernel)** | `C=5.0, gamma=0.01, kernel='rbf'` | **$97.58\%$** | $94.71\%$ | **$98.89\%$** | **$0.9666$** | **$0.9960$** |
| **Random Forest** | `n_est=50, max_depth=12, balanced` | $95.82\%$ | $94.12\%$ | $94.69\%$ | $0.9439$ | $0.9913$ |

### B. Independent Validation Holdout Set ($N=57$: 36 Benign, 21 Malignant)
| Model Architecture | Val Accuracy | Malignant Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 94.74% | 85.71% | **100.00%** | **100.00%** | 0.9231 | 0.9934 | $\text{TP}=18, \text{FN}=3, \text{TN}=36, \text{FP}=0$ |
| **SVM (RBF Kernel)** | **96.49%** | 90.48% | **100.00%** | **100.00%** | 0.9500 | 0.9934 | $\text{TP}=19, \text{FN}=2, \text{TN}=36, \text{FP}=0$ |
| **Random Forest** | **96.49%** | **95.24%** | 97.22% | 95.24% | **0.9524** | **0.9947** | $\text{TP}=20, \text{FN}=1, \text{TN}=35, \text{FP}=1$ |

---

## 🔍 5. Key Clinical & Technical Insights

1. **Top Biomarkers:** Feature importance (Random Forest Gini & Logistic Regression standardized weights) identified `concave_points_worst`, `perimeter_worst`, and `radius_worst` as the primary discriminators for malignancy ($r > 0.78$), matching clinical cytology findings that cancerous nuclei exhibit irregular contour folds and hypertrophy.
2. **False Negative Minimization:** Random Forest achieved **$95.24\%$ Malignant Recall** on holdout validation, producing only **$1$ False Negative**, proving ideal for initial screening triage.
3. **Flawless Specificity:** SVM with RBF kernel achieved **$100.00\%$ Specificity** ($\text{FP}=0$), eliminating false positive surgical referrals.

---

## 📁 6. Repository Structure

```
Health_Project_3_Breast_Cancer/
│
├── data/                                 # Raw & Stratified Processed Data
│   ├── breast_cancer.csv                 # Raw WDBC dataset
│   └── processed/                        # Train/Val/Test scaled features & target labels
│
├── models/                               # Serialized Trained Model Checkpoints (.pkl)
│   ├── logistic_regression_best.pkl      # Best Regularized Logistic Regression
│   ├── svm_rbf_best.pkl                  # Best Support Vector Classifier (RBF)
│   ├── random_forest_best.pkl            # Best Random Forest Classifier
│   ├── dummy_baseline.pkl                # Majority class lower-bound baseline
│   ├── breast_cancer_scaler.pkl          # Fitted StandardScaler pipeline
│   └── model_training_summary.json       # JSON metadata of hyperparameter search
│
├── notebooks/                            # Executable Python Experiment Scripts
│   ├── experiment_2_dataset_audit.py     # Data audit and validation
│   ├── experiment_3_data_preprocessing.py# Preprocessing and scaling
│   ├── experiment_4_data_exploration.py  # Comprehensive EDA and plotting
│   ├── experiment_5_model_training.py    # Training, GridSearchCV & validation
│   ├── generate_experiment_pdf.py        # Automated ReportLab PDF generator
│   └── generate_word_doc.py              # Word Document (.docx) generator
│
├── outputs/                              # High-Resolution Visualization Charts
│   ├── experiment4_boxplots_by_diagnosis.png
│   ├── experiment4_correlation_heatmap.png
│   ├── experiment4_radius_vs_area_scatter.png
│   ├── experiment5_cv_comparison.png
│   ├── experiment5_confusion_matrices.png
│   ├── experiment5_roc_curves.png
│   ├── experiment5_precision_recall_curves.png
│   ├── experiment5_feature_importance.png
│   └── experiment5_learning_curves.png
│
├── report/                               # Markdown Reports & Word Documents (.docx)
│   ├── Experiment_1_Objective_Definition.md
│   ├── Experiment_2_Dataset_Details.md
│   ├── Experiment_3_Data_Preprocessing.md
│   ├── Experiment_4_Data_Exploration_EDA.md
│   ├── Experiment_5_Model_Training.md
│   └── Experiment_5_Model_Training.docx
│
├── experiment_pdfs/                      # Ready-to-Submit Academic Lab PDFs
│   ├── Experiment_1_Objective_Definition.pdf
│   ├── Experiment_2_Dataset_Details.pdf
│   ├── Experiment_3_Data_Preprocessing.pdf
│   ├── Experiment_4_Data_Exploration_EDA.pdf
│   └── Experiment_5_Model_Training.pdf
│
├── summary/                              # Viva Preparation & Visual Guides
│   ├── VIVA_AND_EXPERIMENT_EXPLANATION_GUIDE.md
│   └── PROJECT_FILES_AND_VISUALS_GUIDE.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 7. Installation & Reproducibility Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Rahulkumar0029/Health_Project_3_Breast_Cancer.git
cd Health_Project_3_Breast_Cancer
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Pipelines
```bash
# Run Preprocessing (Exp 3)
python notebooks/experiment_3_data_preprocessing.py

# Run Exploratory Data Analysis (Exp 4)
python notebooks/experiment_4_data_exploration.py

# Run Model Training & Tuning (Exp 5)
python notebooks/experiment_5_model_training.py

# Generate Submission PDFs
python notebooks/generate_experiment_pdf.py

# Generate Word Document (.docx)
python notebooks/generate_word_doc.py
```

---

## 👤 Author & Acknowledgements

- **Developer:** Rahul Kumar
- **Registration No:** 230160223017
- **Course:** CSE4001 Applications of Machine Learning in Industry
- **Dataset Source:** [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian.
