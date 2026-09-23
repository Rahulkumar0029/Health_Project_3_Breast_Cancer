# Experiment 3: Data Pre-processing

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Status:** Completed  

---

## 1. Aim

To preprocess the Breast Cancer Wisconsin (Diagnostic) dataset by handling missing values, removing duplicate records, dropping non-predictive identifiers, encoding the target variable, analyzing class imbalance, scaling numerical features using standardization, and preparing a stratified train-validation-test dataset for machine-learning classification and clinical deployment.

---

## 2. Objectives

1. **Load and inspect** the raw breast cancer dataset structure and schema.
2. **Identify and handle** missing values and duplicate records using robust data hygiene practices.
3. **Remove unnecessary identifiers** (`id` and index artifacts) that do not provide diagnostic predictive utility.
4. **Encode categorical diagnosis labels** into binary numeric values ($M \rightarrow 1, B \rightarrow 0$) ensuring the positive class consistently represents malignancy.
5. **Analyze class distribution** and compute clinical imbalance weights to account for majority benign cases.
6. **Partition the data** into stratified training (80%), validation (10%), and testing (10%) subsets to preserve class proportions.
7. **Normalize and standardize** numerical features using `StandardScaler`, fitting strictly on the training partition to prevent data leakage.
8. **Persist the preprocessing scaler pipeline** and cleaned data splits for reproducible model training (Experiments 5 & 6) and deployment (Experiment 7).

---

## 3. Dataset Characteristics

| Metric / Property | Value | Notes |
|---|---|---|
| **Dataset Source** | UCI Machine Learning Repository / Kaggle | Breast Cancer Wisconsin (Diagnostic) |
| **Total Samples ($N$)** | 569 | Digitized fine-needle aspirate (FNA) biopsy samples |
| **Total Raw Columns** | 32 | 1 `id`, 1 `diagnosis`, 30 diagnostic features |
| **Predictive Features** | 30 | 10 nuclear measurements $\times$ 3 statistics (mean, se, worst) |
| **Target Variable** | `diagnosis` | Binary: Benign (`B` $\rightarrow$ 0), Malignant (`M` $\rightarrow$ 1) |
| **Missing Values** | 0 (0.00%) | Complete dataset verified |
| **Duplicate Rows** | 0 (0.00%) | No duplicate patient biopsy records |

---

## 4. Preprocessing Methodology and Steps

### Step 1: Data Ingestion and Attribute Cleanup
The dataset was loaded from `data/breast_cancer.csv`. Any unnamed parsing artifact columns (e.g., `Unnamed: 32`) and the metadata identifier column `id` were dropped from the feature matrix. Retaining `id` would cause the model to memorize arbitrary database keys rather than learning nuclear morphology.

### Step 2: Missing Value and Duplicate Auditing
- **Null Value Audit:** Verified 0 missing entries across all 30 features. If missing values are encountered in future batches, median imputation (`fillna(median)`) is incorporated into the pipeline to resist outlier distortion.
- **Duplicate Audit:** Zero duplicate records detected across the 569 instances.

### Step 3: Target Encoding and Clinical Definition
The target variable was mapped to binary integers:
$$\text{diagnosis} = \begin{cases} 1 & \text{if } \text{Malignant (M)} \\ 0 & \text{if } \text{Benign (B)} \end{cases}$$
*Clinical Rationale:* In oncology diagnostic workflows, missing a malignant case (False Negative) has severe health consequences. Setting Malignant as Class 1 establishes standard sensitivity/recall tracking during later evaluation.

### Step 4: Class Distribution Analysis
- **Benign ($y=0$):** 357 samples ($62.74\%$)
- **Malignant ($y=1$):** 212 samples ($37.26\%$)
- **Imbalance Ratio:** $1.68 : 1$ (Benign : Malignant)
- **Clinical Positive Class Weight:** $\frac{N_{\text{benign}}}{N_{\text{malignant}}} = \frac{357}{212} \approx 1.684$

### Step 5: Stratified Train-Validation-Test Splitting
A two-stage stratified split was performed with `random_state=42`:
1. **Train vs. Temp (80% / 20%):** 455 training samples, 114 temp samples.
2. **Validation vs. Test (50% / 50% of Temp):** 57 validation samples (10%), 57 test samples (10%).

Stratification ensures that the $62.7\% / 37.3\%$ class ratio is preserved identically across training, validation, and testing partitions.

### Step 6: Feature Standardization & Data Leakage Prevention
Because nuclear features exhibit vastly different physical scales (e.g., `area_mean` ranges up to $2500\,\text{mm}^2$ while `smoothness_mean` ranges from $0.05$ to $0.16$), standard score normalization (Z-score) is applied:
$$z = \frac{x - \mu}{\sigma}$$
- **Data Leakage Safeguard:** `StandardScaler` is fitted **strictly on the 455 training samples** ($\mu_{\text{train}}, \sigma_{\text{train}}$). The validation and test sets are transformed using these stored parameters without re-estimating statistical moments.

---

## 5. Experimental Results and Output Summary

```
======================================================================
EXPERIMENT 3: DATA PRE-PROCESSING EXECUTION SUMMARY
======================================================================
Raw Dataset Shape: (569, 32)
Features Matrix X: (569, 30)
Target Vector y:   (569,)

Target Class Counts:
- Benign (0):    357 (62.74%)
- Malignant (1): 212 (37.26%)

Split Partition Sizes:
- Training Set:   455 samples (80.0%) | Benign: 285, Malignant: 170
- Validation Set:  57 samples (10.0%) | Benign:  36, Malignant:  21
- Test Set:        57 samples (10.0%) | Benign:  36, Malignant:  21

StandardScaler Output:
- X_train_scaled Mean: ~0.0000 | Std Dev: ~1.0000
- Persisted Artifact: outputs/breast_cancer_scaler.pkl
```

---

## 6. Visualizations

1. **Target Distribution & Stratified Splits:** `outputs/experiment3_class_distribution.png` confirms exact stratified proportions across all subsets.
2. **Feature Scaling Comparison:** `outputs/experiment3_scaling_comparison.png` illustrates the transformation of raw skewed features (`radius_mean`, `area_mean`, `smoothness_mean`) into standardized zero-mean unit-variance distributions.

---

## 7. Persisted Artifacts

The following train-ready datasets and pipeline objects were generated and saved:
- `outputs/breast_cancer_scaler.pkl` & `models/breast_cancer_scaler.pkl` — Trained `StandardScaler` pipeline.
- `data/processed/X_train_scaled.csv`, `y_train.csv` — Training partition ($N=455$).
- `data/processed/X_val_scaled.csv`, `y_val.csv` — Validation partition ($N=57$).
- `data/processed/X_test_scaled.csv`, `y_test.csv` — Test partition ($N=57$).
- `data/processed/train_preprocessed.csv`, `val_preprocessed.csv`, `test_preprocessed.csv` — Combined feature-target sets.
- `data/processed/breast_cancer_preprocessed.csv` — Cleaned full dataset for EDA (Experiment 4).

---

## 8. Observations

1. The raw dataset contains 569 fine-needle aspirate biopsy records with zero missing entries and zero duplicate rows.
2. Dropping the `id` column prevents machine learning algorithms from overfitting to sample identifiers.
3. Encoding Malignant as `1` and Benign as `0` establishes unambiguous positive-class labeling for oncology screening.
4. Stratified partitioning successfully maintains the ~63% : 37% class balance across training, validation, and test splits.
5. Standardizing features eliminates dimensional dominance by high-magnitude variables (like `area`), facilitating faster gradient descent convergence and equal weighting in distance-based algorithms (SVM, KNN, Logistic Regression).
6. Preserving the fitted `StandardScaler` artifact ensures zero data leakage and ensures seamless inference in the production FastAPI scoring service (Experiment 7).

---

## 9. Conclusion

The Breast Cancer Wisconsin (Diagnostic) dataset has been successfully cleaned, encoded, stratified, standardized, and saved. The preprocessed data and persisted scaling pipeline are now fully prepared for Exploratory Data Analysis (Experiment 4) and candidate model training (Experiment 5).
