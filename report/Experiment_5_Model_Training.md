# Experiment 5: Model Training, Hyperparameter Tuning & Cross-Validation

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Status:** Completed  

---

## 1. Aim

To implement, train, hyperparameter-tune, and validate candidate machine learning classification algorithms (**Logistic Regression**, **Support Vector Machine**, and **Random Forest**) on the standardized Breast Cancer Wisconsin dataset using Stratified $k$-Fold Cross-Validation, compare model performance against baseline heuristics, and save the optimal model checkpoints for downstream evaluation.

---

## 2. Objectives

1. **Establish Baseline Classifiers:** Implement a Zero-Rule Dummy classifier (majority class heuristic) and an untuned standard Logistic Regression baseline to define the minimum performance bounds.
2. **Train Candidate Model Families:** Implement three distinct supervised classification paradigms:
   - *Linear Parametric Model:* Regularized Logistic Regression with L2 (Ridge) penalty.
   - *Non-Linear Maximum-Margin Classifier:* Support Vector Machine (SVM) with Radial Basis Function (RBF) and Linear kernels.
   - *Non-Parametric Ensemble Classifier:* Random Forest with Bootstrap Aggregation (Bagging) and random feature subspace sampling.
3. **Hyperparameter Optimization:** Conduct systematic Stratified Grid Search Cross-Validation (`GridSearchCV`, $k=5$) over regularizers ($C$), kernel coefficients ($\gamma$), tree depths, and sample splitting criteria.
4. **Evaluate Multi-Metric Generalization:** Quantify predictive performance across balanced accuracy, malignant recall (sensitivity), specificity, precision, F1-score, ROC-AUC, and Average Precision (PR-AUC).
5. **Analyze Error Boundaries & Confusion Matrices:** Evaluate models on the holdout validation set ($N=57$) to assess clinical trade-offs between False Negatives (missed malignancies) and False Positives (unnecessary biopsies).
6. **Interpret Feature Attribution:** Extract and compare Mean Decrease in Impurity (Gini) feature importances from Random Forest against standardized coefficients from Logistic Regression.
7. **Serialize Production Artifacts:** Save optimal trained model pipelines (`.pkl`), training logs, and evaluation manifests in `models/` and `outputs/`.

---

## 3. Mathematical Foundations of Candidate Algorithms

```
                                  ┌──────────────────────────────┐
                                  │   Standardized Train Data    │
                                  │  (N = 455, 30 Scaled Feats)  │
                                  └──────────────┬───────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   │                             │                             │
                   ▼                             ▼                             ▼
       ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
       │  Logistic Regression  │     │ Support Vector Mach.  │     │     Random Forest     │
       │  P(y=1|x)=σ(w^T x + b)│     │  max Margin s.t. K(x) │     │  Bagg. of 50-200 DTs  │
       └───────────┬───────────┘     └───────────┬───────────┘     └───────────┬───────────┘
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │ Stratified 5-Fold CV Tuning  │
                                  │ (Scoring Metric: ROC-AUC)    │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Checkpoints & Logs (.pkl)   │
                                  │   Validation Set (N = 57)    │
                                  └──────────────────────────────┘
```

### 3.1 Logistic Regression with L2 Regularization (Ridge)

Logistic Regression estimates the posterior probability of malignancy $P(y=1|\mathbf{x})$ through the logistic sigmoid activation:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

To mitigate multicollinearity detected in Experiment 4 (where $|r| > 0.90$ for nuclear geometry features), an L2 weight regularization penalty is integrated into the negative log-likelihood (binary cross-entropy) loss function:

$$\mathcal{L}_{\text{LR}}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \ln (1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

where $C > 0$ is the inverse regularization strength. Smaller $C$ enforces greater weight shrinkage towards zero.

---

### 3.2 Support Vector Machine (SVM) with RBF Kernel

Support Vector Machines construct an optimal separating hyperplane in a high-dimensional reproducing kernel Hilbert space (RKHS) that maximizes the geometric margin $\frac{2}{\|\mathbf{w}\|}$ while penalizing margin violations via slack variables $\xi_i$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Non-linear morphometric boundaries are mapped implicitly using the Radial Basis Function (Gaussian) kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right), \quad \gamma = \frac{1}{2\sigma^2}$$

- **$C$ Parameter:** Governs the trade-off between maximizing margin width and penalizing classification errors.
- **$\gamma$ (Gamma) Parameter:** Defines the radius of influence for individual support vectors.

---

### 3.3 Random Forest Classifier (Ensemble Bagging)

Random Forest constructs an ensemble of $B$ decorrelated decision trees $\{T_1, T_2, \dots, T_B\}$. Each tree is trained on a bootstrap sample $\mathcal{D}_b \subset \mathcal{D}_{\text{train}}$ of size $N$, selecting a random subset of $m = \sqrt{p}$ features at each candidate split node.

The split criterion minimizes Gini Impurity $I_G$:

$$I_G(t) = 1 - \sum_{k=0}^{1} p_k^2$$

The ensemble prediction aggregates individual tree posterior probability distributions:

$$\hat{P}(y=1|\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

Random Forest provides natural resilience against monotonic transformations, handles non-linear interactions without explicit kernel selection, and produces Mean Decrease in Impurity (MDI) feature importances.

---

## 4. Experimental Setup & Dataset Partitions

The training pipeline uses the clean, standardized partitions generated in Experiment 3:

| Dataset Partition | Sample Count ($N$) | Benign Cases ($y=0$) | Malignant Cases ($y=1$) | Proportion | Feature Scaling |
|---|---|---|---|---|---|
| **Training Set (`X_train_scaled`)** | **455** | 285 (62.64%) | 170 (37.36%) | 80.0% | `StandardScaler` fitted on train |
| **Validation Set (`X_val_scaled`)** | **57** | 36 (63.16%) | 21 (36.84%) | 10.0% | Transformed via train scaler |
| **Holdout Test Set (`X_test_scaled`)** | **57** | 36 (63.16%) | 21 (36.84%) | 10.0% | Transformed via train scaler |
| **Total Cohort** | **569** | **357 (62.74%)** | **212 (37.26%)** | **100.0%** | 30 Continuous Features |

- **Cross-Validation Scheme:** Stratified 5-Fold CV ($K=5$, shuffle=True, seed=42) on the 455 training samples.
- **Primary Optimization Metric:** `roc_auc` (to ensure optimal ranking and threshold flexibility across clinical operating points).

---

## 5. Algorithm & Methodology

```
Algorithm: Model Training & Hyperparameter Tuning Pipeline
─────────────────────────────────────────────────────────────────────────────
Input  : X_train_scaled, y_train, X_val_scaled, y_val (from Experiment 3)
Output : Trained model checkpoints (.pkl), performance metrics, visual plots
─────────────────────────────────────────────────────────────────────────────
1. Load X_train_scaled, y_train, X_val_scaled, y_val from 'data/processed/'.
2. Fit Baseline DummyClassifier(strategy='most_frequent') to establish lower bound.
3. Fit Untuned LogisticRegression(random_state=42) as baseline linear model.
4. Define StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
5. For each candidate model family (Logistic Regression, SVM, Random Forest):
     a. Define multi-dimensional hyperparameter grid.
     b. Configure GridSearchCV with scoring=[accuracy, recall, precision, f1, roc_auc].
     c. Execute grid search on training partition with refit='roc_auc'.
     d. Extract best estimator, optimal hyperparameters, and cross-validation scores (mean ± std).
     e. Record elapsed execution time and training logs.
6. Evaluate best estimators on holdout validation set (N=57):
     a. Predict class labels and posterior probabilities.
     b. Compute Confusion Matrix (TP, FP, TN, FN).
     c. Calculate Accuracy, Malignant Recall, Specificity, Precision, F1-score, and ROC-AUC.
7. Save serialized model artifacts (.pkl) to 'models/' directory.
8. Generate publication-quality figures (CV comparison, Confusion Matrices, ROC curves, PR curves, Feature Importance, Learning curves, Tuning heatmap).
9. Output training summary JSON and execution log text file.
─────────────────────────────────────────────────────────────────────────────
```

---

## 6. Python Implementation Code

The complete script is located at [`notebooks/experiment_5_model_training.py`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/notebooks/experiment_5_model_training.py).

### 6.1 Hyperparameter Grid Definition & Model Training

```python
# Stratified 5-Fold CV Setup
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "params": {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": ["balanced", None],
        }
    },
    "Support Vector Machine": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "gamma": ["scale", "auto", 0.01, 0.05, 0.1],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced", None],
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", None],
        }
    }
}
```

---

## 7. Hyperparameter Tuning Results

Through exhaustive Stratified Grid Search over the candidate parameter space, the optimal hyperparameter configurations were identified:

| Model Architecture | Optimal Hyperparameters Identified | Tuning Duration |
|---|---|---|
| **Logistic Regression** | `{'C': 1.0, 'class_weight': None, 'penalty': 'l2', 'solver': 'lbfgs'}` | 7.74 s |
| **Support Vector Machine** | `{'C': 5.0, 'class_weight': None, 'gamma': 0.01, 'kernel': 'rbf'}` | 6.18 s |
| **Random Forest** | `{'class_weight': 'balanced', 'max_depth': 12, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}` | 221.04 s |

---

## 8. Cross-Validation Benchmark Results

The table below summarizes the 5-Fold Stratified Cross-Validation performance metrics (Mean $\pm$ Standard Deviation) across the 455 training samples:

| Model Candidate | CV Accuracy | CV Recall (Malignant) | CV Precision | CV F1-Score | CV ROC-AUC |
|---|---|---|---|---|---|
| **Baseline (Majority Class)** | $0.6264 \pm 0.0022$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.5000 \pm 0.0000$ |
| **Untuned Logistic Reg.** | $0.9714 \pm 0.0152$ | $0.9471 \pm 0.0432$ | $0.9758 \pm 0.0235$ | $0.9608 \pm 0.0210$ | $0.9942 \pm 0.0051$ |
| **Logistic Regression (Tuned)** | **$0.9736 \pm 0.0149$** | **$0.9529 \pm 0.0399$** | $0.9771 \pm 0.0280$ | $0.9640 \pm 0.0207$ | **$0.9958 \pm 0.0047$** |
| **Support Vector Machine (RBF)** | **$0.9758 \pm 0.0128$** | $0.9471 \pm 0.0432$ | **$0.9889 \pm 0.0222$** | **$0.9666 \pm 0.0185$** | **$0.9960 \pm 0.0050$** |
| **Random Forest Classifier** | $0.9582 \pm 0.0162$ | $0.9412 \pm 0.0263$ | $0.9469 \pm 0.0219$ | $0.9439 \pm 0.0218$ | $0.9913 \pm 0.0048$ |

---

## 9. Validation Set Evaluation & Error Analysis

Evaluating the best model checkpoints on the independent validation holdout set ($N=57$, Benign $= 36$, Malignant $= 21$):

| Model Name | Val Accuracy | Malignant Recall | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 94.74% | 85.71% | **100.00%** | **100.00%** | 0.9231 | 0.9934 | 0.9902 |
| **Support Vector Machine** | **96.49%** | 90.48% | **100.00%** | **100.00%** | 0.9500 | 0.9934 | 0.9901 |
| **Random Forest** | **96.49%** | **95.24%** | 97.22% | 95.24% | **0.9524** | **0.9947** | **0.9918** |

### Confusion Matrix Breakdown ($N=57$)

```
Logistic Regression:           SVM (RBF Kernel):              Random Forest:
┌───────────┬───────────┐      ┌───────────┬───────────┐      ┌───────────┬───────────┐
│ TN = 36   │ FP = 0    │      │ TN = 36   │ FP = 0    │      │ TN = 35   │ FP = 1    │
│ (100.0%)  │ (0.0%)    │      │ (100.0%)  │ (0.0%)    │      │ (97.2%)   │ (2.8%)    │
├───────────┼───────────┤      ├───────────┼───────────┤      ├───────────┼───────────┤
│ FN = 3    │ TP = 18   │      │ FN = 2    │ TP = 19   │      │ FN = 1    │ TP = 20   │
│ (14.3%)   │ (85.7%)   │      │ (9.5%)    │ (90.5%)   │      │ (4.8%)    │ (95.2%)   │
└───────────┴───────────┘      └───────────┴───────────┘      └───────────┴───────────┘
```

- **Clinical Insight:** In oncology screening, False Negatives (FN) carry a severe penalty (a malignant tumor is missed). **Random Forest achieved the lowest False Negative rate ($\text{FN} = 1$, Recall $= 95.24\%$)**, while **SVM and Logistic Regression achieved perfect Specificity ($100\%$, $\text{FP} = 0$)**.

---

## 10. Visualizations & Analytical Charts

All generated high-resolution figures are preserved in `outputs/`:

1. **Figure 1 — Cross-Validation Benchmark:** `outputs/experiment5_cv_comparison.png`  
   *Compares Accuracy, Recall, Precision, F1, and ROC-AUC with standard deviation error bars across all candidate architectures.*
2. **Figure 2 — Validation Confusion Matrices:** `outputs/experiment5_confusion_matrices.png`  
   *Side-by-side display of true vs predicted diagnostic outcomes with percentages.*
3. **Figure 3 — ROC Curves:** `outputs/experiment5_roc_curves.png`  
   *Displays the True Positive Rate against False Positive Rate for all models against the no-skill baseline.*
4. **Figure 4 — Precision-Recall Curves:** `outputs/experiment5_precision_recall_curves.png`  
   *Shows precision retention across recall levels with prevalence baseline line ($36.8\%$).*
5. **Figure 5 — Interpretability & Feature Attribution:** `outputs/experiment5_feature_importance.png`  
   *Compares Random Forest Gini importance with Logistic Regression standardized feature coefficients.*
6. **Figure 6 — Learning Curves:** `outputs/experiment5_learning_curves.png`  
   *Plots training vs validation score curves over increasing sample counts to confirm absence of severe overfitting.*
7. **Figure 7 — SVM Hyperparameter Sensitivity Surface:** `outputs/experiment5_hyperparameter_heatmaps.png`  
   *Visualizes the 2D tuning grid of regularization $C$ versus kernel coefficient $\gamma$.*

---

## 11. Saved Model Artifacts

The following model artifacts and metadata files are serialized in `models/`:

- [`models/logistic_regression_best.pkl`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/models/logistic_regression_best.pkl) — Best L2-regularized Logistic Regression ($1.81\text{ KB}$)
- [`models/svm_rbf_best.pkl`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/models/svm_rbf_best.pkl) — Best RBF-kernel Support Vector Classifier ($18.37\text{ KB}$)
- [`models/random_forest_best.pkl`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/models/random_forest_best.pkl) — Best Random Forest Classifier ($165.45\text{ KB}$)
- [`models/dummy_baseline.pkl`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/models/dummy_baseline.pkl) — Zero-Rule Dummy Baseline Model ($1.35\text{ KB}$)
- [`models/model_training_summary.json`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/models/model_training_summary.json) — Complete machine-readable performance metadata
- [`outputs/experiment5_training_summary.txt`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/outputs/experiment5_training_summary.txt) — Text log of CV metrics and validation comparisons

---

## 12. Key Observations & Findings

1. **Substantial Baseline Superiority:** All three tuned candidate models achieved $>95.8\%$ CV accuracy and $>0.991$ ROC-AUC, outperforming the majority class baseline ($62.6\%$ accuracy, $0.00$ recall) by a massive margin.
2. **SVM (RBF Kernel) Overall CV Excellence:** SVM with $C=5.0, \gamma=0.01$ achieved the highest cross-validation score ($0.9960 \text{ ROC-AUC}, 97.58\% \text{ Accuracy}, 98.89\% \text{ Precision}$), proving that non-linear kernel projection effectively separates subtle nuclear irregularities.
3. **Random Forest Superiority in Sensitivity:** Random Forest with `class_weight='balanced'` achieved the highest Malignant Recall on the validation holdout set ($95.24\%$, $\text{FN}=1$), which is the most critical metric for preventing false cancer clearances.
4. **Consistency with EDA Insights:** Top features identified during Experiment 4 (`concave_points_worst`, `radius_worst`, `perimeter_worst`, `area_worst`, and `concavity_mean`) emerged as the leading drivers in both Random Forest Gini importance and Logistic Regression coefficient weights.
5. **Zero Overfitting on Scaled Inputs:** Learning curves show tight convergence between training and cross-validation ROC-AUC curves ($>0.99$), confirming robust generalization without memorization.

---

## 13. Result

Three candidate machine learning models (Logistic Regression, SVM RBF, and Random Forest) were successfully implemented, trained, hyperparameter-tuned via Stratified 5-Fold Cross-Validation, and evaluated on the Breast Cancer Wisconsin dataset. All trained model checkpoints, training logs, metadata manifests, and analytical charts were generated and saved to their respective directories.

---

## 14. Conclusion

Experiment 5 demonstrates that both kernelized margin classifiers (SVM RBF) and ensemble tree methods (Random Forest) provide near-perfect discriminative capability on cell nuclear features. The hyperparameter-tuned models are saved as serialized artifacts ready for comprehensive evaluation, calibration, threshold tuning, and error triage in Experiment 6.
