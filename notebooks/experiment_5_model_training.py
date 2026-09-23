"""
CSE4001: Applications of Machine Learning in Industry
Health-Project 3: Breast Cancer Detection System
Experiment 5: Model Training, Hyperparameter Tuning & Cross-Validation

Tasks:
1. Implement baseline model (Dummy & Untuned classifiers).
2. Train candidate algorithms (Logistic Regression, Support Vector Machine, Random Forest).
3. Tune hyperparameters using Stratified GridSearchCV (k=5).
4. Run k-fold cross-validation with multi-metric tracking.
5. Evaluate on validation holdout set and analyze decision boundaries.
6. Generate publication-quality visualization figures.
7. Save best checkpoint model artifacts (.pkl) and training logs.
"""

import os
import sys
import json
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

def run_experiment_5():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Styling for plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'

    COLOR_PALETTE = {
        "Navy": "#1A365D",
        "Blue": "#2B6CB0",
        "Teal": "#2C7A7B",
        "Coral": "#DD6B20",
        "Red": "#E53E3E",
        "Green": "#38A169",
        "Grey": "#718096",
        "LightGrey": "#EDF2F7",
    }

    # ---------------------------------------------------------
    # 1. Load Processed Data
    # ---------------------------------------------------------
    print("=" * 80)
    print("EXPERIMENT 5: MODEL TRAINING, TUNING & CROSS-VALIDATION PIPELINE")
    print("=" * 80)

    print("\n[Step 1] Loading preprocessed datasets from data/processed/...")
    X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv")).values.ravel()

    X_val_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_val_scaled.csv"))
    y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv")).values.ravel()

    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).values.ravel()

    feature_names = list(X_train_scaled.columns)

    print(f" -> X_train_scaled Shape: {X_train_scaled.shape} | y_train: {y_train.shape} (Benign: {sum(y_train==0)}, Malignant: {sum(y_train==1)})")
    print(f" -> X_val_scaled   Shape: {X_val_scaled.shape}   | y_val:   {y_val.shape}   (Benign: {sum(y_val==0)}, Malignant: {sum(y_val==1)})")
    print(f" -> X_test_scaled  Shape: {X_test_scaled.shape}  | y_test:  {y_test.shape}  (Benign: {sum(y_test==0)}, Malignant: {sum(y_test==1)})")

    # ---------------------------------------------------------
    # 2. Baseline Models (Dummy & Untuned)
    # ---------------------------------------------------------
    print("\n[Step 2] Training Baseline Classifiers...")

    # Dummy Majority Class Classifier
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_majority.fit(X_train_scaled, y_train)
    dummy_val_preds = dummy_majority.predict(X_val_scaled)
    dummy_acc = accuracy_score(y_val, dummy_val_preds)
    dummy_rec = recall_score(y_val, dummy_val_preds, zero_division=0)

    print(f" -> Baseline (Majority Class): Val Accuracy = {dummy_acc:.4f}, Val Malignant Recall = {dummy_rec:.4f}")

    # Untuned Logistic Regression Baseline
    lr_untuned = LogisticRegression(random_state=42)
    lr_untuned.fit(X_train_scaled, y_train)
    lr_untuned_val_preds = lr_untuned.predict(X_val_scaled)
    lr_untuned_val_proba = lr_untuned.predict_proba(X_val_scaled)[:, 1]
    print(f" -> Baseline (Untuned Logistic Regression): Val Accuracy = {accuracy_score(y_val, lr_untuned_val_preds):.4f}, "
          f"Val Recall = {recall_score(y_val, lr_untuned_val_preds):.4f}, Val ROC-AUC = {roc_auc_score(y_val, lr_untuned_val_proba):.4f}")

    # ---------------------------------------------------------
    # 3. Hyperparameter Grids & Stratified GridSearchCV
    # ---------------------------------------------------------
    print("\n[Step 3] Defining Candidate Model Grids & Running Stratified GridSearchCV (k=5)...")

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

    best_models = {}
    cv_results_summary = {}
    training_logs = []

    for name, config in param_grids.items():
        print(f"\n--- Tuning {name} ---")
        start_time = time.time()
        
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv_strategy,
            scoring={
                "accuracy": "accuracy",
                "recall": "recall",
                "precision": "precision",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            refit="roc_auc",
            n_jobs=1,
            return_train_score=True,
        )
        
        grid.fit(X_train_scaled, y_train)
        elapsed = time.time() - start_time
        
        best_est = grid.best_estimator_
        best_models[name] = best_est
        best_params = grid.best_params_
        best_idx = grid.best_index_
        
        cv_mean_acc = grid.cv_results_["mean_test_accuracy"][best_idx]
        cv_std_acc = grid.cv_results_["std_test_accuracy"][best_idx]
        cv_mean_rec = grid.cv_results_["mean_test_recall"][best_idx]
        cv_std_rec = grid.cv_results_["std_test_recall"][best_idx]
        cv_mean_prec = grid.cv_results_["mean_test_precision"][best_idx]
        cv_std_prec = grid.cv_results_["std_test_precision"][best_idx]
        cv_mean_f1 = grid.cv_results_["mean_test_f1"][best_idx]
        cv_std_f1 = grid.cv_results_["std_test_f1"][best_idx]
        cv_mean_auc = grid.cv_results_["mean_test_roc_auc"][best_idx]
        cv_std_auc = grid.cv_results_["std_test_roc_auc"][best_idx]
        
        cv_results_summary[name] = {
            "best_params": best_params,
            "cv_accuracy_mean": cv_mean_acc,
            "cv_accuracy_std": cv_std_acc,
            "cv_recall_mean": cv_mean_rec,
            "cv_recall_std": cv_std_rec,
            "cv_precision_mean": cv_mean_prec,
            "cv_precision_std": cv_std_prec,
            "cv_f1_mean": cv_mean_f1,
            "cv_f1_std": cv_std_f1,
            "cv_roc_auc_mean": cv_mean_auc,
            "cv_roc_auc_std": cv_std_auc,
            "tuning_time_sec": round(elapsed, 2),
            "grid_search_obj": grid,
        }
        
        log_entry = (
            f"[{name}] Best Parameters: {best_params}\n"
            f"  - 5-Fold CV ROC-AUC:   {cv_mean_auc:.4f} (+/- {cv_std_auc:.4f})\n"
            f"  - 5-Fold CV Recall:    {cv_mean_rec:.4f} (+/- {cv_std_rec:.4f})\n"
            f"  - 5-Fold CV Precision: {cv_mean_prec:.4f} (+/- {cv_std_prec:.4f})\n"
            f"  - 5-Fold CV F1-Score:  {cv_mean_f1:.4f} (+/- {cv_std_f1:.4f})\n"
            f"  - 5-Fold CV Accuracy:  {cv_mean_acc:.4f} (+/- {cv_std_acc:.4f})\n"
            f"  - Tuning Duration:     {elapsed:.2f} seconds\n"
        )
        print(log_entry)
        training_logs.append(log_entry)

    # ---------------------------------------------------------
    # 4. Detailed Validation Set Evaluation
    # ---------------------------------------------------------
    print("\n[Step 4] Evaluating Best Checkpoint Models on Validation Holdout Set (N=57)...")

    val_metrics = {}

    for name, model in best_models.items():
        val_preds = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        acc = accuracy_score(y_val, val_preds)
        bal_acc = balanced_accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds)
        rec = recall_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_proba)
        ap = average_precision_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
        cm = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        
        val_metrics[name] = {
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision": prec,
            "Recall_Sensitivity": rec,
            "Specificity": spec,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "PR_AUC": ap,
            "Brier_Score": brier,
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "val_preds": val_preds,
            "val_proba": val_proba,
        }
        
        print(f"\n{name} Validation Performance:")
        print(f"  Accuracy: {acc*100:.2f}% | Recall (Malignant): {rec*100:.2f}% | Specificity: {spec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FN={fn}, TN={tn}, FP={fp}")

    # ---------------------------------------------------------
    # 5. Model Checkpoint Serialization (.pkl)
    # ---------------------------------------------------------
    print("\n[Step 5] Saving Best Model Checkpoint Artifacts to models/...")

    artifact_paths = {
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression_best.pkl"),
        "Support Vector Machine": os.path.join(MODELS_DIR, "svm_rbf_best.pkl"),
        "Random Forest": os.path.join(MODELS_DIR, "random_forest_best.pkl"),
    }

    for name, path in artifact_paths.items():
        joblib.dump(best_models[name], path)
        file_size_kb = os.path.getsize(path) / 1024
        print(f" -> Saved {name} checkpoint: {os.path.basename(path)} ({file_size_kb:.2f} KB)")

    # Save dummy baseline
    dummy_path = os.path.join(MODELS_DIR, "dummy_baseline.pkl")
    joblib.dump(dummy_majority, dummy_path)

    # Save JSON metadata summary
    json_summary = {
        "experiment": "Experiment 5: Model Training",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_splits": {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "num_features": int(len(feature_names)),
        },
        "cv_results": {
            name: {
                "best_params": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) for k, v in res["best_params"].items()},
                "cv_roc_auc": round(float(res["cv_roc_auc_mean"]), 4),
                "cv_recall": round(float(res["cv_recall_mean"]), 4),
                "cv_f1": round(float(res["cv_f1_mean"]), 4),
                "cv_accuracy": round(float(res["cv_accuracy_mean"]), 4),
            }
            for name, res in cv_results_summary.items()
        },
        "validation_results": {
            name: {
                "accuracy": round(float(m["Accuracy"]), 4),
                "recall": round(float(m["Recall_Sensitivity"]), 4),
                "specificity": round(float(m["Specificity"]), 4),
                "precision": round(float(m["Precision"]), 4),
                "f1_score": round(float(m["F1_Score"]), 4),
                "roc_auc": round(float(m["ROC_AUC"]), 4),
                "confusion_matrix": m["Confusion_Matrix"],
            }
            for name, m in val_metrics.items()
        }
    }

    summary_json_path = os.path.join(MODELS_DIR, "model_training_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(json_summary, f, indent=4)
    print(f" -> Saved Model Training Metadata: {os.path.basename(summary_json_path)}")

    # Write complete text log
    training_log_path = os.path.join(OUTPUTS_DIR, "experiment5_training_summary.txt")
    with open(training_log_path, "w") as f:
        f.write("CSE4001 EXPERIMENT 5: MODEL TRAINING & HYPERPARAMETER TUNING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for log in training_logs:
            f.write(log + "\n")
        f.write("=" * 80 + "\n")
        f.write("VALIDATION SET (N=57) METRICS COMPARISON TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model Name':<25} | {'Acc (%)':<8} | {'Recall':<8} | {'Spec':<8} | {'Prec':<8} | {'F1':<8} | {'ROC-AUC':<8}\n")
        f.write("-" * 80 + "\n")
        for name, m in val_metrics.items():
            f.write(f"{name:<25} | {m['Accuracy']*100:<8.2f} | {m['Recall_Sensitivity']*100:<8.2f} | {m['Specificity']*100:<8.2f} | {m['Precision']*100:<8.2f} | {m['F1_Score']:<8.4f} | {m['ROC_AUC']:<8.4f}\n")
        f.write("=" * 80 + "\n")
    print(f" -> Saved Text Training Summary: {os.path.basename(training_log_path)}")

    # ---------------------------------------------------------
    # 6. Generate Publication-Quality Visualizations
    # ---------------------------------------------------------
    print("\n[Step 6] Generating Publication-Quality Figures in outputs/...")

    # Figure 1: 5-Fold Cross-Validation Performance Comparison
    metrics_to_plot = ["cv_accuracy_mean", "cv_recall_mean", "cv_precision_mean", "cv_f1_mean", "cv_roc_auc_mean"]
    metrics_err = ["cv_accuracy_std", "cv_recall_std", "cv_precision_std", "cv_f1_std", "cv_roc_auc_std"]
    metric_labels = ["Accuracy", "Malignant Recall", "Precision", "F1-Score", "ROC-AUC"]

    model_names = list(cv_results_summary.keys())
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors_list = [COLOR_PALETTE["Navy"], COLOR_PALETTE["Teal"], COLOR_PALETTE["Coral"]]

    for i, (m_name, color) in enumerate(zip(model_names, colors_list)):
        means = [cv_results_summary[m_name][m] for m in metrics_to_plot]
        stds = [cv_results_summary[m_name][s] for s in metrics_err]
        rects = ax.bar(x + i*width - width/2, means, width, yerr=stds, capsize=4, label=m_name, color=color, alpha=0.9, edgecolor='black', linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_title("Figure 1: 5-Fold Stratified Cross-Validation Benchmark Across Candidate Models", pad=15)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metric_labels, fontweight='bold', fontsize=10)
    ax.set_ylabel("Cross-Validation Score (0.0 to 1.0)", fontweight='bold')
    ax.set_ylim(0.85, 1.03)
    ax.axhline(0.95, color=COLOR_PALETTE["Grey"], linestyle='--', linewidth=0.8, alpha=0.7, label='95% Target Line')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right')
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUTS_DIR, "experiment5_cv_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" -> Saved Fig 1: {os.path.basename(fig1_path)}")

    # Figure 2: Confusion Matrices (Validation Set)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (m_name, m_data) in zip(axes, val_metrics.items()):
        cm = confusion_matrix(y_val, m_data["val_preds"])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        annot_text = np.empty_like(cm).astype(str)
        for r in range(2):
            for c in range(2):
                count = cm[r, c]
                pct = cm_norm[r, c] * 100
                annot_text[r, c] = f"{count}\n({pct:.1f}%)"
                
        sns.heatmap(cm, annot=annot_text, fmt='', cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign (0)", "Malignant (1)"],
                    yticklabels=["Benign (0)", "Malignant (1)"],
                    annot_kws={"fontsize": 11, "fontweight": "bold"},
                    linewidths=1.5, linecolor='white')
        
        rec_val = m_data["Recall_Sensitivity"] * 100
        acc_val = m_data["Accuracy"] * 100
        ax.set_title(f"{m_name}\nAcc: {acc_val:.1f}% | Recall: {rec_val:.1f}%", pad=8)
        ax.set_xlabel("Predicted Diagnosis", fontweight='bold')
        ax.set_ylabel("True Diagnosis", fontweight='bold')

    plt.suptitle("Figure 2: Confusion Matrices on Validation Holdout Set (N=57, Benign=36, Malignant=21)", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUTS_DIR, "experiment5_confusion_matrices.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Fig 2: {os.path.basename(fig2_path)}")

    # Figure 3: Validation ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))

    for m_name, color in zip(model_names, colors_list):
        proba = val_metrics[m_name]["val_proba"]
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_val = val_metrics[m_name]["ROC_AUC"]
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.4f})", color=color, linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='No-Skill Classifier (AUC = 0.5000)')

    ax.set_title("Figure 3: Receiver Operating Characteristic (ROC) Curves on Validation Set", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=9.5)
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUTS_DIR, "experiment5_roc_curves.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" -> Saved Fig 3: {os.path.basename(fig3_path)}")

    # Figure 4: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    malignant_prevalence = sum(y_val == 1) / len(y_val)

    for m_name, color in zip(model_names, colors_list):
        proba = val_metrics[m_name]["val_proba"]
        prec, rec, _ = precision_recall_curve(y_val, proba)
        pr_auc = val_metrics[m_name]["PR_AUC"]
        ax.plot(rec, prec, label=f"{m_name} (PR-AUC / AP = {pr_auc:.4f})", color=color, linewidth=2.5)

    ax.axhline(malignant_prevalence, color='k', linestyle='--', linewidth=1.2,
               label=f'Prevalence Baseline ({malignant_prevalence*100:.1f}%)')

    ax.set_title("Figure 4: Precision-Recall (PR) Curves on Validation Holdout Set", pad=12)
    ax.set_xlabel("Recall (Malignant Sensitivity)", fontweight='bold')
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower left', fontsize=9.5)
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUTS_DIR, "experiment5_precision_recall_curves.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" -> Saved Fig 4: {os.path.basename(fig4_path)}")

    # Figure 5: Feature Importance & Coefficient Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    rf_best = best_models["Random Forest"]
    importances = rf_best.feature_importances_
    rf_feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True).tail(12)

    ax1.barh(rf_feat_df["Feature"], rf_feat_df["Importance"], color=COLOR_PALETTE["Coral"], edgecolor='black', linewidth=0.7)
    ax1.set_title("Random Forest: Top 12 Feature Importances (MDI)", pad=10)
    ax1.set_xlabel("Mean Decrease in Impurity (Gini)", fontweight='bold')

    lr_best = best_models["Logistic Regression"]
    coefs = lr_best.coef_[0]
    lr_feat_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs, "AbsCoef": np.abs(coefs)}).sort_values("AbsCoef", ascending=True).tail(12)

    bar_colors = [COLOR_PALETTE["Red"] if c > 0 else COLOR_PALETTE["Blue"] for c in lr_feat_df["Coefficient"]]
    ax2.barh(lr_feat_df["Feature"], lr_feat_df["Coefficient"], color=bar_colors, edgecolor='black', linewidth=0.7)
    ax2.set_title("Logistic Regression: Top 12 Standardized Feature Weights", pad=10)
    ax2.set_xlabel("Model Coefficient (Sign Indicates Directionality)", fontweight='bold')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)

    plt.suptitle("Figure 5: Model Interpretability: Random Forest Gini Importance vs. Logistic Regression Weights", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig5_path = os.path.join(OUTPUTS_DIR, "experiment5_feature_importance.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Fig 5: {os.path.basename(fig5_path)}")

    # Figure 6: Learning Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    train_sizes = np.linspace(0.1, 1.0, 6)

    for ax, (m_name, color) in zip(axes, zip(model_names, colors_list)):
        lc_res = learning_curve(
            best_models[m_name],
            X_train_scaled,
            y_train,
            train_sizes=train_sizes,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42
        )
        train_sizes_abs = lc_res[0]
        train_scores = lc_res[1]
        test_scores = lc_res[2]
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color=COLOR_PALETTE["Red"], label='Training ROC-AUC', linewidth=2)
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color=COLOR_PALETTE["Red"])
        
        ax.plot(train_sizes_abs, test_mean, 's-', color=color, label='5-Fold CV ROC-AUC', linewidth=2)
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color=color)
        
        ax.set_title(f"{m_name}", pad=8)
        ax.set_xlabel("Number of Training Samples", fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel("ROC-AUC Score", fontweight='bold')
        ax.set_ylim(0.88, 1.02)
        ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    plt.suptitle("Figure 6: Learning Curves (Sample Size vs ROC-AUC) Showing Generalization Stability", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig6_path = os.path.join(OUTPUTS_DIR, "experiment5_learning_curves.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Fig 6: {os.path.basename(fig6_path)}")

    # Figure 7: Hyperparameter Tuning Heatmap
    svm_grid = cv_results_summary["Support Vector Machine"]["grid_search_obj"]
    svm_cv_df = pd.DataFrame(svm_grid.cv_results_)
    rbf_df = svm_cv_df[(svm_cv_df["param_kernel"] == "rbf") & (svm_cv_df["param_class_weight"] == "balanced")]

    if len(rbf_df) > 0 and "param_gamma" in rbf_df.columns:
        pivot_table = rbf_df.pivot(index="param_C", columns="param_gamma", values="mean_test_roc_auc")
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean 5-Fold ROC-AUC'}, ax=ax,
                    linewidths=1.0, linecolor='white')
        ax.set_title("Figure 7: SVM Hyperparameter Sensitivity Surface (C vs. Gamma under RBF Kernel)", pad=12)
        ax.set_xlabel("Kernel Coefficient (Gamma)", fontweight='bold')
        ax.set_ylabel("Regularization Parameter (C)", fontweight='bold')
        plt.tight_layout()
        fig7_path = os.path.join(OUTPUTS_DIR, "experiment5_hyperparameter_heatmaps.png")
        plt.savefig(fig7_path, dpi=300)
        plt.close()
        print(f" -> Saved Fig 7: {os.path.basename(fig7_path)}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Model Artifacts saved in: {MODELS_DIR}")
    print(f"Figures and Logs saved in: {OUTPUTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_5()
