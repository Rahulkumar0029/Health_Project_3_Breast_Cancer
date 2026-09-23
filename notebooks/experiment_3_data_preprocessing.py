import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def run_experiment_3():
    print("=" * 70)
    print("EXPERIMENT 3: DATA PRE-PROCESSING PIPELINE")
    print("Health-Project 3: Breast Cancer Detection")
    print("=" * 70)

    # 1. Paths Setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    outputs_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    report_dir = os.path.join(base_dir, "report")

    for d in [processed_dir, outputs_dir, models_dir, pdf_dir, report_dir]:
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(data_dir, "breast_cancer.csv")
    wdbc_path = os.path.join(data_dir, "wdbc.data")

    # 2. Step 1: Load Dataset
    if os.path.exists(csv_path):
        print(f"[Step 1] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    elif os.path.exists(wdbc_path):
        print(f"[Step 1] Loading dataset from UCI raw file: {wdbc_path}")
        cols = [
            "id", "diagnosis",
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]
        df = pd.read_csv(wdbc_path, header=None, names=cols)
    else:
        raise FileNotFoundError("Could not locate breast_cancer.csv or wdbc.data in data directory.")

    initial_shape = df.shape
    print(f"Raw Dataset Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")
    print(df.head())

    # 3. Step 2 & 5: Clean Columns (Drop Unnamed and ID)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
    dropped_id = False
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        dropped_id = True
        print("[Step 2] Successfully dropped metadata column 'id'.")

    # 4. Step 3: Missing Value Audit and Imputation
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"[Step 3] Total Missing Values in Dataset: {total_nulls}")
    if total_nulls > 0:
        print("Missing values per column:\n", null_counts[null_counts > 0])
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        print("Handled missing values using median imputation.")
    else:
        print("Verified: 0 missing values detected. Dataset is complete.")

    # 5. Step 4: Duplicate Records Check
    dup_count = df.duplicated().sum()
    print(f"[Step 4] Duplicate Records: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count} duplicate records.")
    else:
        print("Verified: 0 duplicate rows detected.")

    # 6. Step 6: Encode Target Variable (M -> 1, B -> 0)
    print("\n[Step 5] Encoding Diagnosis Target:")
    diagnosis_raw_counts = df["diagnosis"].value_counts().to_dict()
    print(f"Raw categorical counts: {diagnosis_raw_counts}")
    
    # Map M->1 (Malignant), B->0 (Benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0, 1: 1, 0: 0})
    df["diagnosis"] = df["diagnosis"].astype(int)
    
    encoded_counts = df["diagnosis"].value_counts().to_dict()
    print(f"Encoded Target Distribution (0=Benign, 1=Malignant): {encoded_counts}")

    # 7. Step 7: Separate Features and Target
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    feature_names = list(X.columns)
    print(f"\n[Step 6] Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Number of numerical diagnostic features: {len(feature_names)}")

    # 8. Step 8: Class Distribution Analysis
    n_total = len(y)
    n_benign = int((y == 0).sum())
    n_malignant = int((y == 1).sum())
    pct_benign = (n_benign / n_total) * 100
    pct_malignant = (n_malignant / n_total) * 100
    imbalance_ratio = n_benign / n_malignant
    # Positive weight for clinical loss weighting
    pos_weight = n_benign / n_malignant

    print(f"\n[Step 7] Class Distribution Summary:")
    print(f" - Benign (Class 0): {n_benign} samples ({pct_benign:.2f}%)")
    print(f" - Malignant (Class 1): {n_malignant} samples ({pct_malignant:.2f}%)")
    print(f" - Imbalance Ratio (Benign:Malignant): {imbalance_ratio:.2f}:1")
    print(f" - Suggested Clinical Loss Weight for Malignant (pos_weight): {pos_weight:.3f}")

    # 9. Step 9: Stratified Train-Validation-Test Split (80% / 10% / 10%)
    # First split: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Second split: Temp -> 50% Val, 50% Test (each is 10% of total dataset)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\n[Step 8] Stratified Dataset Split Completed:")
    print(f" - Training Set:   {X_train.shape[0]} samples ({X_train.shape[0]/n_total*100:.1f}%) | Benign: {(y_train==0).sum()}, Malignant: {(y_train==1).sum()}")
    print(f" - Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/n_total*100:.1f}%) | Benign: {(y_val==0).sum()}, Malignant: {(y_val==1).sum()}")
    print(f" - Test Set:       {X_test.shape[0]} samples ({X_test.shape[0]/n_total*100:.1f}%) | Benign: {(y_test==0).sum()}, Malignant: {(y_test==1).sum()}")

    # 10. Step 10: Feature Standardization (StandardScaler)
    # FIT ONLY on Training data to prevent Data Leakage!
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_val_scaled_arr = scaler.transform(X_val)
    X_test_scaled_arr = scaler.transform(X_test)

    # Convert back to clean DataFrames with feature names & indices
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=feature_names, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled_arr, columns=feature_names, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=feature_names, index=X_test.index)

    print("\n[Step 9] Feature Scaling Completed via StandardScaler:")
    print(" - Fitted solely on Training data (mean & std computed from X_train only).")
    print(f" - X_train_scaled mean ~ {X_train_scaled.mean().mean():.4f}, std ~ {X_train_scaled.std().mean():.4f}")
    print(f" - Scaled Shapes -> Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # 11. Step 11: Save Preprocessing Scaler and Datasets
    scaler_out_path = os.path.join(outputs_dir, "breast_cancer_scaler.pkl")
    scaler_model_path = os.path.join(models_dir, "breast_cancer_scaler.pkl")
    joblib.dump(scaler, scaler_out_path)
    joblib.dump(scaler, scaler_model_path)
    print(f"\n[Step 10] Scaler pipeline successfully saved to:\n - {scaler_out_path}\n - {scaler_model_path}")

    # Save Processed CSVs
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(processed_dir, "X_val_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Combined full splits (features + target)
    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val_scaled, y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    train_df.to_csv(os.path.join(processed_dir, "train_preprocessed.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_preprocessed.csv"), index=False)

    # Master preprocessed dataset (unsplit, for EDA reference)
    master_scaled_arr = scaler.transform(X)
    master_df = pd.DataFrame(master_scaled_arr, columns=feature_names)
    master_df["diagnosis"] = y.values
    master_df.to_csv(os.path.join(processed_dir, "breast_cancer_preprocessed.csv"), index=False)
    print(f"Processed datasets successfully saved to {processed_dir}")

    # 12. Generate Visualizations for Report
    # Chart 1: Class Distribution & Split Consistency
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Subplot A: Overall Class Pie/Donut
    colors = ['#2b5c8f', '#d9534f']
    axes[0].pie(
        [n_benign, n_malignant],
        labels=[f'Benign (0)\n{n_benign} ({pct_benign:.1f}%)', f'Malignant (1)\n{n_malignant} ({pct_malignant:.1f}%)'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    axes[0].set_title('Overall Target Class Distribution\n(569 Biopsy Samples)', fontsize=13, weight='bold', pad=15)

    # Subplot B: Split Proportions Bar Chart
    split_names = ['Train (80%)', 'Validation (10%)', 'Test (10%)']
    b_counts = [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()]
    m_counts = [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()]

    x_indices = np.arange(len(split_names))
    width = 0.35

    rects1 = axes[1].bar(x_indices - width/2, b_counts, width, label='Benign (0)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    rects2 = axes[1].bar(x_indices + width/2, m_counts, width, label='Malignant (1)', color='#d9534f', edgecolor='black', alpha=0.9)

    axes[1].set_ylabel('Number of Samples', fontsize=11, weight='bold')
    axes[1].set_title('Stratified Split Class Balance', fontsize=13, weight='bold', pad=15)
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(split_names, fontsize=11, weight='bold')
    axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        axes[1].annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    dist_chart_path = os.path.join(outputs_dir, "experiment3_class_distribution.png")
    plt.savefig(dist_chart_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_chart_path}")

    # Chart 2: Feature Scaling Comparison (Before vs After StandardScaler)
    sample_features = ["radius_mean", "texture_mean", "area_mean", "smoothness_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes = axes.flatten()

    for idx, feat in enumerate(sample_features):
        ax = axes[idx]
        raw_vals = X_train[feat]
        scaled_vals = X_train_scaled[feat]

        ax2 = ax.twinx()
        sns.kdeplot(raw_vals, ax=ax, color='#2b5c8f', fill=True, alpha=0.3, label='Raw Feature (Left Axis)', linewidth=2)
        sns.kdeplot(scaled_vals, ax=ax2, color='#e67e22', fill=True, alpha=0.3, label='Standardized (Right Axis)', linewidth=2)

        ax.set_title(f'Scaling Effect: {feat}', fontsize=12, weight='bold')
        ax.set_xlabel('Raw Value Magnitude', fontsize=10, color='#2b5c8f', weight='bold')
        ax2.set_xlabel('Standardized Value (Z-Score)', fontsize=10, color='#e67e22', weight='bold')
        ax.set_ylabel('Raw Density', color='#2b5c8f')
        ax2.set_ylabel('Standardized Density', color='#e67e22')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Comparison of Feature Distributions Before and After StandardScaler', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    scaling_chart_path = os.path.join(outputs_dir, "experiment3_scaling_comparison.png")
    plt.savefig(scaling_chart_path, dpi=300)
    plt.close()
    print(f"Saved: {scaling_chart_path}")

    # 13. Write Preprocessing Text Summary
    summary_text_path = os.path.join(outputs_dir, "experiment3_preprocessing_summary.txt")
    with open(summary_text_path, "w") as f:
        f.write("EXPERIMENT 3: DATA PRE-PROCESSING SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"1. Dataset: Breast Cancer Wisconsin (Diagnostic)\n")
        f.write(f"   - Initial Shape: {initial_shape}\n")
        f.write(f"   - Cleaned Feature Shape: {X.shape}\n")
        f.write(f"   - Total Samples: {n_total}\n")
        f.write(f"   - Diagnostic Features: {len(feature_names)}\n\n")
        f.write(f"2. Data Hygiene:\n")
        f.write(f"   - Missing Values: {total_nulls}\n")
        f.write(f"   - Duplicate Records: {dup_count}\n")
        f.write(f"   - Metadata Dropped: 'id', unnamed columns\n\n")
        f.write(f"3. Target Encoding:\n")
        f.write(f"   - Mapping: 'M' -> 1 (Malignant, Positive Class), 'B' -> 0 (Benign, Negative Class)\n")
        f.write(f"   - Class Distribution: Benign = {n_benign} ({pct_benign:.2f}%), Malignant = {n_malignant} ({pct_malignant:.2f}%)\n")
        f.write(f"   - Class Imbalance Ratio: {imbalance_ratio:.2f}:1 (Benign : Malignant)\n")
        f.write(f"   - Recommended Class Weight (pos_weight): {pos_weight:.3f}\n\n")
        f.write(f"4. Stratified Dataset Partitions:\n")
        f.write(f"   - Training Set (80%):   {X_train.shape[0]} samples (Benign: {(y_train==0).sum()}, Malignant: {(y_train==1).sum()})\n")
        f.write(f"   - Validation Set (10%): {X_val.shape[0]} samples (Benign: {(y_val==0).sum()}, Malignant: {(y_val==1).sum()})\n")
        f.write(f"   - Test Set (10%):       {X_test.shape[0]} samples (Benign: {(y_test==0).sum()}, Malignant: {(y_test==1).sum()})\n\n")
        f.write(f"5. Feature Standardization Pipeline:\n")
        f.write(f"   - Method: StandardScaler (Z-score normalization: z = (x - u) / s)\n")
        f.write(f"   - Data Leakage Prevention: Fitted STRICTLY on X_train only; X_val and X_test transformed using training parameters.\n")
        f.write(f"   - Persisted Scaler File: outputs/breast_cancer_scaler.pkl\n\n")
        f.write("6. Processed Data Artifacts:\n")
        f.write(f"   - {os.path.join(processed_dir, 'X_train_scaled.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'X_val_scaled.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'X_test_scaled.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'train_preprocessed.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'val_preprocessed.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'test_preprocessed.csv')}\n")
        f.write(f"   - {os.path.join(processed_dir, 'breast_cancer_preprocessed.csv')}\n")

    print(f"Summary text written to: {summary_text_path}")
    print("\n[SUCCESS] Experiment 3 Data Preprocessing completed successfully!")

if __name__ == "__main__":
    run_experiment_3()
