"""
Experiment 4: Exploratory Data Analysis (EDA)
Health-Project 3: Breast Cancer Detection
Course: CSE4001 - Applications of Machine Learning in Industry

This script implements the full Exploratory Data Analysis workflow:
1. Load raw and preprocessed datasets
2. Inspect dataset structure and summary statistics
3. Analyze and visualize target class distribution
4. Analyze feature distributions with histograms
5. Compare feature characteristics across Benign and Malignant diagnoses with boxplots
6. Generate feature correlation matrix and heatmap
7. Identify highly correlated / multicollinear feature pairs (|r| > 0.90)
8. Detect outliers using boxplots and IQR analysis
9. Calculate and visualize class-wise feature means
10. Analyze bivariate feature relationships (e.g., Mean Radius vs Mean Area)
11. Compute and rank feature correlations with the diagnosis target
12. Export all high-resolution figures and statistical summaries
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style configuration for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def run_experiment_4():
    print("=" * 75)
    print("EXPERIMENT 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("Health-Project 3: Breast Cancer Detection")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1 & 2. Load Dataset
    # ---------------------------------------------------------
    raw_csv_path = os.path.join(DATA_DIR, "breast_cancer.csv")
    cleaned_csv_path = os.path.join(PROCESSED_DIR, "breast_cancer_preprocessed.csv")

    if os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path)
        print(f"\n[1] Raw Dataset Loaded from: {raw_csv_path}")
        print(f"    Raw Shape: {df_raw.shape}")
    else:
        raise FileNotFoundError(f"Raw dataset not found at {raw_csv_path}")

    # Standardize working dataframe
    df = df_raw.copy()
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Standardize column names (replace spaces with underscores if any)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Map diagnosis to standard representations
    if df['diagnosis'].dtype == object or isinstance(df['diagnosis'].iloc[0], str):
        diagnosis_labels = df['diagnosis'].map({'B': 'Benign (B)', 'M': 'Malignant (M)'})
        diagnosis_numeric = df['diagnosis'].map({'B': 0, 'M': 1})
    else:
        diagnosis_labels = df['diagnosis'].map({0: 'Benign (0)', 1: 'Malignant (1)'})
        diagnosis_numeric = df['diagnosis']

    df_plot = df.copy()
    df_plot['diagnosis_label'] = diagnosis_labels
    df_plot['diagnosis_num'] = diagnosis_numeric

    # ---------------------------------------------------------
    # 3. Dataset Information & Statistical Summary
    # ---------------------------------------------------------
    print("\n[2] Dataset Information & Summary:")
    print(f"    Rows (Samples): {df.shape[0]}")
    print(f"    Columns: {df.shape[1]}")
    print(f"    Missing Values: {df.isnull().sum().sum()} (0.0%)")
    print(f"    Duplicate Rows: {df.duplicated().sum()} (0.0%)")

    # Define feature groups
    mean_features = [c for c in df.columns if c.endswith('_mean')]
    se_features = [c for c in df.columns if c.endswith('_se')]
    worst_features = [c for c in df.columns if c.endswith('_worst')]
    all_numeric_features = mean_features + se_features + worst_features

    summary_stats = df[mean_features].describe()
    print("\nStatistical Summary (Mean Features):")
    print(summary_stats.round(3))

    # ---------------------------------------------------------
    # 4. Target Class Distribution Analysis
    # ---------------------------------------------------------
    class_counts = df['diagnosis'].value_counts()
    n_benign = class_counts.get('B', class_counts.get(0, 0))
    n_malignant = class_counts.get('M', class_counts.get(1, 0))
    pct_benign = (n_benign / len(df)) * 100
    pct_malignant = (n_malignant / len(df)) * 100

    print(f"\n[3] Target Class Distribution:")
    print(f"    Benign:    {n_benign} ({pct_benign:.2f}%)")
    print(f"    Malignant: {n_malignant} ({pct_malignant:.2f}%)")
    print(f"    Imbalance Ratio: {n_benign / n_malignant:.2f} : 1")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_colors = ['#2B6CB0', '#C53030']
    bars = ax.bar(['Benign (B / 0)', 'Malignant (M / 1)'], [n_benign, n_malignant], color=bar_colors, width=0.5, edgecolor='black', linewidth=0.8)
    for bar, count, pct in zip(bars, [n_benign, n_malignant], [pct_benign, pct_malignant]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 8, f'{count} ({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 420)
    ax.set_title('Target Class Distribution: Benign vs Malignant Cases', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Diagnosis Class', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Biopsy Samples', fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart1_path = os.path.join(OUTPUTS_DIR, "experiment4_class_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart1_path}")

    # ---------------------------------------------------------
    # 5. Feature Distribution (Histograms for Mean Features)
    # ---------------------------------------------------------
    print("\n[4] Generating Feature Distribution Histograms...")
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    axes = axes.flatten()

    palette_map = {
        'Benign (B)': '#2B6CB0', 'Malignant (M)': '#C53030',
        'Benign (0)': '#2B6CB0', 'Malignant (1)': '#C53030'
    }

    for idx, feature in enumerate(mean_features):
        ax = axes[idx]
        sns.histplot(
            data=df_plot,
            x=feature,
            hue='diagnosis_label',
            kde=True,
            bins=25,
            ax=ax,
            palette=palette_map,
            alpha=0.5,
            edgecolor='white'
        )
        ax.set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Exploratory Histograms: Nuclear Mean Feature Distributions by Diagnosis', fontsize=14, fontweight='bold', y=1.002)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUTS_DIR, "experiment4_feature_distributions.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved figure: {chart2_path}")

    # ---------------------------------------------------------
    # 6, 7, 8, 9, 10. Compare Features Between Diagnoses (Boxplots)
    # ---------------------------------------------------------
    print("\n[5] Generating Comparative Boxplots...")
    important_features = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean"
    ]
    # Filter to actual available features
    important_features = [f for f in important_features if f in df_plot.columns]

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(important_features):
        ax = axes[idx]
        sns.boxplot(
            data=df_plot,
            x='diagnosis_label',
            y=feature,
            hue='diagnosis_label',
            palette=palette_map,
            legend=False,
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.85),
            fliersize=3
        )
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Measurement Value', fontsize=8.5)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.suptitle('Comparative Nuclear Morphology: Benign vs Malignant Boxplots', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    chart3_path = os.path.join(OUTPUTS_DIR, "experiment4_boxplots_by_diagnosis.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved figure: {chart3_path}")

    # ---------------------------------------------------------
    # 11 & 12. Correlation Analysis & Heatmap
    # ---------------------------------------------------------
    print("\n[6] Performing Correlation Analysis & Generating Heatmap...")
    numeric_df = df[all_numeric_features].copy()
    correlation_matrix = numeric_df.corr()

    # Full 30-feature heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        annot=False,
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'},
        ax=ax,
        linewidths=0.2
    )
    ax.set_title('Correlation Heatmap: 30 Continuous Nuclear Morphological Features', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    chart4_path = os.path.join(OUTPUTS_DIR, "experiment4_correlation_heatmap.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart4_path}")

    # Mean features annotated correlation heatmap for easy reading
    mean_corr = df[mean_features].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(
        mean_corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8.5},
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('Correlation Matrix of Key Nuclear Mean Features', fontsize=12, fontweight='bold', pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    chart4b_path = os.path.join(OUTPUTS_DIR, "experiment4_mean_correlation_matrix.png")
    plt.savefig(chart4b_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart4b_path}")

    # ---------------------------------------------------------
    # 13. Find Highly Correlated Features (|r| > 0.90)
    # ---------------------------------------------------------
    threshold = 0.90
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            val = correlation_matrix.iloc[i, j]
            if abs(val) > threshold:
                high_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    round(val, 4)
                ))

    print(f"\n[7] Highly Correlated Feature Pairs (|r| > {threshold}):")
    print(f"    Total Multicollinear Pairs Detected: {len(high_corr)}")
    for f1, f2, r in high_corr:
        print(f"    - {f1} <--> {f2}: r = {r:.4f}")

    # ---------------------------------------------------------
    # 14. Detect Outliers (IQR Method & Boxplot)
    # ---------------------------------------------------------
    print("\n[8] Outlier Detection & Analysis...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scale_cols = [c for c in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"] if c in df_plot.columns]
    df_plot[scale_cols].boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor='#E2E8F0', color='#2D3748'))
    ax.set_title('Outlier Inspection across Scale-Dominant Mean Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Raw Physical Measurement Value', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=15, fontweight='bold')
    plt.tight_layout()
    chart5_path = os.path.join(OUTPUTS_DIR, "experiment4_outlier_detection.png")
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart5_path}")

    # Outlier counting by IQR
    outlier_counts = {}
    for col in all_numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_counts[col] = n_outliers

    print("    Top Outlier Features (IQR Rule):")
    sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    for col, count in sorted_outliers:
        print(f"    - {col}: {count} samples ({count/len(df)*100:.1f}%)")

    # ---------------------------------------------------------
    # 15 & 16. Calculate & Visualize Class-Wise Feature Means
    # ---------------------------------------------------------
    print("\n[9] Calculating Class-Wise Feature Means...")
    class_means = df_plot.groupby("diagnosis_label")[important_features].mean()
    print("\nClass-Wise Feature Means Table:")
    print(class_means.T.round(3))

    fig, ax = plt.subplots(figsize=(12, 6))
    class_means.T.plot(kind='bar', ax=ax, color=['#2B6CB0', '#C53030'], width=0.7, edgecolor='black', linewidth=0.5)
    ax.set_title('Average Feature Values by Diagnosis Class (Nuclear Mean Features)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Nuclear Morphological Feature', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Physical Value (Raw Units)', fontsize=10, fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    ax.legend(title='Diagnosis')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart6_path = os.path.join(OUTPUTS_DIR, "experiment4_classwise_means.png")
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart6_path}")

    # ---------------------------------------------------------
    # 17. Feature Relationship: Radius Mean vs Area Mean Scatter
    # ---------------------------------------------------------
    print("\n[10] Generating Bivariate Scatter Plot (Radius Mean vs Area Mean)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for diag, color, marker in [('Benign (B)', '#2B6CB0', 'o'), ('Malignant (M)', '#C53030', '^')]:
        sub = df_plot[df_plot['diagnosis_label'] == diag]
        if len(sub) == 0:
            sub = df_plot[df_plot['diagnosis_label'].str.startswith(diag[:1])]
        ax.scatter(sub['radius_mean'], sub['area_mean'], color=color, label=diag, alpha=0.7, edgecolors='white', linewidth=0.5, s=45, marker=marker)

    ax.set_title('Bivariate Relationship: Mean Radius vs Mean Area by Diagnosis', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Mean Radius (um)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Area (um^2)', fontsize=10, fontweight='bold')
    ax.legend(title='Diagnosis', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart7_path = os.path.join(OUTPUTS_DIR, "experiment4_radius_vs_area_scatter.png")
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart7_path}")

    # ---------------------------------------------------------
    # 18. Feature Correlation with Diagnosis
    # ---------------------------------------------------------
    print("\n[11] Computing Feature Correlation with Binary Diagnosis Target...")
    numeric_with_target = df[all_numeric_features].copy()
    numeric_with_target['diagnosis'] = df_plot['diagnosis_num'].values
    target_corr = numeric_with_target.corr()['diagnosis'].drop('diagnosis').sort_values(ascending=False)

    print("\nTop 10 Positively Correlated Features with Malignancy (Target = 1):")
    print(target_corr.head(10).round(4))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_corr = ['#C53030' if x > 0 else '#2B6CB0' for x in target_corr]
    target_corr.plot(kind='barh', ax=ax, color=colors_corr, edgecolor='black', linewidth=0.4)
    ax.set_title('Pearson Correlation of 30 Nuclear Morphological Features with Malignancy (1)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Morphological Feature', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart8_path = os.path.join(OUTPUTS_DIR, "experiment4_target_correlations.png")
    plt.savefig(chart8_path, dpi=300)
    plt.close()
    print(f"    [+] Saved figure: {chart8_path}")

    # ---------------------------------------------------------
    # Export Text Summary
    # ---------------------------------------------------------
    summary_path = os.path.join(OUTPUTS_DIR, "experiment4_eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("EXPERIMENT 4: EXPLORATORY DATA ANALYSIS (EDA) SUMMARY\n")
        f.write("Health-Project 3: Breast Cancer Detection\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features\n")
        f.write(f"Class Distribution: Benign = {n_benign} ({pct_benign:.2f}%), Malignant = {n_malignant} ({pct_malignant:.2f}%)\n")
        f.write(f"Imbalance Ratio: {n_benign/n_malignant:.3f} : 1\n\n")
        f.write("Highly Correlated Feature Pairs (|r| > 0.90):\n")
        for f1, f2, r in high_corr:
            f.write(f"  - {f1:28s} <--> {f2:28s}: r = {r:.4f}\n")
        f.write("\nTop 10 Diagnostic Correlates with Malignancy (Target = 1):\n")
        for feat, val in target_corr.head(10).items():
            f.write(f"  - {feat:28s}: r = {val:.4f}\n")
        f.write("\nLowest Correlated Features:\n")
        for feat, val in target_corr.tail(5).items():
            f.write(f"  - {feat:28s}: r = {val:.4f}\n")

    print(f"\n[+] Statistical Summary Report saved to: {summary_path}")
    print("\n[SUCCESS] Experiment 4 EDA Execution Completed Successfully!")

if __name__ == "__main__":
    run_experiment_4()
