from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
CSV_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

WDBC_COLUMNS = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def load_dataset() -> pd.DataFrame:
    """Load the headerless UCI WDBC file without changing its values."""
    return pd.read_csv(DATA_PATH, header=None, names=WDBC_COLUMNS)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    df.to_csv(CSV_PATH, index=False)
    feature_columns = [column for column in WDBC_COLUMNS if column not in {"id", "diagnosis"}]
    diagnosis_counts = df["diagnosis"].value_counts().sort_index()
    diagnosis_percentages = (diagnosis_counts / len(df) * 100).round(2)

    summary = [
        "EXPERIMENT 2: DATASET INSPECTION AND AUDIT",
        "",
        f"Source file: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Predictive numeric features: {len(feature_columns)}",
        f"Missing values: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"Unique IDs: {df['id'].nunique()}",
        "",
        "Diagnosis distribution:",
    ]
    summary.extend(
        f"  {label}: {count} ({diagnosis_percentages[label]:.2f}%)"
        for label, count in diagnosis_counts.items()
    )
    summary.extend(["", "Data types:", df.dtypes.value_counts().to_string(), "", "Statistics:"])
    summary.append(df[feature_columns].describe().round(3).to_string())

    report_path = OUTPUT_DIR / "experiment2_audit_summary.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")

    axis = diagnosis_counts.plot(kind="bar", color=["#2a9d8f", "#e76f51"], figsize=(7, 5))
    axis.set_title("Breast Cancer Diagnosis Distribution")
    axis.set_xlabel("Diagnosis (B = Benign, M = Malignant)")
    axis.set_ylabel("Number of Samples")
    axis.tick_params(axis="x", rotation=0)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experiment2_class_distribution.png", dpi=300)
    plt.close(figure)

    print("\n".join(summary[:16]))
    print(f"Wrote: {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\nWrote: {report_path.relative_to(PROJECT_ROOT)}")
    print("Wrote: outputs/experiment2_class_distribution.png")


if __name__ == "__main__":
    main()