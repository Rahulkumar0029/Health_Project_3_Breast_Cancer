# Experiment 2: Dataset Details

## 1. Objective

This experiment performs a non-modifying audit of the Breast Cancer Wisconsin (Diagnostic) dataset. It documents the source, dimensions, schema, feature meanings, target distribution, data quality, privacy considerations, licensing constraints, and known limitations. No preprocessing, scaling, imputation, resampling, feature selection, or model training is performed.

## 2. Dataset Source and Provenance

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Primary source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)
- **UCI URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Local data file:** `data/wdbc.data`
- **Metadata file:** `breast+cancer+wisconsin+diagnostic/wdbc.names`
- **Creators:** William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian
- **Donor:** Nick Street
- **Date in metadata:** November 1995
- **Kaggle mirror:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

The local `wdbc.data` file is the original headerless comma-separated UCI format. The audit code assigns the documented column names at read time; it does not modify the raw file.

### License and citation

The downloaded `wdbc.names` metadata does not state a license. Therefore, this report does not assign a Creative Commons license to the dataset. Before submission or redistribution, verify the terms displayed by the exact UCI or Kaggle download page being used. Academic use should cite the dataset creators and the original publications listed in `wdbc.names`.

Suggested citation:

> Wolberg, W. H., Street, W. N., and Mangasarian, O. L. Wisconsin Diagnostic Breast Cancer dataset, UCI Machine Learning Repository, 1995.

## 3. Dataset Size and Structure

The audit was run on the local raw file using `notebooks/experiment_2_dataset_audit.py`.

| Item | Result |
|---|---:|
| Records | 569 |
| Raw attributes | 32 |
| Metadata field | 1 (`id`) |
| Target field | 1 (`diagnosis`) |
| Numeric predictive features | 30 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Unique IDs | 569 |

The raw structure is:

`id` + `diagnosis` + 30 numeric measurements = 32 columns.

## 4. Feature and Label Dictionary

The features are computed from digitized fine-needle aspirate (FNA) images and describe cell nuclei. Ten measurements are summarized using three statistics: mean, standard error (`se`), and worst (the mean of the three largest values). The word "worst" therefore does not mean a single maximum value.

### Label and metadata

| Column | Type | Meaning | Use in this experiment |
|---|---|---|---|
| `id` | Integer | Dataset/sample identification number | Metadata only; not a predictive feature |
| `diagnosis` | Categorical | `B` = benign, `M` = malignant | Target label |

### 30 predictive features

| Measurement | Meaning |
|---|---|
| `radius` | Mean distance from the nucleus center to points on its perimeter |
| `texture` | Standard deviation of gray-scale values |
| `perimeter` | Length of the nucleus perimeter |
| `area` | Area enclosed by the nucleus contour |
| `smoothness` | Local variation in radius lengths |
| `compactness` | Perimeter squared divided by area, minus 1.0 |
| `concavity` | Severity of concave portions of the contour |
| `concave_points` | Number of concave portions of the contour |
| `symmetry` | Symmetry of the nucleus |
| `fractal_dimension` | Boundary complexity using a coastline approximation, minus 1 |

Each measurement has these three columns:

| Suffix | Meaning |
|---|---|
| `_mean` | Mean value for the measurement |
| `_se` | Standard error of the measurement |
| `_worst` | Mean of the three largest values for the measurement |

Thus, for example, `radius_mean` is the mean radius, `radius_se` is its standard error, and `radius_worst` is the mean of the three largest radius values. The complete feature columns are:

`radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave_points_mean`, `symmetry_mean`, `fractal_dimension_mean`, `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave_points_se`, `symmetry_se`, `fractal_dimension_se`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave_points_worst`, `symmetry_worst`, `fractal_dimension_worst`.

## 5. Target Distribution and Class Balance

| Label | Meaning | Count | Percentage |
|---|---|---:|---:|
| `B` | Benign | 357 | 62.74% |
| `M` | Malignant | 212 | 37.26% |
| **Total** |  | **569** | **100.00%** |

Benign cases are the majority class. This is a moderate imbalance, so later experiments should report class-sensitive measures such as malignant-class recall/sensitivity, precision, F1-score, and a confusion matrix rather than accuracy alone. No balancing is performed in Experiment 2.

The class-distribution chart is saved at `outputs/experiment2_class_distribution.png` and the complete audit output is saved at `outputs/experiment2_audit_summary.txt`.

## 6. Data-Quality Assessment

- All 569 records have values for all 32 columns.
- No duplicate rows were detected.
- The 30 predictive columns are numeric floating-point values.
- `diagnosis` is categorical, with only the documented `B` and `M` labels.
- All 569 `id` values are unique, but uniqueness does not make `id` clinically meaningful. It should remain metadata and be excluded from predictive modeling in a later preprocessing experiment.
- The local UCI file has no header row. This is a file-format issue, not a missing-data issue; the audit assigns documented names while reading it.
- The local file does not contain the optional Kaggle parsing artifact `Unnamed: 32`.

## 7. Consent, Privacy, and Licensing Constraints

The dataset contains no names, addresses, ages, diagnoses linked to identifiable people, or direct clinical histories in the local files. The `id` field is treated as a dataset/sample identifier, not as a patient identity. Nevertheless, the metadata does not provide enough information in this project folder to independently verify participant consent, institutional review-board status, de-identification procedures, or the chain of custody of the records. The data should therefore be used for educational and research analysis, not for identifying individuals or making clinical decisions.

The exact license and redistribution terms must be checked at the chosen source before publishing the CSV or sharing derived materials. Creator and source attribution should be retained.

## 8. Known Limitations

1. The sample contains only 569 records, which limits generalization and makes careful validation important.
2. The features come from FNA-derived nuclear images and do not include patient history, age, genetics, mammography, or histopathology images.
3. The dataset is historical and associated with one institutional source, so performance may not transfer to other populations, equipment, or clinical workflows.
4. The class distribution is imbalanced toward benign cases.
5. The dataset is a benchmark for machine-learning experiments, not a clinically validated diagnostic system.
6. The local metadata does not document consent, privacy procedures, or a license; those details must be verified from the source provider.

## 9. Conclusion

The raw WDBC dataset was successfully audited without modifying its values. It contains 569 records, 32 attributes, 30 numeric predictive features, one metadata field, and one categorical target. The audit found zero missing values and zero duplicate rows. The target distribution is 357 benign cases and 212 malignant cases. These findings satisfy the Experiment 2 dataset-details requirements and establish the documented baseline for later preprocessing and modeling experiments.
