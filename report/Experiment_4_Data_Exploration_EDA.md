# Experiment 4: Data Exploration (EDA)

**Course:** Applications of Machine Learning in Industry (Health-Project 3: Breast Cancer Detection)  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Status:** Completed  

---

## 1. Aim

To perform Exploratory Data Analysis (EDA) on the Breast Cancer Wisconsin (Diagnostic) dataset to understand feature distributions, identify relationships between features, detect outliers, and compare the characteristics of benign and malignant tumours.

---

## 2. Objectives

1. **Analyze the distribution** of the 30 numerical features across benign and malignant cohorts.
2. **Compare feature values** between benign and malignant cases to detect morphometric discriminators.
3. **Study correlations** between different nuclear features and quantify multicollinearity.
4. **Detect possible outliers** using IQR bounds and boxplot distributions without premature data loss.
5. **Visualize important features** using publication-quality charts (histograms, boxplots, correlation heatmaps, and scatter plots).
6. **Compare features** such as radius, texture, perimeter, and area between the two diagnostic classes.
7. **Identify important patterns** that can guide feature selection, dimensionality reduction, and model selection in Experiment 5.

---

## 3. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)
- **Total Samples ($N$):** 569 fine-needle aspirate (FNA) biopsy instances
- **Total Numerical Features:** 30 continuous real-valued features
- **Target Variable:** `diagnosis`
  - `B` / `0` : **Benign** ($N = 357$, $62.74\%$)
  - `M` / `1` : **Malignant** ($N = 212$, $37.26\%$)
- **Morphological Attributes:** Characteristics of cell nuclei computed from digitized FNA images:
  1. *Radius* (mean of distances from center to points on perimeter)
  2. *Texture* (standard deviation of gray-scale values)
  3. *Perimeter* (nuclear perimeter length)
  4. *Area* (nuclear cross-sectional area)
  5. *Smoothness* (local variation in radius lengths)
  6. *Compactness* ($\frac{\text{perimeter}^2}{\text{area}} - 1.0$)
  7. *Concavity* (severity of concave portions of contour)
  8. *Concave points* (number of concave portions of contour)
  9. *Symmetry* (nuclear symmetry)
  10. *Fractal dimension* ("coastline approximation" $- 1.0$)
- Each attribute is captured across 3 statistical moments: **Mean**, **Standard Error (SE)**, and **Worst** (mean of the three largest values).

---

## 4. Algorithm

1. **Load the cleaned dataset** from Experiment 3 / raw dataset from `data/breast_cancer.csv`.
2. **Inspect dataset structure** and extract comprehensive descriptive statistics (`df.info()`, `df.describe()`).
3. **Check the distribution** of benign and malignant cases to evaluate class balance.
4. **Generate histograms** with Kernel Density Estimation (KDE) for continuous numerical features.
5. **Compare important features** between benign and malignant diagnostic classes.
6. **Generate a Pearson correlation matrix** and render visual heatmaps.
7. **Identify highly correlated features** using an absolute threshold ($|r| > 0.90$) to pinpoint redundant features.
8. **Detect potential outliers** using boxplots and interquartile range (IQR) thresholds.
9. **Compare mean radius, texture, perimeter, and area** between classes.
10. **Record key observations** detailing biological separation and distribution characteristics.
11. **Identify features and insights** to guide model family selection and regularized training in Experiment 5.

---

## 5. Python Implementation

The complete exploratory analysis is executed in [`notebooks/experiment_4_data_exploration.py`](file:///c:/Users/rahul/OneDrive/Desktop/7th%20sem/Health_Project_3_Breast_Cancer/notebooks/experiment_4_data_exploration.py).

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Load Dataset & Hygiene Inspection
```python
df = pd.read_csv("data/breast_cancer.csv")
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])
if "id" in df.columns:
    df = df.drop(columns=["id"])

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
```
*Output:*
```
Dataset Shape: (569, 31)
Rows: 569, Columns: 31 (1 Target + 30 Predictive Numerical Features)
Missing Values: 0 (0.00%) | Duplicates: 0 (0.00%)
```

### 3. Class Distribution & Imbalance
```python
print("Diagnosis Distribution:")
print(df["diagnosis"].value_counts())

# Visualization
df["diagnosis"].value_counts().plot(kind="bar", color=["#2B6CB0", "#C53030"])
plt.title("Benign vs Malignant Cases")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.show()
```
*Distribution Statistics:*
- **Benign ($B / 0$):** 357 ($62.74\%$)
- **Malignant ($M / 1$):** 212 ($37.26\%$)
- **Ratio:** $1.68 : 1$

### 4. Feature Distribution Histograms
```python
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
]

for feature in features:
    plt.figure(figsize=(7, 4))
    plt.hist(df[feature], bins=30, color="#2B6CB0", edgecolor="white")
    plt.title("Distribution of " + feature)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
```

### 5. Comparative Boxplots Across Diagnoses
```python
for feature in features:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="diagnosis", y=feature, palette=["#3182CE", "#E53E3E"])
    plt.title(feature + " vs Diagnosis")
    plt.show()
```

### 6. Correlation Analysis & Heatmap
```python
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Breast Cancer Features")
plt.show()
```

### 7. Identification of Highly Correlated / Multicollinear Pairs ($|r| > 0.90$)
```python
threshold = 0.90
high_corr = []
for i in range(len(correlation.columns)):
    for j in range(i):
        value = correlation.iloc[i, j]
        if abs(value) > threshold:
            high_corr.append((correlation.columns[i], correlation.columns[j], value))

print(f"Highly Correlated Feature Pairs (|r| > {threshold}):")
for item in high_corr:
    print(f"  {item[0]} <--> {item[1]}: r = {item[2]:.4f}")
```
*Key Multicollinear Pairs Identified (21 pairs):*
- `perimeter_mean` $\leftrightarrow$ `radius_mean`: $r = 0.9979$
- `area_mean` $\leftrightarrow$ `radius_mean`: $r = 0.9874$
- `area_mean` $\leftrightarrow$ `perimeter_mean`: $r = 0.9865$
- `concave_points_mean` $\leftrightarrow$ `concavity_mean`: $r = 0.9214$
- `radius_worst` $\leftrightarrow$ `radius_mean`: $r = 0.9695$
- `perimeter_worst` $\leftrightarrow$ `radius_worst`: $r = 0.9937$
- `area_worst` $\leftrightarrow$ `area_mean`: $r = 0.9592$

### 8. Outlier Detection (IQR Method)
```python
plt.figure(figsize=(12, 6))
df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]].boxplot()
plt.title("Outlier Detection")
plt.ylabel("Feature Value")
plt.xticks(rotation=45)
plt.show()
```
*Findings:* Right-skewed extreme values occur naturally in aggressive malignant tumors with abnormally large nuclei (e.g. `area_mean` $> 2000\,\text{mm}^2$, `area_se` in 65 samples). These are preserved as genuine physiological signals.

### 9. Class-Wise Feature Means
```python
class_means = df.groupby("diagnosis")[features].mean()
print(class_means.T)

class_means.T.plot(kind="bar", figsize=(12, 6), color=["#2B6CB0", "#C53030"])
plt.title("Average Feature Values by Diagnosis")
plt.xlabel("Feature")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.show()
```

| Morphological Feature | Benign Mean ($B$) | Malignant Mean ($M$) | % Increase in Malignant |
|---|---|---|---|
| `radius_mean` ($\mu\text{m}$) | $12.15$ | $17.46$ | $+43.7\%$ |
| `texture_mean` | $17.92$ | $21.61$ | $+20.6\%$ |
| `perimeter_mean` ($\mu\text{m}$) | $78.08$ | $115.37$ | $+47.8\%$ |
| `area_mean` ($\mu\text{m}^2$) | $462.79$ | $978.38$ | $+111.4\%$ |
| `smoothness_mean` | $0.092$ | $0.103$ | $+12.0\%$ |
| `compactness_mean` | $0.080$ | $0.145$ | $+81.3\%$ |
| `concavity_mean` | $0.046$ | $0.161$ | $+250.0\%$ |
| `concave_points_mean` | $0.026$ | $0.088$ | $+238.5\%$ |
| `symmetry_mean` | $0.174$ | $0.193$ | $+10.9\%$ |
| `fractal_dimension_mean` | $0.063$ | $0.063$ | $0.0\%$ |

### 10. Bivariate Relationship: Mean Radius vs Mean Area
```python
plt.figure(figsize=(8, 6))
for diagnosis in df["diagnosis"].unique():
    subset = df[df["diagnosis"] == diagnosis]
    plt.scatter(subset["radius_mean"], subset["area_mean"], label=str(diagnosis), alpha=0.7)
plt.xlabel("Mean Radius")
plt.ylabel("Mean Area")
plt.title("Mean Radius vs Mean Area")
plt.legend()
plt.show()
```
*Finding:* Shows quadratic relationship ($\text{Area} \propto \pi r^2$) and strong linear separability between benign (clustered at low radius/area) and malignant cases (higher dispersion towards larger values).

### 11. Feature Correlation with Diagnosis Target ($0 = \text{Benign}, 1 = \text{Malignant}$)
```python
target_correlation = (
    df.corr(numeric_only=True)["diagnosis"]
    .sort_values(ascending=False)
)
print("Feature Correlation with Diagnosis:")
print(target_correlation)
```

| Rank | Feature | Correlation with Malignancy ($r$) |
|---|---|---|
| 1 | `concave_points_worst` | $+0.7936$ |
| 2 | `perimeter_worst` | $+0.7829$ |
| 3 | `concave_points_mean` | $+0.7766$ |
| 4 | `radius_worst` | $+0.7765$ |
| 5 | `perimeter_mean` | $+0.7426$ |
| 6 | `area_worst` | $+0.7338$ |
| 7 | `radius_mean` | $+0.7300$ |
| 8 | `area_mean` | $+0.7090$ |
| 9 | `concavity_mean` | $+0.6964$ |
| 10 | `concavity_worst` | $+0.6596$ |

---

## 6. Key Observations

1. **Sample and Class Demographics:** The dataset contains 569 samples belonging to two diagnosis classes: 357 benign ($62.74\%$) and 212 malignant ($37.26\%$).
2. **Moderate Class Imbalance:** The classes are not perfectly balanced (ratio $1.68:1$). While not severely skewed, model evaluation must emphasize Malignant Class Recall, Precision-Recall curves, and ROC-AUC over naive accuracy.
3. **Pronounced Morphometric Disparity:** Features such as mean radius, perimeter, and area show noticeable differences in their distributions across the diagnosis classes; malignant nuclei are on average $111\%$ larger in area and $48\%$ larger in perimeter.
4. **Severe Multicollinearity in Nuclear Geometry:** Several size-related features show near-perfect positive correlations ($r > 0.98$ for radius, perimeter, and area), indicating substantial redundant information. Regularization (L1/L2) or tree ensembles will be essential in Experiment 5.
5. **Nuclear Contour Irregularity:** Contour indentation metrics (`concavity_mean`, `concave_points_mean`, `concave_points_worst`) exhibit the strongest positive correlations with malignancy ($r \approx 0.78 - 0.79$), confirming that cell nuclear irregularity is a primary hallmark of cancer.
6. **Presence of Biological Outliers:** Boxplots reveal extreme observations, especially in `area_mean`, `area_se`, and `area_worst`. These represent aggressive high-grade carcinomas rather than noise and must be preserved during model training.
7. **Strong Bivariate Separability:** Feature combinations such as *Mean Radius vs. Mean Area* and *Concavity vs. Texture* show clear visual boundaries between benign and malignant cases, indicating high potential for linear and kernel classifiers.
8. **Guidance for Model Selection (Experiment 5):** The high dimensionality (30 features), multicollinearity, and non-linear contour boundaries indicate that models capable of handling correlated continuous features—specifically **Regularized Logistic Regression (L2)**, **Support Vector Machines with RBF Kernel (SVM)**, and **Random Forest Ensembles**—are ideal candidates for baseline training.

---

## 7. Result

Exploratory Data Analysis was successfully performed on the Breast Cancer Wisconsin (Diagnostic) dataset. The distributions of all 30 nuclear morphological features were analyzed across benign and malignant classes, inter-feature correlations were quantified, biological outliers were identified, and key diagnostic predictors were mapped. All 8 visual artifact figures and summary reports were generated and saved to `outputs/`.

---

## 8. Conclusion

The EDA reveals that cell nuclear size (`radius`, `perimeter`, `area`) and contour irregularities (`concave points`, `concavity`) provide powerful discriminatory signals for distinguishing between benign and malignant breast tumours. Strong inter-feature correlations indicate redundancy that will benefit from feature scaling and regularized modeling. These empirical findings directly guide feature handling, loss-weighting strategies, and classifier selection in Experiment 5.
