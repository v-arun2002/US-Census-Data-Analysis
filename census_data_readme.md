# 📊 Census Data Analysis & Visualization Project

*A comprehensive analysis of US Census health metrics data with focus on static visualizations and statistical modeling*

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Analysis Pipeline](#-analysis-pipeline)
- [Visualizations](#-visualizations)
- [Machine Learning Models](#-machine-learning-models)
- [Key Findings](#-key-findings)
- [Installation & Usage](#-installation--usage)
- [Results](#-results)

---

## 🎯 Overview

This project provides an in-depth examination of **communication trends, patterns, and health metrics** from US Census data. Using sophisticated visualization methods and machine learning techniques, we transform complex data into actionable insights.

**Main Objectives:**
- Investigate and showcase significant insights from census health data
- Uncover trends, correlations, and performance metrics
- Provide visual displays for decision-making support
- Apply advanced statistical and ML techniques for deeper analysis

---

## 📁 Dataset

**Source:** `census.csv`

**Composition:**
- **72,337 rows × 81 columns**
- Geographic identifiers (State, County, Tract FIPS)
- Demographic attributes
- Health condition prevalence rates
- Disability measures
- Healthcare access indicators

**Key Variables:**
- Total Population
- Chronic disease prevalence (Diabetes, Arthritis, Cancer, etc.)
- Mental health indicators (Depression, Sleep disorders)
- Physical health metrics (Obesity, Mobility, Self-care)
- Demographic information

---

## ✨ Key Features

### 🧹 Data Preparation
- Missing value handling using Pandas
- Duplicate detection and removal
- Data type conversion and normalization
- Categorical variable encoding
- Min-Max scaling for numerical features

### 📊 Exploratory Data Analysis
- Distribution analysis via histograms and box plots
- Correlation heatmaps
- Descriptive statistics
- Hypothesis testing (t-tests, chi-squared)
- Regression analysis

### 🤖 Advanced Analytics
- Linear Regression modeling
- Random Forest classification
- SHAP (SHapley Additive Explanations) for feature importance
- ARIMA forecasting for time series
- Clustering analysis (K-means)
- Cross-validation techniques

---

## 🛠 Technologies Used

```python
# Core Libraries
- Python 3.8+
- Pandas (Data manipulation)
- NumPy (Numerical operations)

# Visualization
- Matplotlib
- Seaborn
- Plotly (interactive plots)

# Machine Learning
- Scikit-learn
- SHAP (Model interpretability)
- Statsmodels (Statistical modeling)

# Statistical Analysis
- SciPy
- Statsmodels
```

---

## 📂 Project Structure

```
census-data-analysis/
│
├── data/
│   └── census.csv                 # Raw dataset
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb  # Data loading & inspection
│   ├── 02_data_cleaning.ipynb     # Preprocessing
│   ├── 03_eda.ipynb               # Exploratory analysis
│   └── 04_advanced_analysis.ipynb # ML models
│
├── visualizations/
│   ├── histogram_population.png
│   ├── correlation_heatmap.png
│   ├── boxplot_states.png
│   ├── scatter_diabetes.png
│   └── shap_analysis.png
│
├── models/
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
│
├── reports/
│   └── Project_Report.pdf         # Full documentation
│
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🔄 Analysis Pipeline

```mermaid
graph LR
    A[Data Acquisition] --> B[Data Inspection]
    B --> C[Data Cleaning]
    C --> D[EDA]
    D --> E[Statistical Analysis]
    E --> F[ML Modeling]
    F --> G[Visualization]
    G --> H[Insights & Conclusions]
```

### Step-by-Step Process

1. **Data Acquisition** → Load census.csv using Pandas
2. **Data Inspection** → Examine structure, types, dimensions
3. **Data Cleaning** → Handle missing values, remove duplicates
4. **Exploratory Analysis** → Generate descriptive statistics
5. **Visualization** → Create histograms, heatmaps, scatter plots
6. **Statistical Modeling** → Apply regression and hypothesis tests
7. **Machine Learning** → Train Linear Regression & Random Forest
8. **Interpretation** → SHAP analysis for feature importance

---

## 📈 Visualizations

### Histogram - Population Distribution
Shows right-skewed distribution with most census tracts having populations under 5,000.

### Correlation Heatmap
Reveals strong correlations between:
- Mobility issues and Self-care difficulties
- Obesity and Diabetes prevalence
- Depression and Sleep disorders

### Box Plot by State
Identifies population outliers and variance across US states.

### Scatter Plot
Demonstrates negative correlation between total population and diabetes prevalence rates.

### SHAP Analysis
Top features impacting Diabetes Prevalence:
1. Mobility limitations (highest impact)
2. Self-care difficulties
3. High blood pressure
4. Depression
5. Kidney disease

---

## 🤖 Machine Learning Models

### Model Performance Comparison

| Model | R² Score | MSE | MAE |
|-------|----------|-----|-----|
| **Linear Regression** | 0.963878 | 0.407217 | 0.496 |
| **Random Forest** | 0.980882 | 0.257567 | 0.377 |

**Winner:** Random Forest (3.8% better R² score)

### Additional Techniques

- **ARIMA Forecasting:** Time series prediction for diabetes prevalence trends
- **K-means Clustering:** Segmented counties into 3 health clusters
- **Cross-Validation:** 5-fold CV ensuring model robustness
- **Feature Engineering:** Created composite health indices

---

## 🔍 Key Findings

### Health Insights
- **Mobility challenges** are the strongest predictor of diabetes prevalence
- Strong correlation between mental health (depression) and physical health outcomes
- Geographic clustering shows distinct health profiles across US regions

### Population Patterns
- Most census tracts: 2,000-5,000 residents
- High-population outliers concentrated in urban centers
- Mississippi has highest average obesity prevalence
- Colorado has lowest average obesity prevalence

### Predictive Accuracy
- Random Forest achieves 98% accuracy in predicting health outcomes
- Model can forecast diabetes prevalence trends with high confidence
- Feature importance analysis identifies intervention opportunities


## 📊 Results

### Statistical Highlights

**Descriptive Statistics:**
- Mean Total Population: 4,268
- Median Diabetes Prevalence: 11.4%
- Standard Deviation (Obesity): 5.99%

**Correlation Findings:**
- Diabetes ↔ Obesity: r = 0.72
- Depression ↔ Sleep Issues: r = 0.68
- Mobility ↔ Self-care: r = 0.81

### Business Impact

This analysis enables:
- **Targeted public health interventions** based on geographic clustering
- **Resource allocation** prioritizing high-risk communities
- **Predictive modeling** for future health trends
- **Evidence-based policy making** for health equity initiatives




