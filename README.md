# 🔍 Employee Segmentation & Attrition Analysis

> **Unsupervised Machine Learning pipeline** that discovers natural employee behavioral segments from the IBM HR dataset and links them to attrition risk — enabling HR to move from reactive exit interviews to proactive, segment-targeted retention strategies.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Findings](#-key-findings)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Visualizations](#-visualizations)
- [Statistical Validation](#-statistical-validation)
- [Technologies Used](#-technologies-used)

---

## 🎯 Project Overview

Employee attrition costs organizations **50–200% of an employee's annual salary** per departure (SHRM). With a 16.1% attrition rate across 1,470 employees, that's **237 departures per cycle** — a significant financial burden.

This project applies **unsupervised learning** (zero labeled data) to automatically discover natural employee segments purely from behavioral data, then links those segments to attrition outcomes. The result is a data-driven HR strategy that targets retention resources where they matter most.

### What Makes This Project Different

| Feature | Typical ML Project | This Project |
|---------|-------------------|--------------|
| Approach | Single K-Means run | **Two-round design** — Round 1 failure → diagnosis → Round 2 correction |
| Validation | Visual inspection | **Silhouette + Kruskal-Wallis + Bootstrap stability** |
| Comparison | None | **Agglomerative clustering cross-validation** (ARI = 0.1075) |
| Anomaly Detection | Not attempted | **4 methods** (Z-score, IQR, Isolation Forest, CBLOF) linked to attrition |
| Output | Academic findings | **Actionable HR interventions** with $2.83M quantified savings |

---

## 🏆 Key Findings

| Segment | Employees | Avg Age | Avg Income | Experience | Attrition | Risk |
|---------|-----------|---------|------------|------------|-----------|------|
| 🟢 Experienced Loyal | 287 (19.5%) | 45.1 | $13,349 | 22.8 yr | **8.7%** | LOW |
| 🟡 Mid-Level Moderate | 275 (18.7%) | 36.1 | $5,284 | 9.5 yr | **11.3%** | LOW |
| 🟠 Mid-Career Moderate | 506 (34.4%) | 35.5 | $4,720 | 8.3 yr | **13.4%** | MODERATE |
| 🔴 Young At-Risk | 402 (27.3%) | 33.5 | $4,693 | 8.0 yr | **28.1%** | HIGH |

> **Critical Insight:** The Young At-Risk segment (402 employees) has a **3.2× higher attrition rate** than Experienced Loyal. Targeted intervention on this segment alone could save an estimated **$2.83M annually** (56 retained × $50K replacement cost).

---

## 🏗️ Pipeline Architecture

```
┌─────────────┐    ┌──────────┐    ┌───────────────┐    ┌─────────────┐
│  load_data  │───▶│   eda    │───▶│ preprocessing │───▶│  clustering │
│  Section 1  │    │Sections  │    │  Sections 3-5 │    │Sections 6-9 │
│             │    │  2a, 2b  │    │               │    │  R1 → R2    │
└─────────────┘    └──────────┘    └───────────────┘    └──────┬──────┘
                                                              │
         ┌────────────────────────────────────────────────────┘
         ▼
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│cluster_analysis │───▶│ anomaly_detection    │───▶│   hierarchical   │
│   Section 10    │    │     Section 12       │    │    _clustering    │
│  Deep EDA       │    │ Z, IQR, IF, CBLOF   │    │   Section 13     │
└────────┬────────┘    └──────────────────────┘    └────────┬─────────┘
         │                                                  │
         ▼                                                  ▼
┌─────────────────┐                               ┌──────────────────┐
│recommendations  │                               │   statistical    │
│   Section 11    │                               │   _validation    │
│  HR Actions     │                               │   Section 14     │
└─────────────────┘                               │  KW + Bootstrap  │
                                                  └──────────────────┘
```

**Orchestrator:** `main.py` runs all modules in sequence and passes data between them.

---

## 📊 Dataset

**IBM HR Employee Attrition Dataset**
- **Source:** IBM Watson Analytics Sample Dataset
- **Size:** 1,470 employees × 35 features
- **Quality:** Zero null values, zero duplicate rows
- **Target:** Attrition column (used for post-hoc analysis only — NOT for clustering)

| Category | Features | Count |
|----------|----------|-------|
| Demographics | Age, Gender, MaritalStatus | 5 |
| Job Info | JobRole, Department, JobLevel | 8 |
| Compensation | MonthlyIncome, HourlyRate | 4 |
| Satisfaction | JobSatisfaction, WorkLifeBalance | 4 |
| Performance | PerformanceRating, OverTime | 3 |
| Tenure | TotalWorkingYears, YearsAtCompany | 5 |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis across all 35 features
- Attrition breakdown by department, job role, overtime, marital status
- Cohen's d effect sizes: TotalWorkingYears (0.465), JobLevel (0.460), MonthlyIncome (0.435)
- OverTime is the strongest risk factor: **30.5% vs 10.4%** attrition rate

### 2. Preprocessing
- **Drop** 4 zero-variance/ID columns → 31 features
- **Encode** 3 binary + 5 one-hot categorical columns → 49 features
- **Scale** with StandardScaler (μ=0.000, σ=1.000) — mandatory for Euclidean distance

### 3. Two-Round Clustering Design
- **Round 1:** K-Means on all 49 features → clusters mirror org chart (silhouette = 0.1062)
- **Diagnosis:** 12 department/job role dummies dominate Euclidean distances
- **Round 2:** Remove 12 dummies → 37 features → genuine behavioral segments (silhouette = 0.0689)
- **k selection:** Elbow method + silhouette analysis → k=4 chosen for interpretability

### 4. Anomaly Detection (4 Methods)
| Method | Anomalies | Attrition Rate | vs Normal |
|--------|-----------|---------------|-----------|
| Z-Score (threshold=3) | 21 | 19.0% | 1.2× |
| IQR | 86 | 17.4% | 1.1× |
| Isolation Forest | 74 | **20.3%** | **1.3×** |
| CBLOF | 107 | 15.9% | 1.0× |

### 5. Cross-Algorithm Validation
- Agglomerative (Ward linkage) vs K-Means comparison
- **ARI = 0.1075** — weak agreement → both found valid but different structures
- K-Means achieved higher silhouette and more balanced cluster sizes

### 6. Statistical Validation
- **Kruskal-Wallis H-test:** 5/8 features significant at p < 0.001
- **Effect sizes:** η² up to 0.4005 (TotalWorkingYears) — LARGE
- **Bootstrap stability:** Silhouette = 0.0659 ± 0.0096 across 20 resamples

---

## 📈 Results

### Cluster Profiles
- **Experienced Loyal** (Cluster 0): Senior veterans with highest income ($13,349), longest tenure (22.8yr), lowest attrition (8.7%)
- **Mid-Level Moderate** (Cluster 1): Mid-range employees, moderate risk (11.3% attrition)
- **Mid-Career Moderate** (Cluster 2): Largest group (34.4%), moderate attrition (13.4%)
- **Young At-Risk** (Cluster 3): Youngest, lowest paid, **highest attrition (28.1%)**

### Business Impact
- HIGH RISK segment: 402 employees × 28.1% = **~113 departures/cycle**
- 50% reduction through targeted intervention → 56 retained
- At $50K replacement cost → **$2.83M annual savings**

### HR Recommendations (Segment-Specific)
- **Young At-Risk:** Compensation review, overtime caps, structured mentorship
- **Mid-Career Moderate:** Role enrichment, lateral movement, recognition programs
- **Experienced Loyal:** Maintain current policies, leverage as mentors

---

## 📁 Project Structure

```
employee-attrition-unsupervised/
│
├── main.py                          # Pipeline orchestrator — runs all modules
├── load_data.py                     # Section 1: Data loading & initial exploration
├── eda.py                           # Section 2: Exploratory Data Analysis (6 plots)
├── preprocessing.py                 # Sections 3-5: Clean → Encode → Scale
├── clustering.py                    # Sections 6-9: K-Means R1, R2, PCA, validation
├── cluster_analysis.py              # Section 10: Deep EDA on final segments
├── recommendations.py               # Section 11: HR recommendations
├── anomaly_detection.py             # Section 12: Z-score, IQR, IF, CBLOF
├── hierarchical_clustering.py       # Section 13: Agglomerative + dendrogram
├── statistical_validation.py        # Section 14: Kruskal-Wallis + bootstrap
├── employee_segmentation.py         # Monolithic version (all sections in one file)
│
├── hr_employee_attrition.csv        # IBM HR dataset (1,470 × 35)
│
├── plot_eda_*.png                   # EDA visualizations (6 plots)
├── plot_clustering_*.png            # Clustering visualizations (4 plots)
├── plot_analysis_*.png              # Segment analysis plots (3 plots)
├── plot_anomaly_*.png               # Anomaly detection plots (5 plots)
├── plot_hierarchical_*.png          # Hierarchical clustering plots (2 plots)
├── plot_validation_*.png            # Statistical validation plots (2 plots)
│
├── .gitignore                       # Excludes __pycache__
└── README.md                        # This file
```

---

## ⚙️ Installation & Usage

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run the Full Pipeline
```bash
python main.py
```

This executes all 14 sections sequentially:
1. Loads and explores the dataset
2. Generates 6 EDA plots
3. Preprocesses and scales features
4. Runs K-Means Round 1 (demonstrates failure)
5. Runs K-Means Round 2 (behavioral segments)
6. Deep EDA on discovered segments
7. Generates HR recommendations
8. Runs 4-method anomaly detection
9. Compares with Agglomerative clustering
10. Performs statistical validation (KW + Bootstrap)

**Output:** 22 publication-quality PNG plots + console analysis

---

## 📸 Visualizations

The pipeline generates **22 plots** across 6 categories:

| Category | Plots | Key Insight |
|----------|-------|-------------|
| EDA | 6 | OverTime workers leave at 30.5% (3× normal rate) |
| Clustering | 4 | Round 1 failure → Round 2 success with 4 segments |
| Segment Analysis | 3 | $8,656 income gap between top and bottom segments |
| Anomaly Detection | 5 | Isolation Forest anomalies leave at 1.3× normal rate |
| Hierarchical | 2 | ARI = 0.1075 — both methods find valid structures |
| Validation | 2 | η² = 0.4005 — clusters explain 40% of experience variance |

---

## 📐 Statistical Validation

| Feature | H-statistic | p-value | η² | Effect Size |
|---------|-------------|---------|-----|-------------|
| TotalWorkingYears | 590.17 | 1.4 × 10⁻¹²⁷ | **0.4005** | LARGE |
| MonthlyIncome | 503.88 | 6.9 × 10⁻¹⁰⁹ | **0.3417** | LARGE |
| YearsAtCompany | 385.35 | 3.3 × 10⁻⁸³ | **0.2608** | LARGE |
| Age | 292.39 | 4.4 × 10⁻⁶³ | **0.1974** | LARGE |
| YearsSinceLastPromotion | 272.57 | 8.6 × 10⁻⁵⁹ | **0.1839** | LARGE |

**Bootstrap Stability:** Silhouette = 0.0659 ± 0.0096 (95% CI: [0.0474, 0.0753]) across 20 resamples — clusters are stable, not random artifacts.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | K-Means, PCA, Silhouette, Isolation Forest, Agglomerative |
| **matplotlib** | Publication-quality visualizations |
| **seaborn** | Statistical plotting |
| **SciPy** | Kruskal-Wallis H-test, statistical functions |

---

<p align="center">
  <b>Built with ❤️ for Unsupervised Machine Learning</b><br>
  <i>IBM HR Employee Attrition Dataset · K-Means · PCA · Statistical Validation</i>
</p>
