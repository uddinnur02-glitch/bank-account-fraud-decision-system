# Bank Account Fraud Detection & Risk-Based Review Decision System

An end-to-end machine learning project for detecting fraudulent bank-account applications and translating model predictions into an operational **risk-based manual review strategy**.

The project combines temporal validation, class-imbalance handling, model benchmarking, probability calibration, review-capacity optimization, risk segmentation, SHAP explainability, cost-sensitive decision analysis, reusable inference, and Streamlit deployment.

The final system is designed as a **decision-support tool for fraud investigators**, not as an automatic approval or rejection engine.

---

## Project Overview

Fraud detection is not simply a classification problem. In an operational environment, fraud is rare, investigation capacity is limited, false positives consume resources, missed fraud can be costly, raw model scores may not represent reliable probabilities, and decisions need to be explainable.

This project asks:

> **How can a machine learning fraud model identify and prioritize the highest-risk applications for manual investigation while controlling review workload?**

---

## Business Objective

The system is designed to:

1. estimate application-level fraud risk,
2. rank applications from highest to lowest risk,
3. identify a manageable subset for manual review,
4. measure fraud capture under different review capacities,
5. provide interpretable risk segments,
6. explain individual predictions,
7. support cost-sensitive investigation policies.

---

## Dataset

This project uses the **Bank Account Fraud (BAF) dataset**, introduced as part of the NeurIPS 2022 dataset work by Feedzai.

The dataset is **synthetic / privacy-preserving**, generated from anonymized real-world banking data.

### Dataset characteristics

- Rows: **1,000,000**
- Original columns: **32**
- Target: `fraud_bool`
- Fraud cases: **11,029**
- Non-fraud cases: **988,971**
- Overall fraud prevalence: **~1.10%**
- Temporal periods: months `0–7`

| Class | Count | Percentage |
|---|---:|---:|
| Non-fraud | 988,971 | 98.90% |
| Fraud | 11,029 | 1.10% |

Because of the severe class imbalance, the project emphasizes PR-AUC, ROC-AUC, precision, recall, F1, Brier score, review-capacity recall, and lift.

---

## Project Architecture

```text
Raw Application Data
        ↓
Data Quality Audit
        ↓
Sentinel / Missing-Value Handling
        ↓
Feature Engineering
        ↓
Temporal Train / Calibration / Strategy / Test Split
        ↓
Model Benchmarking
        ↓
Hyperparameter Tuning
        ↓
Probability Calibration
        ↓
Review-Capacity Analysis
        ↓
Risk Segmentation
        ↓
SHAP Explainability
        ↓
Business Decision Policy
        ↓
Saved Model Artifacts
        ↓
Streamlit Decision-Support Application
```

---

## Repository Structure

```text
bank-account-fraud-decision-system/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── catboost_fraud_model.cbm
│   ├── feature_schema.json
│   ├── model_metadata.json
│   ├── platt_calibrator.joblib
│   └── risk_thresholds.json
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_and_signal_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing_and_baseline_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_threshold_calibration_and_review_strategy.ipynb
│   ├── 07_explainability_and_risk_segmentation.ipynb
│   ├── 08_business_impact_and_decision_policy.ipynb
│   └── 09_model_persistence_and_inference.ipynb
│
├── reports/
│   └── figures/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Data Quality Audit

The audit examined dataset structure, target imbalance, duplicate rows, categorical cardinality, hidden missing-value patterns, sentinel values, temporal fraud-rate drift, suspicious negative values, and constant features.

The following variables used `-1` as unavailable-information sentinels:

- `prev_address_months_count`
- `current_address_months_count`
- `bank_months_count`
- `session_length_in_minutes`
- `device_distinct_emails_8w`

For each of these features, a missingness indicator was created and sentinel values were converted to missing values.

`device_fraud_count` was removed because it was constant.

Some negative values in other fields were retained because they appeared to be valid values rather than missing-value codes.

---

# 2. Temporal Fraud Drift

Fraud prevalence changed across time:

| Month | Fraud Rate |
|---:|---:|
| 0 | 1.13% |
| 1 | 0.94% |
| 2 | 0.87% |
| 3 | 0.92% |
| 4 | 1.14% |
| 5 | 1.18% |
| 6 | 1.34% |
| 7 | 1.47% |

A temporal validation framework was used instead of a random split.

---

# 3. Temporal Evaluation Strategy

| Period | Purpose |
|---|---|
| Months 0–4 | Model training |
| Month 5 | Probability calibration |
| Month 6 | Review-policy / threshold development |
| Month 7 | Final out-of-time test |

Month 7 was not used to refit the model, recalibrate probabilities, or modify the frozen review threshold.

---

# 4. Feature Engineering

The project created several deterministic features:

### Missing-information features
- individual missingness indicators,
- `missing_info_count`.

### Contact-quality features
- `valid_phone_count`.

### Address-history feature
- `address_history_complete`.

### Heuristic risk indicators
- `low_identity_consistency`,
- `high_credit_request`.

The final modeling dataset contained **40 predictor variables**.

---

# 5. Exploratory Signal Analysis

Meaningful fraud-related structure was observed across:

- device operating system,
- housing status,
- employment status,
- payment type,
- application source,
- address history,
- phone validity,
- card ownership,
- customer age,
- credit-risk score,
- name-email similarity,
- missingness patterns.

---

# 6. Baseline Model Benchmarking

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.8827 | 0.1591 |
| Random Forest | 0.8522 | 0.1280 |
| LightGBM | 0.8887 | 0.1726 |
| XGBoost | 0.8909 | 0.1764 |
| CatBoost | **0.8916** | **0.1809** |

CatBoost produced the strongest baseline PR-AUC and was selected as the primary model family.

---

# 7. Hyperparameter Tuning

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| CatBoost | 0.8942 | **0.1861** |
| XGBoost | **0.8952** | 0.1820 |
| LightGBM | 0.8913 | 0.1758 |

Final CatBoost configuration:

```text
depth = 5
learning_rate = 0.05
iterations = 631
loss_function = Logloss
auto_class_weights = Balanced
random_seed = 42
```

---

# 8. Probability Calibration

Before calibration:

- actual month-6 fraud rate: **1.34%**
- mean raw CatBoost score: **22.73%**

Platt scaling was fitted using month 5.

| Metric | Raw | Calibrated |
|---|---:|---:|
| Mean predicted probability | 22.73% | **1.36%** |
| Actual fraud rate | — | **1.34%** |
| Brier score | 0.1041 | **0.0119** |
| PR-AUC | 0.1850 | 0.1850 |
| ROC-AUC | 0.8921 | 0.8921 |

Calibration improved probability reliability while preserving ranking performance.

---

# 9. Review-Capacity Strategy

| Review Capacity | Precision | Fraud Recall | Lift |
|---:|---:|---:|---:|
| 1% | 27.36% | 20.41% | 20.41× |
| 2% | 20.84% | 31.10% | 15.55× |
| 5% | 13.48% | **50.28%** | **10.05×** |
| 10% | 8.87% | 66.21% | 6.62× |
| 20% | 5.46% | 81.52% | 4.08× |

A **5% review-capacity policy** was selected as the primary portfolio operating point.

---

# 10. Frozen Risk Thresholds

| Risk Band | Calibrated Probability |
|---|---:|
| Low | < 1.27% |
| Medium | 1.27% to < 5.82% |
| High | 5.82% to < 19.18% |
| Critical | ≥ 19.18% |

Exact thresholds:

```text
Medium:   0.0126736516
High:     0.0582048411
Critical: 0.1917505869
Review:   0.0582048411
```

---

# 11. Risk Segmentation

### Month-6 strategy period

| Risk Band | Application Share | Fraud Rate |
|---|---:|---:|
| Low | 80.00% | 0.31% |
| Medium | 15.00% | 2.79% |
| High | 4.00% | 10.01% |
| Critical | 1.00% | 27.36% |

The Critical segment had approximately **88×** the fraud rate of the Low segment.

---

# 12. Out-of-Time Risk Segmentation

| Risk Band | Application Share | Fraud Rate |
|---|---:|---:|
| Low | 85.31% | 0.42% |
| Medium | 11.22% | 3.81% |
| High | 2.78% | 15.42% |
| Critical | 0.69% | 37.99% |

The Critical segment had roughly **91×** the fraud rate of the Low segment.

---

# 13. Final Out-of-Time Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 0.9643 |
| ROC-AUC | **0.8959** |
| PR-AUC | **0.2180** |
| Precision | **19.89%** |
| Recall | **46.85%** |
| F1 | **0.2793** |
| Brier Score | **0.0129** |
| Actual Review Rate | **3.47%** |

Confusion matrix:

```text
              Predicted
              0       1

Actual 0    92721    2694
Actual 1      759     669
```

The frozen threshold identified **669 of 1,428 fraud cases** while reviewing only **3.47%** of month-7 applications.

---

# 14. Fixed-Capacity Out-of-Time Performance

| Review Capacity | Precision | Fraud Recall | Lift |
|---:|---:|---:|---:|
| 1% | 32.82% | 22.27% | 22.26× |
| 2% | 25.40% | 34.45% | 17.23× |
| 5% | **15.86%** | **53.78%** | **10.75×** |
| 10% | 10.03% | 68.00% | 6.80× |
| 20% | 6.04% | 81.86% | 4.09× |

At fixed **5% review capacity**, the model captured **53.78%** of month-7 fraud.

---

# 15. Explainability

Global SHAP analysis identified important model drivers including:

- `device_os`,
- `has_other_cards`,
- `keep_alive_session`,
- `current_address_months_count`,
- `email_is_free`,
- `housing_status`,
- `name_email_similarity`,
- `phone_home_valid`,
- `month`,
- `income`.

SHAP values explain model behavior, not causal relationships.

---

# 16. Local Prediction Explanations

The Streamlit application generates a local SHAP explanation for each submitted application.

The interface displays:

- top risk-increasing features,
- top risk-decreasing features,
- SHAP values,
- individual feature values,
- a visual contribution chart.

---

# 17. Business Decision Policies

| Policy | Review Capacity | Fraud Recall | Precision | Lift |
|---|---:|---:|---:|---:|
| Conservative | 1% | 20.41% | 27.36% | 20.41× |
| Balanced | 5% | 50.28% | 13.48% | 10.05× |
| Aggressive | 10% | 66.21% | 8.87% | 6.62× |

The Balanced policy was retained as the primary portfolio demonstration point.

---

# 18. Illustrative Cost-Sensitive Analysis

Illustrative assumptions:

```text
Review cost = 5 units

Missed fraud cost:
Low      = 100 units
Moderate = 500 units
High     = 1000 units
```

Under these assumptions:

- Balanced performed best under the low fraud-cost scenario.
- Aggressive performed better when missed-fraud cost increased.

These values are hypothetical and are **not real banking cost estimates**.

---

# 19. Investigation Efficiency

| Policy | Review Capacity | Fraud Recall per 1% Reviewed |
|---|---:|---:|
| Conservative | 1% | 20.41 |
| Balanced | 5% | 10.06 |
| Aggressive | 10% | 6.62 |

This demonstrates diminishing returns as review capacity expands.

---

# 20. Streamlit Decision-Support Application

The Streamlit interface accepts full application information and returns:

- calibrated fraud probability,
- risk band,
- manual-review recommendation,
- raw model score,
- local SHAP explanation,
- top risk-increasing factors,
- top risk-decreasing factors.

Derived ML features are generated automatically inside the app.

---

# 21. Model Persistence and Reproducible Inference

Saved artifacts:

```text
catboost_fraud_model.cbm
platt_calibrator.joblib
risk_thresholds.json
model_metadata.json
feature_schema.json
```

Inference pipeline:

```text
Raw application
      ↓
Feature engineering
      ↓
Exact feature ordering
      ↓
CatBoost
      ↓
Raw score
      ↓
Platt calibration
      ↓
Calibrated probability
      ↓
Risk segmentation
      ↓
Manual-review recommendation
```

A consistency check confirmed:

```text
Absolute difference = 0.0
```

between the original notebook result and the reloaded deployment pipeline.

---

# 22. Installation

```bash
git clone <YOUR-REPOSITORY-URL>
cd bank-account-fraud-decision-system
python -m venv .venv
pip install -r requirements.txt
```

---

# 23. Run the Streamlit App

```bash
streamlit run app/app.py
```

or:

```bash
python -m streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

---

# 24. Requirements

```text
pandas
numpy
scikit-learn
matplotlib
jupyter
ipykernel
pyarrow
xgboost
lightgbm
catboost
shap
joblib
streamlit
```

---

# 25. Methodological Strengths

- temporal rather than random evaluation,
- explicit treatment of extreme class imbalance,
- PR-AUC as the primary ranking metric,
- multi-model benchmarking,
- probability calibration,
- fixed-threshold out-of-time testing,
- review-capacity evaluation,
- lift analysis,
- risk segmentation,
- local and global explainability,
- cost-sensitive framing,
- reproducible model persistence,
- deployment consistency testing.

---

# 26. Limitations

### Synthetic / privacy-preserving dataset
The dataset is not raw production banking data.

### Hyperparameter-validation reuse
Month 6 contributed to earlier model-selection work and later review-strategy development. A stricter workflow could use nested temporal validation.

### Feature-engineering exploration
Some heuristic features were informed by exploratory analysis.

### No real economic data
The project does not include actual fraud losses, investigation costs, or staffing constraints.

### Temporal drift
Future deployment would require monitoring of score drift, feature drift, fraud prevalence, calibration, workload, precision, and recall.

### Decision support only
The system should not automatically approve or reject financial applications.

---

# 27. Future Improvements

- rolling temporal cross-validation,
- nested temporal model selection,
- drift monitoring,
- automated recalibration,
- expected-loss modeling,
- institution-specific investigation costs,
- fairness analysis,
- SHAP monitoring,
- model registry,
- API deployment,
- Docker,
- CI/CD,
- database integration,
- real-time scoring,
- investigator feedback loops.

---

# 28. Key Takeaways

Final headline results:

```text
ROC-AUC:            0.8959
PR-AUC:             0.2180
Precision:          19.89%
Recall:             46.85%
F1:                 0.2793
Brier Score:        0.0129
Frozen Review Rate: 3.47%
```

At fixed **5% review capacity**:

```text
Fraud recall: 53.78%
Lift:         10.75×
```

The project demonstrates how a machine-learning classifier can be transformed into an interpretable **fraud-risk prioritization and decision-support system**.

---

## Tech Stack

**Language:** Python

**Data processing:** pandas, NumPy, PyArrow

**Machine learning:** scikit-learn, CatBoost, XGBoost, LightGBM

**Explainability:** SHAP, CatBoost SHAP values

**Deployment:** Streamlit, joblib, JSON metadata

**Development:** Jupyter Notebook, Git, GitHub

---

## Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes.

It should not be used for real financial approval, rejection, fraud accusation, or other consequential banking decisions without appropriate validation, governance, fairness assessment, regulatory review, and human oversight.
