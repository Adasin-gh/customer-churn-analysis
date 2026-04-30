# Customer Churn Prediction

## Project Overview

This project analyzes and predicts telecom customer churn using Python.
The goal is to identify patterns that help businesses understand why customers leave and predict future churn using Machine Learning models.

---

## Dataset

**Telco Customer Churn Dataset** (Kaggle)

The dataset contains information about telecom customers such as:
- Customer demographics
- Services subscribed
- Contract types
- Monthly charges
- Customer tenure
- Churn status (Target Variable)

---

## Tools & Technologies

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- VS Code

---

## Steps Performed

### 1. Data Loading
Loaded the telecom churn dataset using Pandas.

### 2. Data Cleaning
- Checked for missing values
- Converted TotalCharges to numeric
- Handled missing/null values

### 3. Exploratory Data Analysis (EDA)
Analyzed patterns in the dataset using visualizations to understand churn behavior.

### 4. Customer Segmentation
Segmented customers based on tenure groups:
- 0–12 months (New customers)
- 13–36 months (Mid-term customers)
- 37+ months (Long-term customers)

### 5. Advanced Analysis
Analyzed churn behavior based on:
- Contract type
- Payment method
- Internet service type

### 6. Machine Learning — Churn Prediction
Built and compared 3 classification models:

| Model | Task |
|-------|------|
| Logistic Regression | Baseline classifier |
| Random Forest | Ensemble tree model |
| XGBoost | Gradient boosting model |

**Evaluation Metrics:** Accuracy, Recall, ROC-AUC Score

---

## Model Results

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression | ~80% | ~55% | ~0.84 |
| Random Forest | ~79% | ~47% | ~0.83 |
| **XGBoost** ✅ | **~80%** | **~55%** | **~0.85** |

> XGBoost achieves the best ROC-AUC score, making it the best model for churn prediction.

---

## Visualizations

### EDA Charts (graphs/ folder)
- `churn_distribution.png` — Overall churn vs no-churn count
- `tenure_distribution.png` — Customer tenure histogram
- `monthly_charges_boxplot.png` — Monthly charges vs churn
- `tenure_pie_chart.png` — Customer distribution by tenure group
- `tenure_bar_chart.png` — Average monthly charges by tenure
- `churn_by_contract.png` — Churn rate by contract type
- `churn_by_payment.png` — Churn rate by payment method
- `churn_by_internet.png` — Churn rate by internet service

### ML Charts
- `ml_model_results.png` — Accuracy, Recall, ROC-AUC comparison, ROC curves, Confusion matrix, Feature importance

---

## Key Insights

- Customers with **month-to-month contracts** have the highest churn rate (~42%)
- Customers with **Fiber optic internet** churn more than DSL users
- Customers with **higher monthly charges** are more likely to churn
- Customers with **longer tenure** (37+ months) tend to stay with the company
- **Electronic check** payment method has the highest churn among payment types
- **XGBoost** is the best performing model with highest ROC-AUC score

---

## How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
python3 analysis.py
```

---

## Project Structure

```
customer-churn-analysis/
│
├── analysis.py                           # Main script (EDA + ML)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
├── cleaned_customer_churn.csv            # Cleaned dataset
├── README.md
└── graphs/
    ├── churn_distribution.png
    ├── tenure_distribution.png
    ├── monthly_charges_boxplot.png
    ├── tenure_pie_chart.png
    ├── tenure_bar_chart.png
    ├── churn_by_contract.png
    ├── churn_by_payment.png
    ├── churn_by_internet.png
    └── ml_model_results.png              # ML results chart
```

---

## Author

**Adarsh Pratap Singh**
Aspiring Data Analyst | Python | Machine Learning | Data Analysis

GitHub: https://github.com/Adasin-gh

## Tableau Dashboard
Interactive dashboard available here:
https://public.tableau.com/app/profile/adarsh.singh6059/viz/CustomerChurnAnalysisDashboard_17735624507950/CustomerChurnAnalysisDashboard?publish=yes