# Customer Churn Analysis

## Project Overview

This project analyzes telecom customer churn using Python.
The goal is to identify patterns that help businesses understand why customers leave and how to improve customer retention.

---

## Dataset

Telco Customer Churn Dataset (Kaggle)

The dataset contains information about telecom customers such as:

* Customer demographics
* Services subscribed
* Contract types
* Monthly charges
* Customer tenure
* Churn status

---

## Tools & Technologies

* Python
* Pandas
* Matplotlib
* Seaborn
* VS Code

---

## Steps Performed

### 1. Data Loading

Loaded the telecom churn dataset using Pandas.

### 2. Data Cleaning

* Checked for missing values
* Converted TotalCharges to numeric
* Handled missing values

### 3. Exploratory Data Analysis (EDA)

Analyzed patterns in the dataset using visualizations.

### 4. Customer Segmentation

Segmented customers based on tenure groups:

* 0–12 months
* 13–36 months
* 37+ months

### 5. Advanced Analysis

Analyzed churn behavior based on:

* Contract type
* Payment method
* Internet service

---

## Visualizations

The following graphs were generated in the project:

* Churn Distribution
* Tenure Distribution
* Monthly Charges vs Churn
* Customer Distribution by Tenure
* Average Monthly Charges by Tenure Group
* Churn by Contract Type
* Churn by Payment Method
* Churn by Internet Service

---

## Key Insights

* Customers with **month-to-month contracts** have the highest churn rate.
* Customers with **higher monthly charges** are more likely to churn.
* Customers with **longer tenure** tend to stay with the company.

---

## Project Structure

customer-churn-analysis
│
├── analysis.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── graphs
  ├── churn_distribution.png
  ├── tenure_distribution.png
  ├── monthly_charges_boxplot.png
  ├── tenure_pie_chart.png
  ├── tenure_bar_chart.png
  ├── churn_by_contract.png
  ├── churn_by_payment.png
  └── churn_by_internet.png

---

## Author

Adarsh Pratap Singh
Aspiring Data Analyst | Python | Data Analysis

https://github.com/Adasin-gh