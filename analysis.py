import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create graphs folder if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------------
# Task 1 : Understand Dataset
# -----------------------------
print("First 10 rows:")
print(df.head(10))

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Task 2 : Data Cleaning
# -----------------------------

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df = df.fillna(0)

# -----------------------------
# Task 3 : Exploratory Data Analysis
# -----------------------------

# Churn Distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("graphs/churn_distribution.png")
plt.show()

# Tenure Distribution
plt.figure()
plt.hist(df["tenure"], bins=20)
plt.title("Customer Tenure Distribution")
plt.savefig("graphs/tenure_distribution.png")
plt.show()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("graphs/monthly_charges_boxplot.png")
plt.show()

# -----------------------------
# Task 4 : Customer Segmentation
# -----------------------------

df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0,12,36,72],
    labels=['0-12 months','13-36 months','37+ months']
)

# Pie Chart
plt.figure()
df['tenure_group'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Customer Distribution by Tenure")
plt.ylabel("")
plt.savefig("graphs/tenure_pie_chart.png")
plt.show()

# Average Monthly Charges by Tenure
plt.figure()
avg_charges = df.groupby('tenure_group')['MonthlyCharges'].mean()

avg_charges.plot(kind="bar")
plt.title("Average Monthly Charges by Tenure Group")
plt.ylabel("Monthly Charges")
plt.savefig("graphs/tenure_bar_chart.png")
plt.show()

# -----------------------------
# Task 5 : Advanced Analysis
# -----------------------------

# Churn by Contract
plt.figure()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.savefig("graphs/churn_by_contract.png")
plt.show()

# Churn by Payment Method
plt.figure()
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.xticks(rotation=45)
plt.title("Churn by Payment Method")
plt.savefig("graphs/churn_by_payment.png")
plt.show()

# Churn by Internet Service
plt.figure()
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.savefig("graphs/churn_by_internet.png")
plt.show()

df.to_csv("cleaned_customer_churn.csv", index=False)