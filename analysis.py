import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# pip install xgboost scikit-learn pandas matplotlib seaborn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                              roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from xgboost import XGBClassifier

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# ─────────────────────────────────────────────
# TASK 1: LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("   CUSTOMER CHURN PREDICTION")
print("=" * 55)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\n[1/6] Dataset Overview:")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# TASK 2: DATA CLEANING
# ─────────────────────────────────────────────
print("\n[2/6] Data Cleaning...")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.fillna(0)
print("   ✓ TotalCharges converted to numeric")
print("   ✓ Missing values filled")

# ─────────────────────────────────────────────
# TASK 3: EDA VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[3/6] Generating EDA Visualizations...")

# Churn Distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("graphs/churn_distribution.png")
plt.close()

# Tenure Distribution
plt.figure()
plt.hist(df["tenure"], bins=20)
plt.title("Customer Tenure Distribution")
plt.savefig("graphs/tenure_distribution.png")
plt.close()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("graphs/monthly_charges_boxplot.png")
plt.close()

# ─────────────────────────────────────────────
# TASK 4: CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────
print("\n[4/6] Customer Segmentation...")

df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 36, 72],
    labels=['0-12 months', '13-36 months', '37+ months']
)

# Pie Chart
plt.figure()
df['tenure_group'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Customer Distribution by Tenure")
plt.ylabel("")
plt.savefig("graphs/tenure_pie_chart.png")
plt.close()

# Bar Chart
plt.figure()
df.groupby('tenure_group')['MonthlyCharges'].mean().plot(kind="bar")
plt.title("Average Monthly Charges by Tenure Group")
plt.ylabel("Monthly Charges")
plt.savefig("graphs/tenure_bar_chart.png")
plt.close()

# ─────────────────────────────────────────────
# TASK 5: ADVANCED ANALYSIS
# ─────────────────────────────────────────────
print("\n[5/6] Advanced Analysis...")

plt.figure()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.savefig("graphs/churn_by_contract.png")
plt.close()

plt.figure()
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.xticks(rotation=45)
plt.title("Churn by Payment Method")
plt.tight_layout()
plt.savefig("graphs/churn_by_payment.png")
plt.close()

plt.figure()
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.savefig("graphs/churn_by_internet.png")
plt.close()

df.to_csv("cleaned_customer_churn.csv", index=False)
print("   ✓ All EDA charts saved in graphs/")

# ─────────────────────────────────────────────
# TASK 6: ML MODELS — CHURN PREDICTION
# ─────────────────────────────────────────────
print("\n[6/6] Machine Learning — Churn Prediction...")

# ── Feature Engineering ──
df_ml = df.copy()
df_ml.drop(['customerID', 'tenure_group'], axis=1, inplace=True)

# Label encode all object columns
le = LabelEncoder()
for col in df_ml.select_dtypes(include='object').columns:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

print("   ✓ Categorical columns encoded")

X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   ✓ Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── Model 1: Logistic Regression ──
print("\n   🔵 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

lr_acc    = accuracy_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_auc    = roc_auc_score(y_test, lr_proba)
print(f"   Accuracy : {lr_acc*100:.2f}%")
print(f"   Recall   : {lr_recall*100:.2f}%")
print(f"   ROC-AUC  : {lr_auc:.4f}")

# ── Model 2: Random Forest ──
print("\n   🟢 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_acc    = accuracy_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_auc    = roc_auc_score(y_test, rf_proba)
print(f"   Accuracy : {rf_acc*100:.2f}%")
print(f"   Recall   : {rf_recall*100:.2f}%")
print(f"   ROC-AUC  : {rf_auc:.4f}")

# ── Model 3: XGBoost ──
print("\n   🟡 Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                     use_label_encoder=False, eval_metric='logloss',
                     verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred  = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

xgb_acc    = accuracy_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_auc    = roc_auc_score(y_test, xgb_proba)
print(f"   Accuracy : {xgb_acc*100:.2f}%")
print(f"   Recall   : {xgb_recall*100:.2f}%")
print(f"   ROC-AUC  : {xgb_auc:.4f}")

# ── Classification Reports ──
print("\n   📋 Logistic Regression Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn','Churn']))
print("   📋 Random Forest Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn','Churn']))
print("   📋 XGBoost Report:")
print(classification_report(y_test, xgb_pred, target_names=['No Churn','Churn']))

# ── ML Visualizations ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Churn Prediction — ML Model Results',
             fontsize=14, fontweight='bold')

# Plot 1: Model Accuracy Comparison
models  = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
accs    = [lr_acc*100, rf_acc*100, xgb_acc*100]
recalls = [lr_recall*100, rf_recall*100, xgb_recall*100]
aucs    = [lr_auc, rf_auc, xgb_auc]
colors  = ['#3498DB', '#2ECC71', '#F39C12']

bars = axes[0,0].bar(models, accs, color=colors)
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim(0, 110)
for bar, val in zip(bars, accs):
    axes[0,0].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.5, f'{val:.1f}%',
                   ha='center', fontweight='bold')

# Plot 2: Recall Comparison
bars2 = axes[0,1].bar(models, recalls, color=colors)
axes[0,1].set_title('Model Recall Comparison')
axes[0,1].set_ylabel('Recall (%)')
axes[0,1].set_ylim(0, 110)
for bar, val in zip(bars2, recalls):
    axes[0,1].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.5, f'{val:.1f}%',
                   ha='center', fontweight='bold')

# Plot 3: ROC-AUC Comparison
bars3 = axes[0,2].bar(models, aucs, color=colors)
axes[0,2].set_title('ROC-AUC Score Comparison')
axes[0,2].set_ylabel('AUC Score')
axes[0,2].set_ylim(0, 1.15)
for bar, val in zip(bars3, aucs):
    axes[0,2].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.005, f'{val:.3f}',
                   ha='center', fontweight='bold')

# Plot 4: ROC Curves
for pred, name, color in zip(
    [lr_proba, rf_proba, xgb_proba],
    ['Logistic Regression', 'Random Forest', 'XGBoost'],
    ['#3498DB', '#2ECC71', '#F39C12']
):
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc_val = roc_auc_score(y_test, pred)
    axes[1,0].plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{name} (AUC={auc_val:.3f})')
axes[1,0].plot([0,1],[0,1],'k--', linewidth=1)
axes[1,0].set_title('ROC Curves')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].legend(fontsize=8)

# Plot 5: Confusion Matrix — Best Model (XGBoost)
best_pred = xgb_pred
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn','Churn'],
            yticklabels=['No Churn','Churn'], ax=axes[1,1])
axes[1,1].set_title(f'Confusion Matrix — XGBoost\n(Accuracy: {xgb_acc*100:.1f}%)')
axes[1,1].set_xlabel('Predicted')
axes[1,1].set_ylabel('Actual')

# Plot 6: Feature Importance (XGBoost)
feat_imp = pd.Series(xgb.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(10)
axes[1,2].barh(feat_imp.index[::-1], feat_imp.values[::-1], color='#F39C12')
axes[1,2].set_title('Top 10 Feature Importances (XGBoost)')
axes[1,2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig("graphs/ml_model_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ ML charts saved as 'graphs/ml_model_results.png'")

# ── FINAL SUMMARY ──
print("\n" + "="*55)
print("              FINAL SUMMARY")
print("="*55)
print(f"{'Model':<22} {'Accuracy':>10} {'Recall':>10} {'ROC-AUC':>10}")
print("-"*55)
print(f"{'Logistic Regression':<22} {lr_acc*100:>9.2f}% {lr_recall*100:>9.2f}% {lr_auc:>10.4f}")
print(f"{'Random Forest':<22} {rf_acc*100:>9.2f}% {rf_recall*100:>9.2f}% {rf_auc:>10.4f}")
print(f"{'XGBoost':<22} {xgb_acc*100:>9.2f}% {xgb_recall*100:>9.2f}% {xgb_auc:>10.4f}")
print("="*55)

best = max([('Logistic Regression', lr_auc),
            ('Random Forest', rf_auc),
            ('XGBoost', xgb_auc)], key=lambda x: x[1])
print(f"\n🏆 Best Model (by ROC-AUC): {best[0]} ({best[1]:.4f})")
print("\n✅ CHURN PREDICTION COMPLETE!")
print("   All charts saved in graphs/ folder")