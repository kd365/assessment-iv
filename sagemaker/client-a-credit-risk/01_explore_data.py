"""
Client A - Financial Services: Credit Risk Data Exploration
============================================================
Dataset: UCI Default of Credit Card Clients (Taiwan, 2005)
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

This script downloads and explores the credit card default dataset.
We need to understand the data's shape, distributions, class balance,
and feature relationships before we do any preprocessing or modeling.

Business context: Our client (a financial services company) wants to
predict which customers will default on their credit card payment next
month, so they can adjust their loan approval workflow accordingly.

Dataset details:
This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: 
X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;
X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). 
X12 = amount of bill statement in September, 2005; 
X13 = amount of bill statement in August, 2005; . . .; 
X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). 
X18 = amount paid in September, 2005; 
X19 = amount paid in August, 2005; . . .;
X23 = amount paid in April, 2005.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os

# ============================================================
# 1. DOWNLOAD THE DATASET
# ============================================================
# The ucimlrepo package fetches directly from the UCI ML Repository.
# Dataset ID 350 = "Default of Credit Card Clients"
# This is a well-known benchmark dataset for credit scoring.

print("=" * 60)
print("STEP 1: Downloading dataset from UCI ML Repository...")
print("=" * 60)

dataset = fetch_ucirepo(id=350)

# The dataset object has .data.features (X) and .data.targets (y)
X = dataset.data.features  # 30,000 rows x 23 features
y = dataset.data.targets    # 30,000 rows x 1 target column

# IMPORTANT: ucimlrepo returns generic column names (X1-X23, Y) instead of
# the descriptive names from the original dataset. We need to map them back
# using the dataset documentation. This is a common real-world issue — always
# verify your column names match what you expect from the data dictionary.
#
# Original dataset columns (from UCI documentation):
# X1=LIMIT_BAL, X2=SEX, X3=EDUCATION, X4=MARRIAGE, X5=AGE,
# X6-X11=PAY_0 through PAY_6 (repayment status, past 6 months),
# X12-X17=BILL_AMT1 through BILL_AMT6 (bill statement amount),
# X18-X23=PAY_AMT1 through PAY_AMT6 (previous payment amount),
# Y=DEFAULT (target: 1=default, 0=no default)
COLUMN_RENAME = {
    'X1': 'LIMIT_BAL',     # Credit limit (NT dollars)
    'X2': 'SEX',           # 1=male, 2=female
    'X3': 'EDUCATION',     # 1=grad school, 2=university, 3=high school, 4=others
    'X4': 'MARRIAGE',      # 1=married, 2=single, 3=others
    'X5': 'AGE',           # Age in years
    'X6': 'PAY_0',         # Repayment status in Sep 2005 (most recent)
    'X7': 'PAY_2',         # Repayment status in Aug 2005
    'X8': 'PAY_3',         # Repayment status in Jul 2005
    'X9': 'PAY_4',         # Repayment status in Jun 2005
    'X10': 'PAY_5',        # Repayment status in May 2005
    'X11': 'PAY_6',        # Repayment status in Apr 2005
    'X12': 'BILL_AMT1',    # Bill statement amount in Sep 2005
    'X13': 'BILL_AMT2',    # Bill statement amount in Aug 2005
    'X14': 'BILL_AMT3',    # Bill statement amount in Jul 2005
    'X15': 'BILL_AMT4',    # Bill statement amount in Jun 2005
    'X16': 'BILL_AMT5',    # Bill statement amount in May 2005
    'X17': 'BILL_AMT6',    # Bill statement amount in Apr 2005
    'X18': 'PAY_AMT1',     # Payment amount in Sep 2005
    'X19': 'PAY_AMT2',     # Payment amount in Aug 2005
    'X20': 'PAY_AMT3',     # Payment amount in Jul 2005
    'X21': 'PAY_AMT4',     # Payment amount in Jun 2005
    'X22': 'PAY_AMT5',     # Payment amount in May 2005
    'X23': 'PAY_AMT6',     # Payment amount in Apr 2005
    'Y': 'DEFAULT',        # Target: 1=default next month, 0=no default
}

X = X.rename(columns=COLUMN_RENAME)
y = y.rename(columns=COLUMN_RENAME)

# Combine into a single DataFrame for exploration
df = pd.concat([X, y], axis=1)

print(f"Renamed {len(COLUMN_RENAME)} columns from generic (X1..Y) to descriptive names")

# Save raw data locally so we don't need to re-download
os.makedirs('data', exist_ok=True)
df.to_csv('data/credit_default_raw.csv', index=False)
print(f"Dataset saved to data/credit_default_raw.csv")

# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================
# Before doing anything fancy, look at the raw data.
# Key questions: How many records? How many features? Any nulls?
# What types are the columns?

print("\n" + "=" * 60)
print("STEP 2: Basic Data Inspection")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]} records x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nBasic statistics:")
print(df.describe())

# Check for missing values - critical before modeling
# SageMaker XGBoost can handle missing values, but we should know about them
print(f"\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# ============================================================
# 3. TARGET VARIABLE ANALYSIS (CLASS IMBALANCE)
# ============================================================
# Credit default prediction is almost always imbalanced - most people
# DON'T default. We need to know the ratio so we can:
# 1. Set XGBoost's scale_pos_weight hyperparameter
# 2. Choose appropriate evaluation metrics (AUC, not accuracy)
# 3. Decide if we need stratified sampling

print("\n" + "=" * 60)
print("STEP 3: Target Variable (Class Imbalance) Analysis")
print("=" * 60)

target_col = 'DEFAULT'
target_counts = df[target_col].value_counts()
target_pct = df[target_col].value_counts(normalize=True) * 100

print(f"\nTarget distribution:")
print(f"  No default (0): {target_counts[0]:,} ({target_pct[0]:.1f}%)")
print(f"  Default    (1): {target_counts[1]:,} ({target_pct[1]:.1f}%)")

# Calculate scale_pos_weight for XGBoost
# This tells XGBoost to pay more attention to the minority class (defaults)
# Formula: count(negative) / count(positive)
scale_pos_weight = target_counts[0] / target_counts[1]
print(f"\n  scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
print(f"  (This means the model should weight default cases {scale_pos_weight:.1f}x more)")

# ============================================================
# 4. FEATURE ANALYSIS
# ============================================================
# Understanding each feature helps us decide what to keep, transform,
# or engineer. The dataset has:
# - LIMIT_BAL: Credit limit (continuous)
# - SEX: 1=male, 2=female (categorical)
# - EDUCATION: 1=grad school, 2=university, 3=high school, 4=others
# - MARRIAGE: 1=married, 2=single, 3=others
# - AGE: Age in years (continuous)
# - PAY_0 to PAY_6: Repayment status for past 6 months
#     (-1=pay duly, 1=1 month delay, 2=2 months delay, etc.)
# - BILL_AMT1 to BILL_AMT6: Bill statement amounts for 6 months
# - PAY_AMT1 to PAY_AMT6: Previous payment amounts for 6 months

print("\n" + "=" * 60)
print("STEP 4: Feature Analysis")
print("=" * 60)

# Categorical features - check value distributions
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts().sort_index())

# Check for unexpected values in EDUCATION and MARRIAGE
# The documentation says EDUCATION should be 1-4, MARRIAGE should be 1-3
# But some records have 0, 5, 6 - these are likely data entry errors
print(f"\nEDUCATION unexpected values (0, 5, 6):")
print(f"  Count: {df[df['EDUCATION'].isin([0, 5, 6])].shape[0]}")

print(f"\nMARRIAGE unexpected values (0):")
print(f"  Count: {df[df['MARRIAGE'] == 0].shape[0]}")

# Payment history features - these are the strongest predictors
# Negative values mean no consumption/no payment due
# 0 means revolving credit usage
# 1+ means months of payment delay
print(f"\nPAY_0 (most recent month) distribution:")
print(df['PAY_0'].value_counts().sort_index())

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
# Which features are most correlated with the target?
# This helps us understand what drives credit default.

print("\n" + "=" * 60)
print("STEP 5: Correlation with Default Target")
print("=" * 60)

correlations = df.corr()[target_col].sort_values(ascending=False)
print(f"\nTop positive correlations (increase default risk):")
print(correlations.head(10))
print(f"\nTop negative correlations (decrease default risk):")
print(correlations.tail(5))

# ============================================================
# 6. SAVE VISUALIZATIONS
# ============================================================
# Generate plots that tell the story of this dataset.
# These are useful for your presentation and documentation.

print("\n" + "=" * 60)
print("STEP 6: Generating Visualizations")
print("=" * 60)

os.makedirs('data/plots', exist_ok=True)

# Plot 1: Class distribution
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['No Default (0)', 'Default (1)'],
              [target_counts[0], target_counts[1]],
              color=colors)
ax.set_title('Credit Card Default Distribution\n(Class Imbalance)', fontsize=14)
ax.set_ylabel('Count')
for bar, pct in zip(bars, [target_pct[0], target_pct[1]]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
            f'{pct:.1f}%', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('data/plots/class_distribution.png', dpi=150)
print("  Saved: data/plots/class_distribution.png")

# Plot 2: Correlation heatmap with target
fig, ax = plt.subplots(figsize=(10, 8))
top_features = correlations.drop(target_col).abs().sort_values(ascending=False).head(12).index.tolist()
top_features.append(target_col)
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax)
ax.set_title('Top Feature Correlations with Default', fontsize=14)
plt.tight_layout()
plt.savefig('data/plots/correlation_heatmap.png', dpi=150)
print("  Saved: data/plots/correlation_heatmap.png")

# Plot 3: Default rate by payment history
fig, ax = plt.subplots(figsize=(10, 5))
pay0_default = df.groupby('PAY_0')[target_col].mean() * 100
pay0_default.plot(kind='bar', ax=ax, color='#3498db')
ax.set_title('Default Rate by Most Recent Payment Status (PAY_0)', fontsize=14)
ax.set_ylabel('Default Rate (%)')
ax.set_xlabel('Payment Status (-1=On Time, 1+=Months Delayed)')
plt.tight_layout()
plt.savefig('data/plots/default_by_payment_status.png', dpi=150)
print("  Saved: data/plots/default_by_payment_status.png")

# Plot 4: Credit limit distribution by default status
fig, ax = plt.subplots(figsize=(10, 5))
df[df[target_col] == 0]['LIMIT_BAL'].hist(bins=50, alpha=0.6, label='No Default', ax=ax, color='#2ecc71')
df[df[target_col] == 1]['LIMIT_BAL'].hist(bins=50, alpha=0.6, label='Default', ax=ax, color='#e74c3c')
ax.set_title('Credit Limit Distribution by Default Status', fontsize=14)
ax.set_xlabel('Credit Limit (NT$)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('data/plots/credit_limit_distribution.png', dpi=150)
print("  Saved: data/plots/credit_limit_distribution.png")

plt.close('all')

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATION SUMMARY")
print("=" * 60)
print(f"""
Dataset: UCI Default of Credit Card Clients
Records: {df.shape[0]:,}
Features: {df.shape[1] - 1} (excluding target)
Target: {target_col}
Class balance: {target_pct[0]:.1f}% no-default / {target_pct[1]:.1f}% default
Missing values: {df.isnull().sum().sum()}
Recommended scale_pos_weight: {scale_pos_weight:.2f}

Key findings:
1. Dataset is imbalanced (~{target_pct[0]:.0f}/{target_pct[1]:.0f} split)
   -> Use scale_pos_weight in XGBoost and evaluate with AUC, not accuracy
2. PAY_0 (most recent payment status) is the strongest predictor
   -> Payment delay history is critical for credit risk
3. Higher credit limits correlate with LOWER default rates
   -> Banks already screen higher-limit customers more carefully
4. Some EDUCATION and MARRIAGE values are outside documented range
   -> Will clean these in the preparation step

Next step: 02_prepare_data.py (feature engineering + SageMaker formatting)
""")
