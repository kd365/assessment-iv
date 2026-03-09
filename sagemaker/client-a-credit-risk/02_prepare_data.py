"""
Client A - Financial Services: Credit Risk Data Preparation & Feature Engineering
=========================================================== 
This script takes the raw credit data from 01_explore_data.py and:
1. Cleans data quality issues found during exploration
2. Engineers new features that capture credit risk signals
3. Formats everything for SageMaker XGBoost (target first, no headers)
4. Splits into train/validation sets with stratification
 
SageMaker XGBoost requirements:
- CSV format, no headers, no index
- Target column MUST be the first column
- All values must be numeric (no strings)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1. Load the raw data (output from 01_explore_data.py)
raw_data_path = "data/credit_default_raw.csv"
df = pd.read_csv(raw_data_path)

# 2. Data Cleaning
# During data exploration, we found some issues:
# - Inconsistent values in 'EDUCATION' and 'MARRIAGE'

# Fix 'EDUCATION' inconsistencies (e.g., 0, 5, 6 -> 4 for 'Other')
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

# Fix 'MARRIAGE' inconsistencies (e.g., 0 -> 3 for 'Unknown')
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

# 3. Feature Engineering
# Create new features based on domain knowledge, creating new features based on business rationale.

# Example 1: Credit Utilization Ratio = BILL_AMT / LIMIT_BAL

# Formula: BILL_AMT / LIMIT_BAL (capped at 1.0 since bills can exceed limit)
for i in range(1, 7):
    col_name = f'UTIL_RATIO_{i}'
    df[col_name] = df[f'BILL_AMT{i}'] / df['LIMIT_BAL'].replace(0, np.nan)
    df[col_name] = df[col_name].clip(upper=2.0).fillna(0)  # Cap extreme values
 
# Example 2: Payment to Bill Ratio = PAY_AMT / BILL_AMT (capped at 1.0 since payments can exceed bills)
for i in range(1, 7):
    col_name = f'PAY_RATIO_{i}'
    bill = df[f'BILL_AMT{i}'].replace(0, np.nan)
    df[col_name] = (df[f'PAY_AMT{i}'] / bill).clip(upper=2.0).fillna(0)
 
df['AVG_PAY_RATIO'] = df[[f'PAY_RATIO_{i}' for i in range(1, 7)]].mean(axis=1)

# Example 3: Payment Delay Summary Stats
# Aggregated Payment columns
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1) # Worst month late
df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1) # Average delay across months
df['NUM_MONTHS_DELAY'] = (df[pay_cols] > 0).sum(axis=1) # How many months late

# Payment Trend: Is the customer improving or worsening?
# Positive trend = improving, Negative trend = worsening
df['PAY_TREND'] = df['PAY_0'] - df['PAY_6']

# Example 4: Balance Change Trend
# Outstanding balance growing or shrinking over time?
df['BALANCE_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
df['BALANCE_GROWTH'] = df['BALANCE_TREND'] / df['LIMIT_BAL'].replace(0, np.nan)
df['BALANCE_GROWTH'] = df['BALANCE_GROWTH'].fillna(0).clip(-2.0, 2.0)

# Example 4: Log Transform of Monetary Features
# Accounting for the skewed distribution of monetary features such as balance limits where
# some customers have very high limits while most have lower limits. Log transformation can help
# reduce the impact of outliers and make the distribution more normal, which can improve model performance
# Using log1p (log(1+x)) to handle zero values safely for calculations.

monetary_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + \
                [f'PAY_AMT{i}' for i in range(1, 7)]
for col in monetary_cols:
    df[f'{col}_LOG'] = np.log1p(df[col].clip(lower=0))
 

# 4. SELECT FINAL FEATURE SET
# Drop only the intermediate utilization/payment ratios per month 
# since we have the averages, keeping the dataset manageable.
df = df.drop(columns=[col for col in df.columns if col.startswith(('UTIL_RATIO_', 'PAY_RATIO_'))])
 

target_col = 'DEFAULT'

# All columns except the target
feature_cols = [col for col in df.columns if col != target_col]
print(f"Feature count: {len(feature_cols)}")
print(f"Features: {feature_cols}")
 
# 5. FORMAT FOR SAGEMAKER XGBOOST
# SageMaker XGBoost requires:
# - Target column FIRST (same pattern as sage/02_prepare_data.py)
# - CSV with no headers, no index
# - All numeric values

cols = [target_col] + feature_cols
df_final = df[cols]
 
print(f"Final shape: {df_final.shape}")
print(f"First column (target): {df_final.columns[0]}")
print(f"Target distribution: {df_final[target_col].value_counts().to_dict()}")
 
# 6. Train/Validation Split

# 80/20 split with STRATIFICATION on the target to maintain class balance in both sets.

train_df, val_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final[target_col])  # Maintain class balance in both splits

# 7. SAVE PREPARED DATA

os.makedirs('prepared_data', exist_ok=True)
 
train_df.to_csv('prepared_data/train.csv', index=False, header=False)
val_df.to_csv('prepared_data/validation.csv', index=False, header=False)
 
# Also save with headers for debugging/reference (not uploaded to SageMaker)
train_df.to_csv('prepared_data/train_with_headers.csv', index=False, header=True)

# Save feature count for the FastAPI service validation
feature_count = len(feature_cols)
with open('prepared_data/feature_count.txt', 'w') as f:
    f.write(str(feature_count))

# Save scale_pos_weight for evaluation within the training script
target_counts = df_final[target_col].value_counts() # Count of records falling into each target class variable
scale_pos_weight = target_counts[0] / target_counts[1] # Divides the majority class count (no default: 0) by the minority class count (default: 1) to get the imbalance ratio
with open('prepared_data/scale_pos_weight.txt', 'w') as f:
    f.write(f"{scale_pos_weight:.4f}")
print(f"Saved: prepared_data/scale_pos_weight.txt ({scale_pos_weight:.4f})")

# Scale weight when fed into XGBoost as a hyperparameter
# will tell the model to treat each positive (default) case
# as more important than each negative (non-default) case, 
# helping to address the class imbalance issue and improve 
# model performance on the minority class.

# SUMMARY
print("\n" + "=" * 60)
print("PREPARATION SUMMARY")
print("=" * 60)
print(f"""
Original features: 23
Engineered features: {feature_count - 23}
Total features: {feature_count}
Training records: {len(train_df):,}
Validation records: {len(val_df):,}
Target class balance preserved: Yes (stratified split)
scale_pos_weight: {scale_pos_weight:.4f}
 
Files ready for SageMaker upload:
  - prepared_data/train.csv
  - prepared_data/validation.csv
 
Next step: 03_train_model.py (upload to S3 + launch SageMaker training job)
""")
 