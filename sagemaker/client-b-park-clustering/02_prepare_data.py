"""
Client B - Park Clustering: Data Preparation & Feature Engineering
Flattens nested JSON into numeric features for SageMaker K-Means.
Two feature groups: Accessibility and Feasibility.

"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler



# 1. Load the raw data (output from 01_explore_data.py)
with open('sagemaker/client-b-park-clustering/data/parks_raw.json', 'r') as f:
    parks = json.load(f)

# 2. Flatten the JSON and create a DataFrame
records = []
for p in parks:
    amenities = p.get('amenities', {})
    record = {
        # Location
        'latitude': p.get('latitude', 0),
        'longitude': p.get('longitude', 0),

        # Accessibility features
        'has_onsite_parking': 1 if 'On-Site' in amenities.get('parking', '') else 0,
        'has_restrooms': 1 if amenities.get('restrooms', 'No') == 'Yes' else 0,
        'has_paved_trails': 1 if 'Paved' in amenities.get('trails', '') else 0,
        'has_any_trails': 0 if amenities.get('trails', 'None') == 'None' else 1,
        'has_playground': 0 if amenities.get('playground', 'No') == 'No' else 1,
        'is_dog_friendly': 1 if 'Yes' in amenities.get('dog_friendly', '') else 0,

        # Feasibility features
        'has_picnic_shelters': 0 if amenities.get('picnic_shelters', 'No') == 'No' else 1,
        'has_water_activities': 1 if any(kw in amenities.get('water_activities', '')
                                        for kw in ['Fishing', 'Swimming', 'Creek']) else 0,
        'num_special_features': len(amenities.get('special_features', [])),
        'num_best_for': len(p.get('best_for', [])),
    }
    records.append(record)

df = pd.DataFrame(records)
print(f"Created DataFrame with shape: {df.shape}")

#3. Normalize numeric features (latitude, longitude, num_special_features, num_best_for)

scaler = MinMaxScaler()
feature_cols = df.columns.tolist()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

#4. Save the processed DataFrame to CSV for SageMaker K-Means
os.makedirs('prepared_data', exist_ok=True)

# SageMaker K-Means expects a label column first (ignored during training)
df.insert(0, 'label', 0)
df.to_csv('prepared_data/train.csv', header=False, index=False)


# SageMaker K-Means: CSV, no headers, no index, all numeric
df.to_csv('prepared_data/train.csv', header=False, index=False)

# Debug version with headers
df.to_csv('prepared_data/train_with_headers.csv', index=False)

# Save feature count (needed for K-Means hyperparameter)
feature_count = len(feature_cols)
with open('prepared_data/feature_count.txt', 'w') as f:
    f.write(str(feature_count))

# Save park names for mapping back after clustering
park_names = [p['park_name'] for p in parks]
with open('prepared_data/park_names.txt', 'w') as f:
    for name in park_names:
        f.write(name + '\n')

#5. Print summary of the prepared dataset

print(f"\nFeature count: {feature_count}")
print(f"Features: {feature_cols}")
print(f"Records: {len(df)}")
print(f"\nSaved: prepared_data/train.csv")
print(f"Saved: prepared_data/feature_count.txt")
print(f"Saved: prepared_data/park_names.txt")
print(f"\nNext step: 03_train_model.py")
