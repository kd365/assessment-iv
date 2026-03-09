"""
Client B - Outdoor Recreation: Park Accessibility & Feasibility Clustering
============================================================
This script loads and explores the Fairfax County parks dataset.
We need to understand the data structure, feature distributions,
and geographic spread before clustering.

Business context: Our client runs a trip-planning app and wants
to rank park locations by accessibility and feasibility scores.
We use K-Means clustering to group parks with similar characteristics,
then score each cluster on these dimensions.

Dataset: Fairfax County Park Authority (public ArcGIS API)
Features: Location (lat/lon), amenities, facilities, activities
Model: K-Means clustering (unsupervised — no target variable)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# 1. LOAD THE DATASET
# ============================================================

with open('/Users/kathleenhill/aico-delta_Fall2025/assessment-iv/sagemaker/client-b-park-clustering/data/parks_raw.json', 'r') as f:
    parks = json.load(f)
print(f"Loaded {len(parks)} parks")

# Convert to DataFrame for easier analysis
df = pd.DataFrame(parks)

# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================


print(f"\nDataset shape: {df.shape[0]} records x {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nBasic statistics:")
print(df.describe())


# ============================================================
# 3. Classification Distribution Analysis
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Classification Distribution")
print("=" * 60)

# How many parks of each type?
print(df['classification'].value_counts())

# City distribution
print(f"\nParks by city (top 10):")
print(df['city'].value_counts().head(10))


# ============================================================
# 4. FEATURE ANALYSIS
# ============================================================
# How many parks have playgrounds, restrooms, trails, on-site parking
# How many are dog-friendly
# Distribution of best_for categories
# Distribution of special_features counts

print("\n" + "=" * 60)
print("STEP 4: Amenity Feature Analysis")
print("=" * 60)

# Extract amenity fields from the nested JSON
# Each park has an 'amenities' dict — we need to count availability

has_playground = sum(1 for p in parks if p['amenities'].get('playground', 'No') != 'No')
has_restrooms = sum(1 for p in parks if p['amenities'].get('restrooms', 'No') == 'Yes')
has_trails = sum(1 for p in parks if p['amenities'].get('trails', 'None') != 'None')
has_onsite_parking = sum(1 for p in parks if 'On-Site' in p['amenities'].get('parking', ''))
has_dog_friendly = sum(1 for p in parks if 'Yes' in p['amenities'].get('dog_friendly', ''))
has_picnic = sum(1 for p in parks if p['amenities'].get('picnic_shelters', 'No') != 'No')

total = len(parks)
print(f"\nAmenity availability ({total} parks):")
print(f"  Playground:      {has_playground} ({has_playground/total*100:.0f}%)")
print(f"  Restrooms:       {has_restrooms} ({has_restrooms/total*100:.0f}%)")
print(f"  Trails:          {has_trails} ({has_trails/total*100:.0f}%)")
print(f"  On-site parking: {has_onsite_parking} ({has_onsite_parking/total*100:.0f}%)")
print(f"  Dog friendly:    {has_dog_friendly} ({has_dog_friendly/total*100:.0f}%)")
print(f"  Picnic shelters: {has_picnic} ({has_picnic/total*100:.0f}%)")

# best_for category distribution
from collections import Counter
all_best_for = []
for p in parks:
    all_best_for.extend(p.get('best_for', []))
best_for_counts = Counter(all_best_for)
print(f"\nMost common 'best_for' categories:")
for category, count in best_for_counts.most_common(10):
    print(f"  {category}: {count}")

# special_features count per park
feature_counts = [len(p['amenities'].get('special_features', [])) for p in parks]
print(f"\nSpecial features per park:")
print(f"  Min: {min(feature_counts)}, Max: {max(feature_counts)}, Avg: {np.mean(feature_counts):.1f}")



# ============================================================
# 5. Geographic ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Geographic Analysis")
print("=" * 60)

lats = [p['latitude'] for p in parks if p.get('latitude')]
lons = [p['longitude'] for p in parks if p.get('longitude')]

print(f"\nLatitude range:  {min(lats):.4f} to {max(lats):.4f}")
print(f"Longitude range: {min(lons):.4f} to {max(lons):.4f}")
print(f"Centroid: ({np.mean(lats):.4f}, {np.mean(lons):.4f})")
print(f"Parks with coordinates: {len(lats)} / {len(parks)}")


# ============================================================
# 6. SAVE VISUALIZATIONS
# ============================================================
# Generate plots that tell the story of this dataset.
# These are useful for your presentation and documentation.
print("\n" + "=" * 60)
print("STEP 6: Generating Visualizations")
print("=" * 60)

os.makedirs('data/plots', exist_ok=True)

# Plot 1: Park locations scatter plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(lons, lats, alpha=0.6, c='#3498db')
ax.set_title('Park Locations — Fairfax County', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('data/plots/park_locations.png', dpi=150)
print("  Saved: data/plots/park_locations.png")

# Plot 2: Classification distribution
fig, ax = plt.subplots(figsize=(8, 5))
df['classification'].value_counts().plot(kind='bar', ax=ax, color='#2ecc71')
ax.set_title('Park Classification Types', fontsize=14)
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/plots/classification_distribution.png', dpi=150)
print("  Saved: data/plots/classification_distribution.png")

# Plot 3: Amenity availability rates
fig, ax = plt.subplots(figsize=(10, 5))
amenities = ['Playground', 'Restrooms', 'Trails', 'Parking', 'Dog Friendly', 'Picnic']
rates = [has_playground/total, has_restrooms/total, has_trails/total,
         has_onsite_parking/total, has_dog_friendly/total, has_picnic/total]
ax.barh(amenities, rates, color='#3498db')
ax.set_title('Amenity Availability Rate', fontsize=14)
ax.set_xlabel('Proportion of Parks')
plt.tight_layout()
plt.savefig('data/plots/amenity_rates.png', dpi=150)
print("  Saved: data/plots/amenity_rates.png")

# Plot 4: Top best_for categories
fig, ax = plt.subplots(figsize=(10, 5))
top_categories = best_for_counts.most_common(8)
ax.barh([c[0] for c in top_categories], [c[1] for c in top_categories], color='#e67e22')
ax.set_title('Most Common Park Activities', fontsize=14)
ax.set_xlabel('Number of Parks')
plt.tight_layout()
plt.savefig('data/plots/best_for_categories.png', dpi=150)
print("  Saved: data/plots/best_for_categories.png")

plt.close('all')


print("\n" + "=" * 60)
print("EXPLORATION SUMMARY")
print("=" * 60)
print(f"""
Dataset: Fairfax County Parks (Public ArcGIS API)
Total parks: {len(parks)}
Classifications: {df['classification'].nunique()} types
Parks with coordinates: {len(lats)} / {len(parks)}

Amenity highlights:
  {has_restrooms} parks have restrooms ({has_restrooms/total*100:.0f}%)
  {has_trails} parks have trails ({has_trails/total*100:.0f}%)
  {has_onsite_parking} parks have on-site parking ({has_onsite_parking/total*100:.0f}%)

Key observations:
1. Unsupervised task — no target variable, using K-Means clustering
2. Features will focus on accessibility (parking, restrooms, trails)
   and feasibility (facilities, activities, capacity)
3. Geographic coordinates available for location-based clustering
4. Amenity data is nested JSON — needs flattening in preparation step

Next step: 02_prepare_data.py (feature engineering + K-Means formatting)
""")
