import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("=" * 80)
print("RESTAURANT CUISINE CLASSIFICATION PROJECT")
print("=" * 80)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("\n[STEP 1] LOADING DATA...")
try:
    df = pd.read_csv('Dataset.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: Dataset file not found!")
    print("  Please ensure 'Dataset.csv' is in the same directory.")
    exit(1)

print(f"\n  Columns: {list(df.columns)}")

print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

print("\nFirst 5 rows:")
print(df.head())

print("\n\nData Types:")
print(df.dtypes)

print("\n\nMissing Values:")
missing = df.isnull().sum()
missing_df = missing[missing > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values found!")

print("\n\nBasic Statistics:")
print(df.describe())

print("\n\nCuisine Column Sample:")
print(df['Cuisines'].head(10))

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

df_processed = df.copy()

# Step 2.1: Handle missing values in Cuisines column
print("\n[STEP 2.1] Handling missing values in Cuisines...")
print(f"  Missing cuisines before: {df_processed['Cuisines'].isnull().sum()}")
df_processed = df_processed.dropna(subset=['Cuisines'])
print(f"  Missing cuisines after: {df_processed['Cuisines'].isnull().sum()}")
print(f"  Rows remaining: {len(df_processed)}")

# Step 2.2: Process cuisines - take primary cuisine (first one listed)
print("\n[STEP 2.2] Processing cuisine labels...")
print("  Extracting primary cuisine from multi-cuisine entries...")

df_processed['Primary_Cuisine'] = df_processed['Cuisines'].apply(
    lambda x: x.split(',')[0].strip() if isinstance(x, str) else x
)

print("  Sample transformations:")
for i in range(min(5, len(df_processed))):
    orig = df_processed['Cuisines'].iloc[i]
    new = df_processed['Primary_Cuisine'].iloc[i]
    print(f"    '{orig}' → '{new}'")

# Step 2.3: Filter cuisines with sufficient samples
print("\n[STEP 2.3] Filtering cuisines with sufficient samples...")
cuisine_counts = df_processed['Primary_Cuisine'].value_counts()
print(f"  Total unique cuisines: {len(cuisine_counts)}")
print("\n  Top 20 cuisines:")
print(cuisine_counts.head(20))

min_samples = 50
valid_cuisines = cuisine_counts[cuisine_counts >= min_samples].index
df_processed = df_processed[df_processed['Primary_Cuisine'].isin(valid_cuisines)]

print(f"\n  Cuisines with at least {min_samples} samples: {len(valid_cuisines)}")
print(f"  Rows remaining: {len(df_processed)}")

print(f"\n  Final cuisine distribution:")
final_dist = df_processed['Primary_Cuisine'].value_counts()
print(final_dist)

# Step 2.4: Handle missing values in features
print("\n[STEP 2.4] Handling missing values in features...")
num_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
for col in num_cols:
    missing_count = df_processed[col].isnull().sum()
    if missing_count > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"  {col}: {missing_count} missing → filled with median {median_val}")

cat_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
for col in cat_cols:
    missing_count = df_processed[col].isnull().sum()
    if missing_count > 0:
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"  {col}: {missing_count} missing → filled with mode '{mode_val}'")

# Step 2.5: Encode categorical variables
print("\n[STEP 2.5] Encoding categorical variables...")

# Binary encoding for Yes/No columns
binary_map = {'Yes': 1, 'No': 0}
for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].map(binary_map)
        print(f"  {col}: encoded as binary (Yes=1, No=0)")

# Encode country and city
le_country = LabelEncoder()
le_city = LabelEncoder()

df_processed['Country_Encoded'] = le_country.fit_transform(df_processed['Country Code'])
df_processed['City_Encoded'] = le_city.fit_transform(df_processed['City'])

print(f"  Country Code: encoded {len(le_country.classes_)} unique countries")
print(f"  City: encoded {len(le_city.classes_)} unique cities")

# Step 2.6: Select features for modeling
print("\n[STEP 2.6] Selecting features for modeling...")
feature_cols = [
    'Country_Encoded', 'City_Encoded', 'Longitude', 'Latitude',
    'Average Cost for two', 'Has Table booking', 'Has Online delivery',
    'Is delivering now', 'Switch to order menu', 'Price range',
    'Aggregate rating', 'Votes'
]

X = df_processed[feature_cols]
y = df_processed['Primary_Cuisine']

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Target variable shape: {y.shape}")
print(f"  Number of classes: {y.nunique()}")

print("\n  Features selected:")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i}. {col}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X)*100):.1f}%)")
print(f"  Testing set: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X)*100):.1f}%)")

print(f"\n  Training set cuisine distribution:")
train_dist = y_train.value_counts()
print(train_dist)

# Feature scaling
print("\n  Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Feature scaling completed")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, 
                                             multi_class='multinomial', n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                           n_jobs=-1, max_depth=20),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15)
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\n[Training {name}]")
    print(f"  Training in progress...", end=" ")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    trained_models[name] = model
    results[name] = y_pred
    
    print(f"✓ Completed")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

evaluation_results = {}

for name, y_pred in results.items():
    print(f"\n{'=' * 60}")
    print(f"{name.upper()}")
    print('=' * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    evaluation_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 6. PERFORMANCE COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame(evaluation_results).T
comparison_df = comparison_df.round(4)
print("\n", comparison_df)

best_model_name = comparison_df['Accuracy'].idxmax()
best_accuracy = comparison_df.loc[best_model_name, 'Accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   Precision: {comparison_df.loc[best_model_name, 'Precision']:.4f}")
print(f"   Recall: {comparison_df.loc[best_model_name, 'Recall']:.4f}")
print(f"   F1-Score: {comparison_df.loc[best_model_name, 'F1-Score']:.4f}")

# ============================================================================
# 7. PER-CUISINE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PER-CUISINE PERFORMANCE ANALYSIS")
print("=" * 80)

best_model_pred = results[best_model_name]

cuisines = sorted(y_test.unique())
per_cuisine_metrics = []

for cuisine in cuisines:
    mask = y_test == cuisine
    if mask.sum() > 0:
        cuisine_accuracy = accuracy_score(y_test[mask], best_model_pred[mask])
        
        # Calculate precision and recall for this specific cuisine
        true_positive = ((y_test[mask] == cuisine) & (best_model_pred[mask] == cuisine)).sum()
        false_positive = ((y_test != cuisine) & (best_model_pred == cuisine)).sum()
        false_negative = ((y_test == cuisine) & (best_model_pred != cuisine)).sum()
        
        cuisine_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        cuisine_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        per_cuisine_metrics.append({
            'Cuisine': cuisine,
            'Test_Samples': mask.sum(),
            'Accuracy': cuisine_accuracy,
            'Precision': cuisine_precision,
            'Recall': cuisine_recall
        })

per_cuisine_df = pd.DataFrame(per_cuisine_metrics)
per_cuisine_df = per_cuisine_df.sort_values('Accuracy', ascending=False)

print("\nPer-Cuisine Performance (sorted by accuracy):")
print("=" * 80)
print(per_cuisine_df.to_string(index=False))

# ============================================================================
# 8. CHALLENGES AND BIASES IDENTIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("CHALLENGES AND BIASES IDENTIFIED")
print("=" * 80)

print("\n1. CUISINES WITH LOW PERFORMANCE (Accuracy < 50%)")
print("-" * 60)
low_perf = per_cuisine_df[per_cuisine_df['Accuracy'] < 0.5]
if len(low_perf) > 0:
    print(low_perf[['Cuisine', 'Test_Samples', 'Accuracy']].to_string(index=False))
    print(f"\n   ⚠️  {len(low_perf)} cuisine(s) have accuracy below 50%")
    print("   → These cuisines may have overlapping features with other cuisines")
else:
    print("   ✓ No cuisines with accuracy below 50%")

print("\n2. CUISINES WITH MODERATE PERFORMANCE (Accuracy 50-70%)")
print("-" * 60)
med_perf = per_cuisine_df[(per_cuisine_df['Accuracy'] >= 0.5) & (per_cuisine_df['Accuracy'] < 0.7)]
if len(med_perf) > 0:
    print(med_perf[['Cuisine', 'Test_Samples', 'Accuracy']].to_string(index=False))
    print(f"\n   ⚠️  {len(med_perf)} cuisine(s) have moderate accuracy (50-70%)")
else:
    print("   ✓ No cuisines in this range")

print("\n3. CLASS IMBALANCE ANALYSIS")
print("-" * 60)
train_dist = y_train.value_counts()
print(f"   Most common cuisine: {train_dist.index[0]} ({train_dist.iloc[0]} samples)")
print(f"   Least common cuisine: {train_dist.index[-1]} ({train_dist.iloc[-1]} samples)")
imbalance_ratio = train_dist.iloc[0] / train_dist.iloc[-1]
print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 5:
    print(f"\n   ⚠️  SIGNIFICANT class imbalance detected! ({imbalance_ratio:.2f}x)")
    print("   → Consider using SMOTE, class weights, or stratified sampling")
elif imbalance_ratio > 3:
    print(f"\n   ⚠️  Moderate class imbalance detected ({imbalance_ratio:.2f}x)")
else:
    print("\n   ✓ Classes are relatively balanced")

print("\n4. SAMPLE SIZE ANALYSIS")
print("-" * 60)
low_sample = per_cuisine_df[per_cuisine_df['Test_Samples'] < 20]
if len(low_sample) > 0:
    print("   Cuisines with very few test samples (< 20):")
    print(low_sample[['Cuisine', 'Test_Samples', 'Accuracy']].to_string(index=False))
    print(f"\n   ⚠️  {len(low_sample)} cuisine(s) may be underrepresented")
    print("   → Results for these cuisines may not be reliable")
else:
    print("   ✓ All cuisines have adequate test samples")

print("\n5. FEATURE IMPORTANCE ANALYSIS")
print("-" * 60)
if best_model_name == 'Random Forest':
    rf_model = trained_models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n   Key Insights:")
    top_feature = feature_importance.iloc[0]
    print(f"   • {top_feature['Feature']} is the most important feature ({top_feature['Importance']:.4f})")
    
    if feature_importance.iloc[0]['Importance'] > 0.3:
        print("   ⚠️  Model heavily relies on a single feature - may indicate bias")
elif best_model_name == 'Logistic Regression':
    print("   Feature importance analysis not available for Logistic Regression")
else:
    print("   Feature importance analysis not available for this model type")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create output directory for figures
import os
os.makedirs('results', exist_ok=True)

# Visualization 1: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar plot of metrics
comparison_df.plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=best_accuracy, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Best: {best_accuracy:.3f}')

# Confusion Matrix
cm = confusion_matrix(y_test, best_model_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
            xticklabels=cuisines, yticklabels=cuisines, 
            cbar_kws={'label': 'Count'})
axes[1].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
axes[1].set_xlabel('Predicted Cuisine', fontsize=11)
axes[1].set_ylabel('Actual Cuisine', fontsize=11)
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/model_evaluation.png")

# Visualization 2: Per-Cuisine Performance
fig, ax = plt.subplots(figsize=(14, max(8, len(per_cuisine_df) * 0.4)))
per_cuisine_df_sorted = per_cuisine_df.sort_values('Accuracy')
bars = ax.barh(per_cuisine_df_sorted['Cuisine'], per_cuisine_df_sorted['Accuracy'])

# Color bars by performance
colors = []
for acc in per_cuisine_df_sorted['Accuracy']:
    if acc < 0.5:
        colors.append('#e74c3c')  # Red
    elif acc < 0.7:
        colors.append('#f39c12')  # Orange
    elif acc < 0.85:
        colors.append('#3498db')  # Blue
    else:
        colors.append('#2ecc71')  # Green

for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Cuisine', fontsize=12, fontweight='bold')
ax.set_title('Per-Cuisine Classification Accuracy', fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Poor (< 0.5)')
ax.axvline(x=0.7, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair (< 0.7)')
ax.axvline(x=0.85, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (< 0.85)')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim([0, 1])

# Add value labels
for i, (idx, row) in enumerate(per_cuisine_df_sorted.iterrows()):
    ax.text(row['Accuracy'] + 0.02, i, f"{row['Accuracy']:.3f}", 
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/per_cuisine_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/per_cuisine_performance.png")

# Visualization 3: Feature Importance (if Random Forest)
if best_model_name == 'Random Forest':
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance_sorted = feature_importance.sort_values('Importance')
    
    bars = ax.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Importance'])
    bars[0].set_color('#3498db')
    
    for bar in bars:
        bar.set_color('#3498db')
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/feature_importance.png")

plt.show()

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save comparison results
comparison_df.to_csv('results/model_comparison.csv')
print("✓ Saved: results/model_comparison.csv")

# Save per-cuisine metrics
per_cuisine_df.to_csv('results/per_cuisine_metrics.csv', index=False)
print("✓ Saved: results/per_cuisine_metrics.csv")

# Save classification report
with open('results/classification_report.txt', 'w') as f:
    f.write(f"Restaurant Cuisine Classification - Results\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_accuracy:.4f}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write("-" * 80 + "\n")
    f.write(classification_report(y_test, best_model_pred, zero_division=0))
print("✓ Saved: results/classification_report.txt")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 SUMMARY:")
print(f"   • Dataset: {len(df)} restaurants")
print(f"   • After preprocessing: {len(df_processed)} restaurants")
print(f"   • Number of cuisines: {y.nunique()}")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Testing samples: {len(X_test)}")
print(f"   • Features used: {len(feature_cols)}")

print(f"\n🏆 BEST MODEL RESULTS:")
print(f"   • Model: {best_model_name}")
print(f"   • Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   • Precision: {comparison_df.loc[best_model_name, 'Precision']:.4f}")
print(f"   • Recall: {comparison_df.loc[best_model_name, 'Recall']:.4f}")
print(f"   • F1-Score: {comparison_df.loc[best_model_name, 'F1-Score']:.4f}")

print(f"\n📁 OUTPUT FILES:")
print(f"   • results/model_evaluation.png")
print(f"   • results/per_cuisine_performance.png")
if best_model_name == 'Random Forest':
    print(f"   • results/feature_importance.png")
print(f"   • results/model_comparison.csv")
print(f"   • results/per_cuisine_metrics.csv")
print(f"   • results/classification_report.txt")

print("\n" + "=" * 80)
print("Thank you for using the Cuisine Classification System!")
print("=" * 80)