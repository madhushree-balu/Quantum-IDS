"""
Quantum IDS Project - Data Preprocessing
File: src/01_preprocessing.py
Purpose: Load, clean, and prepare KDD Cup dataset for quantum and classical models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
class Config:
    # Paths
    DATA_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    RESULTS_DIR = 'results'
    FIGURES_DIR = 'results/figures'
    
    # Dataset parameters
    DATASET_FILE = 'kddcup.data_10_percent'
    ATTACK_TYPE = 'neptune.'  # Options: 'neptune.', 'smurf.', 'back.', 'teardrop.'
    
    # Feature selection
    N_FEATURES = 8  # Number of features for quantum encoding (4-10 recommended)
    
    # Dataset sizes
    SAMPLE_SIZE = 2500  # Samples per class (for balanced dataset)
    TEST_SIZE = 0.3
    
    # Create directories
    for dir_path in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

config = Config()

print("="*80)
print("QUANTUM IDS PROJECT - DATA PREPROCESSING")
print("="*80)

# ========================================
# STEP 1: Load Dataset
# ========================================
print("\n[STEP 1] Loading KDD Cup dataset...")

# Column names for KDD Cup dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Load dataset
dataset_path = os.path.join(config.DATA_DIR, config.DATASET_FILE)
df = pd.read_csv(dataset_path, names=column_names)

print(f"✓ Dataset loaded: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts().head(10))

# ========================================
# STEP 2: Binary Classification Setup
# ========================================
print(f"\n[STEP 2] Creating binary classification: Normal vs {config.ATTACK_TYPE}")

# Filter for binary classification
df_binary = df[df['label'].isin(['normal.', config.ATTACK_TYPE])].copy()
print(f"✓ Filtered dataset: {df_binary.shape}")
print(f"  Normal: {sum(df_binary['label'] == 'normal.')}")
print(f"  {config.ATTACK_TYPE}: {sum(df_binary['label'] == config.ATTACK_TYPE)}")

# Encode labels: 0 = normal, 1 = attack
df_binary['label'] = df_binary['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# ========================================
# STEP 3: Feature Encoding
# ========================================
print("\n[STEP 3] Encoding categorical features...")

# Separate features and labels
X = df_binary.drop('label', axis=1).copy()
y = df_binary['label'].copy()

# Encode categorical features
categorical_columns = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✓ Features encoded: {X.shape}")
print(f"✓ Categorical columns: {categorical_columns}")

# ========================================
# STEP 4: Feature Selection
# ========================================
print("\n[STEP 4] Performing feature selection...")

# Sample data for faster feature selection
if len(X) > 5000:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=RANDOM_SEED)
else:
    X_sample, y_sample = X, y

print(f"  Training Random Forest on {len(X_sample)} samples...")

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
rf.fit(X_sample, y_sample)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop {min(15, len(feature_importance))} Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Select top N features for quantum encoding
top_features = feature_importance.head(config.N_FEATURES)['feature'].tolist()
print(f"\n✓ Selected {config.N_FEATURES} features for quantum encoding:")
for i, feat in enumerate(top_features, 1):
    imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
    print(f"  {i}. {feat:30s} (importance: {imp:.4f})")

X_selected = X[top_features].copy()

# Save feature importance
feature_importance.to_csv(os.path.join(config.RESULTS_DIR, 'feature_importance.csv'), index=False)

# ========================================
# STEP 5: Create Balanced Dataset
# ========================================
print("\n[STEP 5] Creating balanced dataset...")

# Separate majority and minority classes
X_normal = X_selected[y == 0]
y_normal = y[y == 0]
X_attack = X_selected[y == 1]
y_attack = y[y == 1]

print(f"  Before balancing - Normal: {len(X_normal)}, Attack: {len(X_attack)}")

# Determine sample size
sample_size = min(len(X_normal), len(X_attack), config.SAMPLE_SIZE)
print(f"  Using {sample_size} samples per class")

# Downsample both classes
X_normal_sampled = resample(X_normal, n_samples=sample_size, random_state=RANDOM_SEED)
y_normal_sampled = resample(y_normal, n_samples=sample_size, random_state=RANDOM_SEED)
X_attack_sampled = resample(X_attack, n_samples=sample_size, random_state=RANDOM_SEED)
y_attack_sampled = resample(y_attack, n_samples=sample_size, random_state=RANDOM_SEED)

# Combine
X_balanced = pd.concat([X_normal_sampled, X_attack_sampled])
y_balanced = pd.concat([y_normal_sampled, y_attack_sampled])

print(f"✓ Balanced dataset: {X_balanced.shape}")
print(f"  Normal: {sum(y_balanced==0)}, Attack: {sum(y_balanced==1)}")

# ========================================
# STEP 6: Train-Test Split
# ========================================
print("\n[STEP 6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, 
    test_size=config.TEST_SIZE, 
    random_state=RANDOM_SEED, 
    stratify=y_balanced
)

print(f"✓ Train set: {X_train.shape} (Normal: {sum(y_train==0)}, Attack: {sum(y_train==1)})")
print(f"✓ Test set:  {X_test.shape} (Normal: {sum(y_test==0)}, Attack: {sum(y_test==1)})")

# ========================================
# STEP 7: Normalization for Classical Models
# ========================================
print("\n[STEP 7] Normalizing features for classical models...")

# Standard scaling for classical models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")
print(f"  Mean: {X_train_scaled.mean(axis=0)[:3]} ...")
print(f"  Std:  {X_train_scaled.std(axis=0)[:3]} ...")

# ========================================
# STEP 8: Normalization for Quantum Models
# ========================================
print("\n[STEP 8] Normalizing features for quantum encoding...")

# Scale to [0, 2π] for quantum circuits
X_train_min = X_train_scaled.min()
X_train_max = X_train_scaled.max()

X_train_quantum = (X_train_scaled - X_train_min) / (X_train_max - X_train_min) * 2 * np.pi
X_test_quantum = (X_test_scaled - X_train_min) / (X_train_max - X_train_min) * 2 * np.pi

print(f"✓ Features scaled to [0, 2π] for quantum encoding")
print(f"  Range: [{X_train_quantum.min():.3f}, {X_train_quantum.max():.3f}]")

# ========================================
# STEP 9: Save Processed Data
# ========================================
print("\n[STEP 9] Saving processed data...")

# Save as numpy arrays
np.save(os.path.join(config.PROCESSED_DIR, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(config.PROCESSED_DIR, 'X_test_scaled.npy'), X_test_scaled)
np.save(os.path.join(config.PROCESSED_DIR, 'X_train_quantum.npy'), X_train_quantum)
np.save(os.path.join(config.PROCESSED_DIR, 'X_test_quantum.npy'), X_test_quantum)
np.save(os.path.join(config.PROCESSED_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(config.PROCESSED_DIR, 'y_test.npy'), y_test.values)

# Save feature names
with open(os.path.join(config.PROCESSED_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(top_features))

# Save configuration
config_dict = {
    'n_features': config.N_FEATURES,
    'attack_type': config.ATTACK_TYPE,
    'sample_size': sample_size,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'feature_names': top_features,
    'random_seed': RANDOM_SEED
}

import json
with open(os.path.join(config.PROCESSED_DIR, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=4)

print(f"✓ Data saved to {config.PROCESSED_DIR}/")

# ========================================
# STEP 10: Visualization
# ========================================
print("\n[STEP 10] Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Feature Distributions
for idx, feature in enumerate(top_features):
    ax = plt.subplot(3, 3, idx+1)
    ax.hist(X_balanced[feature][y_balanced == 0], alpha=0.6, label='Normal', bins=30, color='blue')
    ax.hist(X_balanced[feature][y_balanced == 1], alpha=0.6, label='Attack', bins=30, color='red')
    ax.set_title(f'{feature}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    if idx == 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Feature distributions saved")

# 2. Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 6))
top_15 = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
bars = ax.barh(range(len(top_15)), top_15['importance'], color=colors)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['feature'])
ax.set_xlabel('Importance Score')
ax.set_title('Top 15 Most Important Features')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Highlight selected features
for i, feature in enumerate(top_15['feature']):
    if feature in top_features:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"✓ Feature importance plot saved")

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation = X_balanced[top_features].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✓ Correlation matrix saved")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original distribution
df['label'].value_counts().head(10).plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Original Dataset Distribution')
axes[0].set_xlabel('Count')

# Binary distribution - FIXED VERSION
binary_counts = pd.Series([sum(y==0), sum(y==1)], index=['Normal', 'Attack'])
binary_counts.plot(kind='bar', ax=axes[1], color=['blue', 'red'])
axes[1].set_title('Binary Classification Distribution')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

# Train-Test split
split_data = pd.DataFrame({
    'Normal': [sum(y_train==0), sum(y_test==0)],
    'Attack': [sum(y_train==1), sum(y_test==1)]
}, index=['Train', 'Test'])
split_data.plot(kind='bar', ax=axes[2], color=['blue', 'red'])
axes[2].set_title('Train-Test Split Distribution')
axes[2].set_ylabel('Count')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'class_distributions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Class distribution plots saved")
plt.close('all')

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nDataset Summary:")
print(f"  Total samples:      {len(X_balanced)}")
print(f"  Training samples:   {len(X_train)}")
print(f"  Testing samples:    {len(X_test)}")
print(f"  Number of features: {config.N_FEATURES}")
print(f"  Feature names:      {', '.join(top_features)}")
print(f"  Attack type:        {config.ATTACK_TYPE}")
print(f"\nFiles saved in:")
print(f"  Data:   {config.PROCESSED_DIR}/")
print(f"  Plots:  {config.FIGURES_DIR}/")
print(f"\nNext step: Run 02_classical_baseline.py")
print("="*80)