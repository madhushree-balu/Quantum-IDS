"""
Quantum IDS Project - Enhanced Data Preprocessing
File: src/01_preprocessing_enhanced.py
Purpose: Load, clean, and prepare KDD Cup dataset with flexible attack type handling
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
    
    # ============================================
    # CLASSIFICATION MODE - CHOOSE ONE:
    # ============================================
    # MODE = 'binary'           # Normal vs. ALL attacks combined
    # MODE = 'multiclass'       # Classify top N most common attack types
    MODE = 'category'         # Classify by attack categories (DoS, Probe, R2L, U2R, Normal)
    
    # For multiclass mode: number of top attack types to include (+ normal)
    TOP_N_ATTACKS = 5
    
    # Attack category mappings
    ATTACK_CATEGORIES = {
        'normal.': 'Normal',
        # DoS attacks
        'back.': 'DoS',
        'land.': 'DoS',
        'neptune.': 'DoS',
        'pod.': 'DoS',
        'smurf.': 'DoS',
        'teardrop.': 'DoS',
        # Probe attacks
        'ipsweep.': 'Probe',
        'nmap.': 'Probe',
        'portsweep.': 'Probe',
        'satan.': 'Probe',
        # R2L attacks
        'ftp_write.': 'R2L',
        'guess_passwd.': 'R2L',
        'imap.': 'R2L',
        'multihop.': 'R2L',
        'phf.': 'R2L',
        'spy.': 'R2L',
        'warezclient.': 'R2L',
        'warezmaster.': 'R2L',
        # U2R attacks
        'buffer_overflow.': 'U2R',
        'loadmodule.': 'U2R',
        'perl.': 'U2R',
        'rootkit.': 'U2R'
    }
    
    # Feature selection
    N_FEATURES = 8  # Number of features for quantum encoding (4-10 recommended)
    
    # Dataset sizes
    SAMPLE_SIZE = 2500  # Max samples per class (for balanced dataset)
    TEST_SIZE = 0.3
    
    # Create directories
    for dir_path in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

config = Config()

print("="*80)
print("QUANTUM IDS PROJECT - ENHANCED DATA PREPROCESSING")
print("="*80)
print(f"Classification Mode: {config.MODE.upper()}")

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
print(f"\nAll attack types found:")
print(df['label'].value_counts())

# ========================================
# STEP 2: Classification Setup (Mode-dependent)
# ========================================
print(f"\n[STEP 2] Setting up {config.MODE.upper()} classification...")

if config.MODE == 'binary':
    # Binary: Normal (0) vs. All Attacks (1)
    df_processed = df.copy()
    df_processed['label'] = df_processed['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    class_names = ['Normal', 'Attack']
    
    print(f"✓ Binary classification created")
    print(f"  Normal (0):  {sum(df_processed['label'] == 0):,}")
    print(f"  Attack (1):  {sum(df_processed['label'] == 1):,}")

elif config.MODE == 'multiclass':
    # Multi-class: Top N most common attacks + Normal
    top_labels = df['label'].value_counts().head(config.TOP_N_ATTACKS + 1).index.tolist()
    
    # Ensure 'normal.' is included
    if 'normal.' not in top_labels:
        top_labels = ['normal.'] + top_labels[:config.TOP_N_ATTACKS]
    
    df_processed = df[df['label'].isin(top_labels)].copy()
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_processed['label'] = label_encoder.fit_transform(df_processed['label'])
    class_names = label_encoder.classes_.tolist()
    
    print(f"✓ Multi-class classification created with {len(class_names)} classes:")
    for i, class_name in enumerate(class_names):
        count = sum(df_processed['label'] == i)
        print(f"  {i}. {class_name:20s} : {count:,}")

elif config.MODE == 'category':
    # Category: DoS, Probe, R2L, U2R, Normal
    df_processed = df.copy()
    df_processed['category'] = df_processed['label'].map(config.ATTACK_CATEGORIES)
    
    # Encode categories
    label_encoder = LabelEncoder()
    df_processed['label'] = label_encoder.fit_transform(df_processed['category'])
    class_names = label_encoder.classes_.tolist()
    
    print(f"✓ Category classification created with {len(class_names)} categories:")
    for i, class_name in enumerate(class_names):
        count = sum(df_processed['label'] == i)
        print(f"  {i}. {class_name:20s} : {count:,}")

else:
    raise ValueError(f"Invalid MODE: {config.MODE}. Choose 'binary', 'multiclass', or 'category'")

# ========================================
# STEP 3: Feature Encoding
# ========================================
print("\n[STEP 3] Encoding categorical features...")

# Separate features and labels
X = df_processed.drop(['label'] + (['category'] if 'category' in df_processed.columns else []), axis=1).copy()
y = df_processed['label'].copy()

# Encode categorical features
categorical_columns = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✓ Features encoded: {X.shape}")

# ========================================
# STEP 4: Feature Selection
# ========================================
print("\n[STEP 4] Performing feature selection...")

# Sample data for faster feature selection
sample_size_fs = min(10000, len(X))
if len(X) > sample_size_fs:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size_fs, stratify=y, random_state=RANDOM_SEED)
else:
    X_sample, y_sample = X, y

print(f"  Training Random Forest on {len(X_sample):,} samples...")

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
rf.fit(X_sample, y_sample)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
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

# Get unique classes
unique_classes = sorted(y.unique())
n_classes = len(unique_classes)

print(f"  Number of classes: {n_classes}")
print(f"  Before balancing:")
for cls in unique_classes:
    print(f"    Class {cls} ({class_names[cls]}): {sum(y == cls):,}")

# Find minimum class size
min_class_size = min([sum(y == cls) for cls in unique_classes])
sample_size = min(min_class_size, config.SAMPLE_SIZE)

print(f"\n  Using {sample_size:,} samples per class")

# Balance by downsampling each class
X_balanced_list = []
y_balanced_list = []

for cls in unique_classes:
    X_cls = X_selected[y == cls]
    y_cls = y[y == cls]
    
    if len(X_cls) > sample_size:
        X_cls_sampled = resample(X_cls, n_samples=sample_size, random_state=RANDOM_SEED)
        y_cls_sampled = resample(y_cls, n_samples=sample_size, random_state=RANDOM_SEED)
    else:
        X_cls_sampled = X_cls
        y_cls_sampled = y_cls
    
    X_balanced_list.append(X_cls_sampled)
    y_balanced_list.append(y_cls_sampled)

X_balanced = pd.concat(X_balanced_list)
y_balanced = pd.concat(y_balanced_list)

print(f"\n✓ Balanced dataset: {X_balanced.shape}")
print(f"  After balancing:")
for cls in unique_classes:
    print(f"    Class {cls} ({class_names[cls]}): {sum(y_balanced == cls):,}")

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

print(f"✓ Train set: {X_train.shape}")
print(f"✓ Test set:  {X_test.shape}")
print(f"\n  Train distribution:")
for cls in unique_classes:
    print(f"    Class {cls} ({class_names[cls]}): {sum(y_train == cls):,}")
print(f"\n  Test distribution:")
for cls in unique_classes:
    print(f"    Class {cls} ({class_names[cls]}): {sum(y_test == cls):,}")

# ========================================
# STEP 7: Normalization for Classical Models
# ========================================
print("\n[STEP 7] Normalizing features for classical models...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")

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
print(f"  Range: [{X_train_quantum.min():.3f}, {X_test_quantum.max():.3f}]")

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

# Save class names
with open(os.path.join(config.PROCESSED_DIR, 'class_names.txt'), 'w') as f:
    f.write('\n'.join(class_names))

# Save configuration
config_dict = {
    'mode': config.MODE,
    'n_classes': n_classes,
    'class_names': class_names,
    'n_features': config.N_FEATURES,
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

# 1. Feature Distributions
fig = plt.figure(figsize=(18, 12))
for idx, feature in enumerate(top_features):
    ax = plt.subplot(3, 3, idx+1)
    for cls in unique_classes:
        ax.hist(X_balanced[feature][y_balanced == cls], alpha=0.5, 
                label=class_names[cls], bins=30)
    ax.set_title(f'{feature}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    if idx == 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Feature distributions saved")
plt.close()

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

for i, feature in enumerate(top_15['feature']):
    if feature in top_features:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"✓ Feature importance plot saved")
plt.close()

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation = X_balanced[top_features].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✓ Correlation matrix saved")
plt.close()

# 4. Class Distribution Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
class_counts = pd.Series([sum(y_balanced == cls) for cls in unique_classes], 
                         index=[class_names[cls] for cls in unique_classes])
class_counts.plot(kind='bar', ax=axes[0], color=plt.cm.Set3(range(n_classes)))
axes[0].set_title('Class Distribution (Balanced)')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Class')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Train-Test split
split_data = pd.DataFrame({
    class_names[cls]: [sum(y_train == cls), sum(y_test == cls)]
    for cls in unique_classes
}, index=['Train', 'Test'])
split_data.plot(kind='bar', ax=axes[1], color=plt.cm.Set3(range(n_classes)))
axes[1].set_title('Train-Test Split Distribution')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3, axis='y')

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
print(f"\nConfiguration:")
print(f"  Classification mode: {config.MODE}")
print(f"  Number of classes:   {n_classes}")
print(f"  Classes:             {', '.join(class_names)}")
print(f"\nDataset Summary:")
print(f"  Total samples:       {len(X_balanced):,}")
print(f"  Training samples:    {len(X_train):,}")
print(f"  Testing samples:     {len(X_test):,}")
print(f"  Number of features:  {config.N_FEATURES}")
print(f"  Feature names:       {', '.join(top_features)}")
print(f"\nFiles saved in:")
print(f"  Data:   {config.PROCESSED_DIR}/")
print(f"  Plots:  {config.FIGURES_DIR}/")
print(f"\nNext step: Run 02_classical_baseline.py")
print("="*80)