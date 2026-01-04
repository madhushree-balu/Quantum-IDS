"""
Quantum IDS Project - Enhanced Data Preprocessing
File: src/01_preprocess_data.py

Optimized for Classical vs Quantum Hybrid Kernel Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS PROJECT - ENHANCED DATA PREPROCESSING")
print("="*80)

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    'mode': 'category',  # 'binary', 'category', or 'multiclass'
    'sample_size': 5000,
    'n_features': 8,  # Reduced for quantum kernel efficiency (4, 8, or 15)
    'test_size': 0.2,
    'val_size': 0.1,  # Validation set from training data
    'random_state': 42,
    'balance_classes': True,
    
    # Scaling options
    'scaling_method': 'both',  # 'standard', 'minmax', or 'both'
    'minmax_range': (0, 1),  # For quantum circuits: (0, 1) or (-1, 1)
    
    # Feature selection
    'feature_selection_method': 'random_forest',  # 'random_forest', 'mutual_info', 'variance'
    'apply_pca': False,  # Optional dimensionality reduction
    'pca_components': 0.95,  # Explained variance or n_components
    
    # Cross-validation
    'n_cv_folds': 5,  # For consistent CV across classical and quantum
}

print(f"\nConfiguration:")
print(f"  Mode:              {CONFIG['mode']}")
print(f"  Sample size:       {CONFIG['sample_size']}")
print(f"  Features:          {CONFIG['n_features']}")
print(f"  Test split:        {CONFIG['test_size']}")
print(f"  Validation split:  {CONFIG['val_size']}")
print(f"  Scaling:           {CONFIG['scaling_method']}")
print(f"  PCA:               {CONFIG['apply_pca']}")

# ========================================
# STEP 1: Create/Check Directories
# ========================================
print("\n[STEP 1] Setting up directories...")

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
MODELS_DIR = 'models'

for directory in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"✓ Directories ready")

# ========================================
# STEP 2: Load KDD Cup Dataset
# ========================================
print("\n[STEP 2] Loading KDD Cup 1999 dataset...")

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

# Try to load existing dataset
dataset_path = os.path.join(RAW_DIR, 'kddcup.data_10_percent.gz')

if os.path.exists(dataset_path):
    print(f"  Loading from {dataset_path}...")
    df = pd.read_csv(dataset_path, names=column_names, compression='gzip')
    print(f"✓ Dataset loaded: {len(df):,} records")
else:
    print(f"  Dataset not found at {dataset_path}")
    print(f"  Please download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html")
    print(f"  Place 'kddcup.data_10_percent.gz' in the {RAW_DIR} directory")
    raise FileNotFoundError("KDD Cup dataset not found")

# ========================================
# STEP 3: Attack Category Mapping
# ========================================
print("\n[STEP 3] Mapping attack types...")

# Attack type categories
attack_mapping = {
    'normal': 'normal',
    # DoS attacks
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
    'processtable': 'dos', 'mailbomb': 'dos',
    # Probe attacks
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    # R2L attacks
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    # U2R attacks
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}

# Clean labels (remove trailing dots)
df['label'] = df['label'].str.rstrip('.')

# Apply mapping
df['category'] = df['label'].map(attack_mapping)

# Handle unmapped labels
unmapped = df['category'].isna().sum()
if unmapped > 0:
    print(f"  Warning: {unmapped} unmapped labels, treating as 'unknown'")
    df['category'] = df['category'].fillna('unknown')

print(f"✓ Attack types mapped")

# ========================================
# STEP 4: Label Encoding Based on Mode
# ========================================
print(f"\n[STEP 4] Creating labels for '{CONFIG['mode']}' mode...")

if CONFIG['mode'] == 'binary':
    # Binary: normal vs attack
    df['target'] = (df['category'] != 'normal').astype(int)
    class_names = ['Normal', 'Attack']
    
elif CONFIG['mode'] == 'category':
    # 5-class: normal, dos, probe, r2l, u2r
    label_encoder = LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['category'])
    class_names = list(label_encoder.classes_)
    class_names = [name.upper() if name != 'normal' else 'Normal' for name in class_names]
    
    # Save label encoder for later use
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
elif CONFIG['mode'] == 'multiclass':
    # Multi-class: individual attack types
    label_encoder = LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['label'])
    class_names = list(label_encoder.classes_)
    
    # Save label encoder
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
else:
    raise ValueError(f"Unknown mode: {CONFIG['mode']}")

print(f"✓ Labels created: {len(class_names)} classes")
print(f"  Classes: {', '.join(class_names)}")

# Show distribution
print(f"\nClass distribution (full dataset):")
for i, name in enumerate(class_names):
    count = (df['target'] == i).sum()
    pct = count / len(df) * 100
    print(f"  {name:15s}: {count:7d} ({pct:5.1f}%)")

# ========================================
# STEP 5: Feature Engineering
# ========================================
print("\n[STEP 5] Encoding categorical features...")

# Separate features and target
X = df.drop(['label', 'category', 'target'], axis=1).copy()
y = df['target'].copy()

# Encode categorical features
categorical_features = ['protocol_type', 'service', 'flag']
label_encoders = {}

for feature in categorical_features:
    if feature in X.columns:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le

# Save categorical encoders
with open(os.path.join(MODELS_DIR, 'categorical_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"✓ Categorical features encoded")
print(f"  Total features: {X.shape[1]}")

# ========================================
# STEP 6: Balanced Sampling (Before Split)
# ========================================
print(f"\n[STEP 6] Creating balanced sample of {CONFIG['sample_size']} records...")

if CONFIG['balance_classes']:
    samples_per_class = CONFIG['sample_size'] // len(class_names)
    
    sampled_indices = []
    for i in range(len(class_names)):
        class_indices = y[y == i].index
        
        if len(class_indices) >= samples_per_class:
            # Sample from majority classes
            sampled = np.random.RandomState(CONFIG['random_state']).choice(
                class_indices, samples_per_class, replace=False
            )
        else:
            # Use all samples if class is small
            sampled = class_indices
            print(f"  Warning: Class '{class_names[i]}' has only {len(class_indices)} samples")
        
        sampled_indices.extend(sampled)
    
    # Shuffle
    np.random.RandomState(CONFIG['random_state']).shuffle(sampled_indices)
    
    X_sampled = X.loc[sampled_indices].reset_index(drop=True)
    y_sampled = y.loc[sampled_indices].reset_index(drop=True)
    
    print(f"✓ Balanced sampling complete")
    print(f"  Target per class: {samples_per_class}")
    print(f"  Total sampled:    {len(X_sampled)}")
else:
    # Random sampling
    sample_indices = np.random.RandomState(CONFIG['random_state']).choice(
        len(X), CONFIG['sample_size'], replace=False
    )
    X_sampled = X.iloc[sample_indices].reset_index(drop=True)
    y_sampled = y.iloc[sample_indices].reset_index(drop=True)
    print(f"✓ Random sampling complete: {len(X_sampled)} records")

# Show final distribution
print(f"\nSampled class distribution:")
for i, name in enumerate(class_names):
    count = (y_sampled == i).sum()
    pct = count / len(y_sampled) * 100
    print(f"  {name:15s}: {count:5d} ({pct:5.1f}%)")

# ========================================
# STEP 7: Train/Val/Test Split
# ========================================
print(f"\n[STEP 7] Splitting into train/val/test sets...")

# First split: train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_sampled, y_sampled,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y_sampled
)

# Second split: train vs val
val_size_adjusted = CONFIG['val_size'] / (1 - CONFIG['test_size'])
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_size_adjusted,
    random_state=CONFIG['random_state'],
    stratify=y_trainval
)

print(f"✓ Split complete:")
print(f"  Training samples:   {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Testing samples:    {len(X_test):,}")

# Save indices for reproducibility
split_indices = {
    'train': X_train.index.tolist(),
    'val': X_val.index.tolist(),
    'test': X_test.index.tolist()
}
with open(os.path.join(PROCESSED_DIR, 'split_indices.json'), 'w') as f:
    json.dump(split_indices, f, indent=2)

# ========================================
# STEP 8: Feature Selection (On Training Data Only)
# ========================================
print(f"\n[STEP 8] Selecting top {CONFIG['n_features']} features...")

if CONFIG['feature_selection_method'] == 'random_forest':
    from sklearn.ensemble import RandomForestClassifier
    
    # Use subset if training data is large
    n_samples_fs = min(5000, len(X_train))
    X_train_sample = X_train.sample(n=n_samples_fs, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    # Train RF for feature importance
    rf_temp = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=10
    )
    rf_temp.fit(X_train_sample, y_train_sample)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
elif CONFIG['feature_selection_method'] == 'mutual_info':
    from sklearn.feature_selection import mutual_info_classif
    
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
elif CONFIG['feature_selection_method'] == 'variance':
    from sklearn.feature_selection import VarianceThreshold
    
    variances = X_train.var()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': variances
    }).sort_values('importance', ascending=False)
else:
    raise ValueError(f"Unknown feature selection method: {CONFIG['feature_selection_method']}")

# Select top features
top_features = feature_importance.head(CONFIG['n_features'])['feature'].tolist()

print(f"✓ Top {CONFIG['n_features']} features selected:")
for idx, row in feature_importance.head(CONFIG['n_features']).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv(
    os.path.join(PROCESSED_DIR, 'feature_importance.csv'), 
    index=False
)

# Keep only selected features
X_train = X_train[top_features]
X_val = X_val[top_features]
X_test = X_test[top_features]

# ========================================
# STEP 9: Feature Scaling
# ========================================
print("\n[STEP 9] Scaling features...")

scalers = {}

# Standard Scaling (zero mean, unit variance)
if CONFIG['scaling_method'] in ['standard', 'both']:
    print("  Applying StandardScaler...")
    scaler_standard = StandardScaler()
    X_train_standard = scaler_standard.fit_transform(X_train)
    X_val_standard = scaler_standard.transform(X_val)
    X_test_standard = scaler_standard.transform(X_test)
    scalers['standard'] = scaler_standard
    
    print(f"    Train mean: {X_train_standard.mean():.4f}, std: {X_train_standard.std():.4f}")
    print(f"    Val mean:   {X_val_standard.mean():.4f}, std: {X_val_standard.std():.4f}")
    print(f"    Test mean:  {X_test_standard.mean():.4f}, std: {X_test_standard.std():.4f}")

# MinMax Scaling (for quantum circuits)
if CONFIG['scaling_method'] in ['minmax', 'both']:
    print(f"  Applying MinMaxScaler (range: {CONFIG['minmax_range']})...")
    scaler_minmax = MinMaxScaler(feature_range=CONFIG['minmax_range'])
    X_train_minmax = scaler_minmax.fit_transform(X_train)
    X_val_minmax = scaler_minmax.transform(X_val)
    X_test_minmax = scaler_minmax.transform(X_test)
    scalers['minmax'] = scaler_minmax
    
    print(f"    Train range: [{X_train_minmax.min():.4f}, {X_train_minmax.max():.4f}]")
    print(f"    Val range:   [{X_val_minmax.min():.4f}, {X_val_minmax.max():.4f}]")
    print(f"    Test range:  [{X_test_minmax.min():.4f}, {X_test_minmax.max():.4f}]")

# Save scalers
with open(os.path.join(MODELS_DIR, 'scalers.pkl'), 'wb') as f:
    pickle.dump(scalers, f)

print(f"✓ Scalers saved to {MODELS_DIR}/scalers.pkl")

# ========================================
# STEP 10: Optional PCA
# ========================================
if CONFIG['apply_pca']:
    print(f"\n[STEP 10] Applying PCA...")
    
    # Use standard-scaled data for PCA
    pca = PCA(n_components=CONFIG['pca_components'], random_state=CONFIG['random_state'])
    X_train_pca = pca.fit_transform(X_train_standard)
    X_val_pca = pca.transform(X_val_standard)
    X_test_pca = pca.transform(X_test_standard)
    
    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()
    
    print(f"✓ PCA complete:")
    print(f"  Components: {n_components}")
    print(f"  Explained variance: {explained_var:.2%}")
    
    # Save PCA transformer
    with open(os.path.join(MODELS_DIR, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    
    # Save PCA-transformed data
    np.save(os.path.join(PROCESSED_DIR, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(PROCESSED_DIR, 'X_val_pca.npy'), X_val_pca)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_pca.npy'), X_test_pca)
else:
    print(f"\n[STEP 10] Skipping PCA (disabled in config)")

# ========================================
# STEP 11: Cross-Validation Folds
# ========================================
print(f"\n[STEP 11] Creating {CONFIG['n_cv_folds']}-fold CV indices...")

# Create stratified K-fold indices on training data
skf = StratifiedKFold(
    n_splits=CONFIG['n_cv_folds'], 
    shuffle=True, 
    random_state=CONFIG['random_state']
)

cv_folds = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    cv_folds.append({
        'fold': fold_idx + 1,
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist()
    })

# Save CV folds
with open(os.path.join(PROCESSED_DIR, 'cv_folds.json'), 'w') as f:
    json.dump(cv_folds, f, indent=2)

print(f"✓ CV folds saved for reproducible comparison")

# ========================================
# STEP 12: Save Processed Data
# ========================================
print("\n[STEP 12] Saving processed data...")

# Save raw (unscaled) data
np.save(os.path.join(PROCESSED_DIR, 'X_train_raw.npy'), X_train.values)
np.save(os.path.join(PROCESSED_DIR, 'X_val_raw.npy'), X_val.values)
np.save(os.path.join(PROCESSED_DIR, 'X_test_raw.npy'), X_test.values)

# Save standard-scaled data
if 'standard' in scalers:
    np.save(os.path.join(PROCESSED_DIR, 'X_train_standard.npy'), X_train_standard)
    np.save(os.path.join(PROCESSED_DIR, 'X_val_standard.npy'), X_val_standard)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_standard.npy'), X_test_standard)

# Save minmax-scaled data (for quantum)
if 'minmax' in scalers:
    np.save(os.path.join(PROCESSED_DIR, 'X_train_minmax.npy'), X_train_minmax)
    np.save(os.path.join(PROCESSED_DIR, 'X_val_minmax.npy'), X_val_minmax)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_minmax.npy'), X_test_minmax)

# Save labels
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val.values)
np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test.values)

# Save configuration
with open(os.path.join(PROCESSED_DIR, 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=2)

# Save class names
with open(os.path.join(PROCESSED_DIR, 'class_names.txt'), 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# Save feature names
with open(os.path.join(PROCESSED_DIR, 'feature_names.txt'), 'w') as f:
    for name in top_features:
        f.write(f"{name}\n")

# Save metadata
metadata = {
    'n_classes': len(class_names),
    'class_names': class_names,
    'n_features': len(top_features),
    'feature_names': top_features,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'scaling_methods': list(scalers.keys()),
    'pca_applied': CONFIG['apply_pca']
}

with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ All data saved to {PROCESSED_DIR}/")
print(f"✓ Models/scalers saved to {MODELS_DIR}/")

# ========================================
# STEP 13: Visualization
# ========================================
print("\n[STEP 13] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Class distribution across splits
ax = axes[0, 0]
train_counts = [np.sum(y_train == i) for i in range(len(class_names))]
val_counts = [np.sum(y_val == i) for i in range(len(class_names))]
test_counts = [np.sum(y_test == i) for i in range(len(class_names))]

x = np.arange(len(class_names))
width = 0.25

ax.bar(x - width, train_counts, width, label='Train', alpha=0.8, color='#3498db')
ax.bar(x, val_counts, width, label='Val', alpha=0.8, color='#2ecc71')
ax.bar(x + width, test_counts, width, label='Test', alpha=0.8, color='#e74c3c')

ax.set_xlabel('Class', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Class Distribution Across Splits', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Feature importance
ax = axes[0, 1]
top_10 = feature_importance.head(10)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
bars = ax.barh(range(len(top_10)), top_10['importance'], color=colors)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'], fontsize=9)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title(f'Top 10 Features ({CONFIG["feature_selection_method"]})', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, top_10['importance'])):
    ax.text(val, i, f' {val:.3f}', va='center', fontsize=8)

# 3. Feature correlation heatmap
ax = axes[1, 0]
if 'standard' in scalers:
    correlation_matrix = pd.DataFrame(X_train_standard, columns=top_features).corr()
else:
    correlation_matrix = X_train.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, ax=ax, cmap='coolwarm', center=0,
            mask=mask, square=True, linewidths=0.5, 
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

# 4. Summary statistics
ax = axes[1, 1]
stats_text = f"""
PREPROCESSING SUMMARY (Classical vs Quantum)
{'='*50}

Configuration:
  Mode:              {CONFIG['mode']}
  Sample Size:       {CONFIG['sample_size']:,}
  Features Selected: {CONFIG['n_features']} (quantum-optimized)
  Feature Method:    {CONFIG['feature_selection_method']}

Dataset Splits:
  Training:          {len(X_train):,} samples
  Validation:        {len(X_val):,} samples  
  Testing:           {len(X_test):,} samples
  CV Folds:          {CONFIG['n_cv_folds']}

Scaling Methods:
  Standard Scaler:   {'✓' if 'standard' in scalers else '✗'}
  MinMax Scaler:     {'✓' if 'minmax' in scalers else '✗'}
  {'  Range:            ' + str(CONFIG['minmax_range']) if 'minmax' in scalers else ''}
  PCA Applied:       {'✓' if CONFIG['apply_pca'] else '✗'}

Class Distribution (Train):
"""

for i, name in enumerate(class_names):
    train_pct = train_counts[i] / len(y_train) * 100
    stats_text += f"  {name:12s}: {train_pct:5.1f}%\n"

stats_text += f"""
{'='*50}
✓ Ready for Classical & Quantum Training!

Files Saved:
  • Processed data (raw, standard, minmax)
  • Scalers & encoders (reusable)
  • CV folds (reproducible)
  • Feature importance
"""

ax.text(0.05, 0.5, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_summary.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to {FIGURES_DIR}/preprocessing_summary.png")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"\n✓ Processed data saved to:     {PROCESSED_DIR}/")
print(f"✓ Configuration saved to:      {PROCESSED_DIR}/config.json")
print(f"✓ Visualizations saved to:     {FIGURES_DIR}/")
print(f"\n📊 Dataset Summary:")
print(f"   Mode:           {CONFIG['mode']}")
print(f"   Classes:        {len(class_names)}")
print(f"   Training:       {len(X_train):,} samples")
print(f"   Testing:        {len(X_test):,} samples")
print(f"   Features:       {CONFIG['n_features']}")
print(f"\n🚀 Ready to run classical baseline training!")
print("="*80)