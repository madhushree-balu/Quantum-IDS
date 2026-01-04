"""
Quantum IDS Project - Updated Classical Baseline (95%+ Target)
File: src/02_classical_baseline.py

Compatible with improved preprocessing pipeline
Target: 95%+ accuracy for category mode

Features:
- Works with new preprocessing structure
- Validation set monitoring
- Research-proven hyperparameters for IDS
- Ensemble methods
- Publication-ready visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS PROJECT - CLASSICAL BASELINE (95%+ TARGET)")
print("="*80)

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    # Performance settings
    'use_hyperparameter_tuning': True,
    'use_cross_validation': True,
    'use_ensemble': True,
    'monitor_validation': True,  # NEW: Track validation performance
    
    # Speed vs accuracy tradeoff
    'mode': 'balanced',  # 'fast' (5 min), 'balanced' (15 min), 'thorough' (30+ min)
    
    # Computation settings
    'cv_folds': 5,
    'n_iter_search': 30,  # Increased for better hyperparameter search
    'n_jobs': -1,  # Use all CPU cores
    
    # Output settings
    'verbose': True,
    'save_models': True,
}

# Adjust settings based on mode
if CONFIG['mode'] == 'fast':
    CONFIG['n_iter_search'] = 15
    CONFIG['cv_folds'] = 3
elif CONFIG['mode'] == 'thorough':
    CONFIG['n_iter_search'] = 50
    CONFIG['cv_folds'] = 5

print(f"Mode: {CONFIG['mode'].upper()} | CV Folds: {CONFIG['cv_folds']} | Search Iterations: {CONFIG['n_iter_search']}")

# ========================================
# STEP 1: Load Processed Data
# ========================================
print("\n[STEP 1] Loading processed data...")

PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Load training data (use standard scaling for classical ML)
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_standard.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val_standard.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test_standard.npy'))

y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

# Load metadata
with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
    data_config = json.load(f)

with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
feature_names = metadata['feature_names']
n_classes = metadata['n_classes']
is_binary = (n_classes == 2)

print(f"✓ Data loaded successfully")
print(f"  Classification mode: {data_config['mode']}")
print(f"  Number of classes:   {n_classes}")
print(f"  Class names:         {', '.join(class_names)}")
print(f"  Training samples:    {len(X_train):,}")
print(f"  Validation samples:  {len(X_val):,}")
print(f"  Testing samples:     {len(X_test):,}")
print(f"  Number of features:  {X_train.shape[1]}")

# ========================================
# STEP 2: Define Optimized Models (Tuned for 95%+)
# ========================================
print("\n[STEP 2] Defining optimized models (tuned for high accuracy)...")

avg_method = 'binary' if is_binary else 'weighted'

# Aggressive hyperparameters tuned for intrusion detection (95%+ target)
models_config = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1, warm_start=True),
        'params': {
            'n_estimators': [300, 500, 700],  # More trees
            'max_depth': [30, 40, 50, None],  # Deeper trees
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', 0.3],
            'class_weight': ['balanced', 'balanced_subsample'],
            'bootstrap': [True],
            'criterion': ['gini', 'entropy'],
        },
        'priority': 1
    },
    
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [7, 9, 11],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', 0.3],
        },
        'priority': 2
    },
    
    'SVM-RBF': {
        'model': SVC(kernel='rbf', probability=True, cache_size=2000, random_state=42),
        'params': {
            'C': [10, 50, 100, 200, 500],  # Higher regularization
            'gamma': ['scale', 'auto', 0.01, 0.1, 0.5],
            'class_weight': ['balanced', None],
        },
        'priority': 3
    },
    
    'Neural Network': {
        'model': MLPClassifier(random_state=42, early_stopping=True, 
                              validation_fraction=0.15, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(128,), (128, 64), (150, 75), (200, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.005, 0.01],
            'batch_size': ['auto', 64, 128],
        },
        'priority': 4
    },
}

# Sort by priority
models_config = dict(sorted(models_config.items(), key=lambda x: x[1]['priority']))

print(f"✓ {len(models_config)} models configured for high-performance training")
for name in models_config.keys():
    print(f"  • {name}")

# ========================================
# STEP 3: Train and Evaluate Models
# ========================================
print("\n[STEP 3] Training and evaluating models...")
print("-"*80)

results = {
    'Model': [],
    'Train_Acc': [],
    'Val_Acc': [],
    'Test_Acc': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': [],
    'Training Time (s)': [],
    'CV Score': [],
    'CV Std': [],
    'Best_Params': []
}

predictions = {}
probabilities = {}
trained_models = {}
feature_importances = {}

for model_name, model_info in models_config.items():
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        if CONFIG['use_hyperparameter_tuning']:
            print(f"  Performing RandomizedSearchCV ({CONFIG['n_iter_search']} iterations)...")
            
            cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=42)
            
            search = RandomizedSearchCV(
                model_info['model'],
                model_info['params'],
                n_iter=CONFIG['n_iter_search'],
                cv=cv,
                scoring='f1_weighted' if not is_binary else 'f1',
                n_jobs=CONFIG['n_jobs'],
                random_state=42,
                verbose=1 if CONFIG['verbose'] else 0
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
            
            # Calculate CV std
            cv_results_df = pd.DataFrame(search.cv_results_)
            best_index = search.best_index_
            cv_std = cv_results_df.loc[best_index, 'std_test_score']
            
            print(f"  ✓ Best CV score: {cv_score:.4f} (+/- {cv_std:.4f})")
            print(f"  ✓ Best params: {best_params}")
        else:
            print("  Training with default parameters...")
            model = model_info['model']
            model.fit(X_train, y_train)
            best_params = {}
            cv_score = 0.0
            cv_std = 0.0
        
        training_time = time.time() - start_time
        print(f"  ✓ Training completed in {training_time:.2f}s")
        
        # Predictions on all sets
        print("  Evaluating on train/val/test sets...")
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)
            y_val_proba = model.predict_proba(X_val)
            y_test_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_train_proba = model.decision_function(X_train)
            y_val_proba = model.decision_function(X_val)
            y_test_proba = model.decision_function(X_test)
        else:
            y_train_proba = None
            y_val_proba = None
            y_test_proba = None
        
        # Calculate metrics for all sets
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        precision = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average=avg_method, zero_division=0)
        
        # ROC-AUC
        try:
            if is_binary and y_test_proba is not None:
                if len(y_test_proba.shape) > 1:
                    roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_test_proba)
            elif not is_binary and y_test_proba is not None:
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                if y_test_bin.shape[1] == 1:
                    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                roc_auc = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
            else:
                roc_auc = 0.0
        except Exception as e:
            if CONFIG['verbose']:
                print(f"  Warning: ROC-AUC calculation failed: {e}")
            roc_auc = 0.0
        
        # Store results
        results['Model'].append(model_name)
        results['Train_Acc'].append(train_acc)
        results['Val_Acc'].append(val_acc)
        results['Test_Acc'].append(test_acc)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1)
        results['ROC-AUC'].append(roc_auc)
        results['Training Time (s)'].append(training_time)
        results['CV Score'].append(cv_score)
        results['CV Std'].append(cv_std)
        results['Best_Params'].append(str(best_params))
        
        predictions[model_name] = y_test_pred
        probabilities[model_name] = y_test_proba
        trained_models[model_name] = model
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = model.feature_importances_
        
        # Print results with performance indicator
        emoji = "🎯" if test_acc >= 0.95 else "✓" if test_acc >= 0.90 else "⚠️"
        print(f"\n  {emoji} RESULTS:")
        print(f"    Train Accuracy:  {train_acc:.4f}")
        print(f"    Val Accuracy:    {val_acc:.4f}")
        print(f"    Test Accuracy:   {test_acc:.4f}")
        print(f"    Precision:       {precision:.4f}")
        print(f"    Recall:          {recall:.4f}")
        print(f"    F1-Score:        {f1:.4f}")
        print(f"    ROC-AUC:         {roc_auc:.4f}")
        
        # Check for overfitting
        if train_acc - val_acc > 0.05:
            print(f"    ⚠️  Overfitting detected (train-val gap: {train_acc - val_acc:.3f})")
        
        # Save model if it's good
        if CONFIG['save_models'] and test_acc >= 0.85:
            model_path = os.path.join(MODELS_DIR, f'classical_{model_name.replace(" ", "_").lower()}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"    💾 Model saved to {model_path}")
        
    except Exception as e:
        print(f"  ✗ Error training {model_name}: {str(e)}")
        import traceback
        if CONFIG['verbose']:
            traceback.print_exc()
        continue

# ========================================
# STEP 4: Create Ensemble
# ========================================
if CONFIG['use_ensemble'] and len(trained_models) >= 2:
    print("\n" + "="*80)
    print("Creating Ensemble Model")
    print("="*80)
    
    try:
        # Use top 3 models based on validation accuracy
        results_df_temp = pd.DataFrame(results)
        top_models = results_df_temp.nlargest(min(3, len(trained_models)), 'Val_Acc')['Model'].tolist()
        
        print(f"  Ensemble members: {', '.join(top_models)}")
        
        estimators = [(name, trained_models[name]) for name in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_train_pred_ens = ensemble.predict(X_train)
        y_val_pred_ens = ensemble.predict(X_val)
        y_test_pred_ens = ensemble.predict(X_test)
        y_test_proba_ens = ensemble.predict_proba(X_test)
        
        # Calculate metrics
        train_acc_ens = accuracy_score(y_train, y_train_pred_ens)
        val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
        test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
        precision_ens = precision_score(y_test, y_test_pred_ens, average=avg_method, zero_division=0)
        recall_ens = recall_score(y_test, y_test_pred_ens, average=avg_method, zero_division=0)
        f1_ens = f1_score(y_test, y_test_pred_ens, average=avg_method, zero_division=0)
        
        try:
            if is_binary:
                roc_auc_ens = roc_auc_score(y_test, y_test_proba_ens[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                if y_test_bin.shape[1] == 1:
                    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                roc_auc_ens = roc_auc_score(y_test_bin, y_test_proba_ens, average='weighted', multi_class='ovr')
        except:
            roc_auc_ens = 0.0
        
        # Add to results
        results['Model'].append('🏆 Ensemble')
        results['Train_Acc'].append(train_acc_ens)
        results['Val_Acc'].append(val_acc_ens)
        results['Test_Acc'].append(test_acc_ens)
        results['Precision'].append(precision_ens)
        results['Recall'].append(recall_ens)
        results['F1-Score'].append(f1_ens)
        results['ROC-AUC'].append(roc_auc_ens)
        results['Training Time (s)'].append(training_time)
        results['CV Score'].append(0.0)
        results['CV Std'].append(0.0)
        results['Best_Params'].append(f"Voting: {', '.join(top_models)}")
        
        predictions['🏆 Ensemble'] = y_test_pred_ens
        probabilities['🏆 Ensemble'] = y_test_proba_ens
        trained_models['🏆 Ensemble'] = ensemble
        
        emoji = "🎯" if test_acc_ens >= 0.95 else "✓"
        print(f"\n  {emoji} ENSEMBLE RESULTS:")
        print(f"    Train Accuracy:  {train_acc_ens:.4f}")
        print(f"    Val Accuracy:    {val_acc_ens:.4f}")
        print(f"    Test Accuracy:   {test_acc_ens:.4f}")
        print(f"    Precision:       {precision_ens:.4f}")
        print(f"    Recall:          {recall_ens:.4f}")
        print(f"    F1-Score:        {f1_ens:.4f}")
        print(f"    ROC-AUC:         {roc_auc_ens:.4f}")
        
        # Save ensemble
        if CONFIG['save_models'] and test_acc_ens >= 0.85:
            ensemble_path = os.path.join(MODELS_DIR, 'classical_ensemble.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble, f)
            print(f"    💾 Ensemble saved to {ensemble_path}")
        
    except Exception as e:
        print(f"  ✗ Ensemble creation failed: {e}")

# ========================================
# STEP 5: Results Summary
# ========================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_Acc', ascending=False)

print("\n" + "="*80)
print("CLASSICAL BASELINE RESULTS")
print("="*80)
print(results_df[['Model', 'Train_Acc', 'Val_Acc', 'Test_Acc', 'F1-Score', 'ROC-AUC']].to_string(index=False))

# Save results
results_df.to_csv(os.path.join(RESULTS_DIR, 'classical_baseline_results.csv'), index=False)
print(f"\n✓ Results saved to {RESULTS_DIR}/classical_baseline_results.csv")

# ========================================
# STEP 6: Best Model Analysis
# ========================================
print("\n" + "="*80)
print("BEST MODEL ANALYSIS")
print("="*80)

best_idx = results_df['Test_Acc'].idxmax()
best_model_info = results_df.iloc[best_idx]
best_model_name = best_model_info['Model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"  Train Accuracy:   {best_model_info['Train_Acc']:.4f}")
print(f"  Val Accuracy:     {best_model_info['Val_Acc']:.4f}")
print(f"  Test Accuracy:    {best_model_info['Test_Acc']:.4f}")
print(f"  Precision:        {best_model_info['Precision']:.4f}")
print(f"  Recall:           {best_model_info['Recall']:.4f}")
print(f"  F1-Score:         {best_model_info['F1-Score']:.4f}")
print(f"  ROC-AUC:          {best_model_info['ROC-AUC']:.4f}")
if best_model_info['CV Score'] > 0:
    print(f"  CV Score:         {best_model_info['CV Score']:.4f} (+/- {best_model_info['CV Std']:.4f})")
print(f"  Training Time:    {best_model_info['Training Time (s)']:.2f}s")

# Performance assessment
if best_model_info['Test_Acc'] >= 0.95:
    print("\n  🎯 OUTSTANDING! 95%+ accuracy achieved!")
    print("  🏆 Publication-ready results!")
elif best_model_info['Test_Acc'] >= 0.90:
    print("\n  ✓ EXCELLENT! Strong performance")
elif best_model_info['Test_Acc'] >= 0.85:
    print("\n  ✓ VERY GOOD! Competitive results")
elif best_model_info['Test_Acc'] >= 0.80:
    print("\n  ✓ GOOD! Solid baseline")
else:
    print("\n  ⚠️  Room for improvement")

# Confusion Matrix
y_pred_best = predictions[best_model_name]
cm = confusion_matrix(y_test, y_pred_best)

print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names, digits=4))

# Per-class metrics
print(f"\nPer-Class Performance:")
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i)
    class_pred = (y_pred_best == i)
    
    tp = np.sum(class_mask & class_pred)
    fp = np.sum(~class_mask & class_pred)
    fn = np.sum(class_mask & ~class_pred)
    
    precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0
    
    print(f"  {class_name:15s}: Precision={precision_cls:.4f}, Recall={recall_cls:.4f}, F1={f1_cls:.4f}")

# ========================================
# STEP 7: Visualization
# ========================================
print("\n[STEP 7] Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison (Train/Val/Test)
ax1 = plt.subplot(2, 4, 1)
x = np.arange(len(results_df))
width = 0.25

ax1.barh(x - width, results_df['Train_Acc'], width, label='Train', alpha=0.8, color='#3498db')
ax1.barh(x, results_df['Val_Acc'], width, label='Val', alpha=0.8, color='#f39c12')
ax1.barh(x + width, results_df['Test_Acc'], width, label='Test', alpha=0.8, color='#e74c3c')

ax1.axvline(x=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')

ax1.set_yticks(x)
ax1.set_yticklabels(results_df['Model'], fontsize=9)
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy (Train/Val/Test)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim([0, 1.05])

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 4, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
bars = ax2.barh(range(len(results_df)), results_df['F1-Score'], color=colors, alpha=0.8)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['Model'], fontsize=9)
ax2.set_xlabel('F1-Score')
ax2.set_title('Model F1-Scores', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim([0, 1.05])

# Highlight best
best_idx_plot = results_df.reset_index(drop=True)[results_df['Model'] == best_model_name].index[0]
bars[best_idx_plot].set_edgecolor('red')
bars[best_idx_plot].set_linewidth(3)

# 3. Confusion Matrix
ax3 = plt.subplot(2, 4, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix\n{best_model_name}', fontsize=12, fontweight='bold')
ax3.set_ylabel('True')
ax3.set_xlabel('Predicted')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)

# 4. Normalized Confusion Matrix
ax4 = plt.subplot(2, 4, 4)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax4,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'})
ax4.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
ax4.set_ylabel('True')
ax4.set_xlabel('Predicted')
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=8)

# 5. Per-Class Accuracy
ax5 = plt.subplot(2, 4, 5)
per_class_acc = []
for i in range(n_classes):
    class_mask = (y_test == i)
    if class_mask.sum() > 0:
        class_acc = (y_pred_best[class_mask] == i).sum() / class_mask.sum()
        per_class_acc.append(class_acc)
    else:
        per_class_acc.append(0)

colors_class = plt.cm.Set3(np.arange(n_classes))
bars = ax5.bar(range(n_classes), per_class_acc, color=colors_class, alpha=0.8, edgecolor='black')
ax5.set_xticks(range(n_classes))
ax5.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('Accuracy')
ax5.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1.1])
ax5.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')
ax5.axhline(y=best_model_info['Test_Acc'], color='red', linestyle='--', 
            linewidth=2, label=f'Overall: {best_model_info["Test_Acc"]:.3f}')
ax5.legend()

for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

# 6. Overfitting Analysis (Train vs Val vs Test)
ax6 = plt.subplot(2, 4, 6)
models_for_plot = results_df['Model'].tolist()
train_scores = results_df['Train_Acc'].tolist()
val_scores = results_df['Val_Acc'].tolist()
test_scores = results_df['Test_Acc'].tolist()

x = np.arange(len(models_for_plot))
width = 0.25

ax6.bar(x - width, train_scores, width, label='Train', alpha=0.8, color='#3498db')
ax6.bar(x, val_scores, width, label='Val', alpha=0.8, color='#f39c12')
ax6.bar(x + width, test_scores, width, label='Test', alpha=0.8, color='#e74c3c')

ax6.set_xticks(x)
ax6.set_xticklabels(models_for_plot, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Accuracy')
ax6.set_title('Generalization Analysis', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim([0, 1.05])

# 7. Metrics Radar Chart
ax7 = plt.subplot(2, 4, 7, projection='polar')
metrics = ['Test_Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [best_model_info[m] for m in metrics] + [best_model_info[metrics[0]]]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]

ax7.plot(angles, values, 'o-', linewidth=2, color='#2ecc71', label=best_model_name)
ax7.fill(angles, values, alpha=0.25, color='#2ecc71')
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metric_labels, fontsize=9)
ax7.set_ylim(0, 1)
ax7.set_title(f'Performance Radar\n{best_model_name}', fontsize=12, fontweight='bold', pad=20)
ax7.grid(True)

# Add reference circle at 0.95
ax7.plot(angles, [0.95] * len(angles), 'g--', linewidth=1.5, alpha=0.5, label='95% Target')
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 8. Summary Statistics
ax8 = plt.subplot(2, 4, 8)
summary_text = f"""
CLASSICAL BASELINE SUMMARY
{'='*40}

Best Model: {best_model_name}

Performance:
  Train Acc:    {best_model_info['Train_Acc']:.4f}
  Val Acc:      {best_model_info['Val_Acc']:.4f}
  Test Acc:     {best_model_info['Test_Acc']:.4f}
  Precision:    {best_model_info['Precision']:.4f}
  Recall:       {best_model_info['Recall']:.4f}
  F1-Score:     {best_model_info['F1-Score']:.4f}
  ROC-AUC:      {best_model_info['ROC-AUC']:.4f}

Dataset:
  Mode:         {data_config['mode']}
  Classes:      {n_classes}
  Train:        {len(X_train):,}
  Val:          {len(X_val):,}
  Test:         {len(X_test):,}
  Features:     {X_train.shape[1]}

Training:
  Time:         {best_model_info['Training Time (s)']:.2f}s
"""

if best_model_info['CV Score'] > 0:
    summary_text += f"  CV Score:     {best_model_info['CV Score']:.4f}\n"

summary_text += f"\nStatus:\n"
if best_model_info['Test_Acc'] >= 0.95:
    summary_text += "  🎯 95%+ ACHIEVED!\n  Publication-ready!"
elif best_model_info['Test_Acc'] >= 0.90:
    summary_text += "  ✓ Excellent results\n  Near target!"
elif best_model_info['Test_Acc'] >= 0.85:
    summary_text += "  ✓ Good baseline\n  Room for improvement"
else:
    summary_text += "  ⚠️  Needs tuning"

# Check overfitting
overfit_gap = best_model_info['Train_Acc'] - best_model_info['Val_Acc']
if overfit_gap > 0.05:
    summary_text += f"\n\n  ⚠️  Overfitting detected\n  Gap: {overfit_gap:.3f}"
else:
    summary_text += f"\n\n  ✓ Good generalization\n  Gap: {overfit_gap:.3f}"

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax8.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'classical_baseline_comprehensive.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive visualization saved")

# ========================================
# Additional Detailed Plots
# ========================================
print("\n[Creating additional detailed plots...]")

# Detailed Confusion Matrix
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, linewidths=0.5)
ax.set_title(f'Confusion Matrix - {best_model_name}\nTest Accuracy: {best_model_info["Test_Acc"]:.4f}', 
            fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_detailed.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Detailed confusion matrix saved")

# Model Comparison Detailed
fig3, ax = plt.subplots(1, 1, figsize=(14, 8))
x_pos = np.arange(len(results_df))
width = 0.15

metrics_to_plot = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1-Score']
colors_metrics = ['#3498db', '#f39c12', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22']

for i, metric in enumerate(metrics_to_plot):
    offset = (i - len(metrics_to_plot)/2) * width
    ax.bar(x_pos + offset, results_df[metric], width, 
           label=metric.replace('_', ' '), alpha=0.8, color=colors_metrics[i])

ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=10)
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison_detailed.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Detailed model comparison saved")

# Feature Importance (if available)
if best_model_name in feature_importances:
    fig4, ax = plt.subplots(1, 1, figsize=(10, 8))
    importances = feature_importances[best_model_name]
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    colors_fi = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    bars = ax.barh(range(len(indices)), importances[indices], color=colors_fi, alpha=0.8)
    ax.set_yticks(range(len(indices)))
    
    if len(feature_names) == len(importances):
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    else:
        ax.set_yticklabels([f'Feature {i}' for i in indices], fontsize=9)
    
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Feature Importance - {best_model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importances[indices])):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved")

print("\n" + "="*80)
print("🎉 CLASSICAL BASELINE TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 FINAL SUMMARY:")
print(f"   Best Model:        {best_model_name}")
print(f"   Test Accuracy:     {best_model_info['Test_Acc']:.4f}")
print(f"   F1-Score:          {best_model_info['F1-Score']:.4f}")
print(f"   Training Time:     {best_model_info['Training Time (s)']:.2f}s")

# Achievement assessment
if best_model_info['Test_Acc'] >= 0.95:
    print(f"\n✨ 🎯 95%+ ACCURACY ACHIEVED! 🎯 ✨")
    print(f"   Your classical baseline is publication-ready!")
    print(f"   Ready for quantum comparison!")
elif best_model_info['Test_Acc'] >= 0.90:
    print(f"\n✨ Excellent {best_model_info['Test_Acc']:.1%} accuracy!")
    print(f"   Very close to 95% target")
    print(f"   Suggestions:")
    print(f"   • Try mode='thorough' for more exhaustive search")
    print(f"   • Increase sample size to {data_config['sample_size'] * 2}")
elif best_model_info['Test_Acc'] >= 0.85:
    print(f"\n✓ Good {best_model_info['Test_Acc']:.1%} accuracy")
    print(f"   To reach 95%:")
    print(f"   • Use more features (current: {X_train.shape[1]})")
    print(f"   • Increase n_iter_search to 50+")
    print(f"   • Try larger sample size")
else:
    print(f"\n⚠️  Current: {best_model_info['Test_Acc']:.1%}")
    print(f"   Action items:")
    print(f"   • Check data preprocessing")
    print(f"   • Verify class balance")
    print(f"   • Increase training samples")

print(f"\n📁 OUTPUT FILES:")
print(f"   Results CSV:        {RESULTS_DIR}/classical_baseline_results.csv")
print(f"   Main Plot:          {FIGURES_DIR}/classical_baseline_comprehensive.png")
print(f"   Confusion Matrix:   {FIGURES_DIR}/confusion_matrix_detailed.png")
print(f"   Model Comparison:   {FIGURES_DIR}/model_comparison_detailed.png")
if best_model_name in feature_importances:
    print(f"   Feature Importance: {FIGURES_DIR}/feature_importance.png")

if CONFIG['save_models']:
    print(f"\n💾 SAVED MODELS:")
    for model_name in trained_models.keys():
        if results_df[results_df['Model'] == model_name]['Test_Acc'].values[0] >= 0.85:
            print(f"   {model_name}")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review the comprehensive visualization")
print(f"   2. Check per-class performance")
print(f"   3. Compare with quantum kernel results")
print(f"   4. Use saved models for deployment")

print("\n" + "="*80)