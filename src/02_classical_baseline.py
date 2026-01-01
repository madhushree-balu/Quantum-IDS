"""
Quantum IDS Project - Classical Baseline Models
File: src/02_classical_baseline.py
Purpose: Train and evaluate classical ML models as baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS PROJECT - CLASSICAL BASELINE MODELS")
print("="*80)

# ========================================
# STEP 1: Load Processed Data
# ========================================
print("\n[STEP 1] Loading processed data...")

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Load data
X_train_scaled = np.load(os.path.join(PROCESSED_DIR, 'X_train_scaled.npy'))
X_test_scaled = np.load(os.path.join(PROCESSED_DIR, 'X_test_scaled.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

# Load configuration
with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

print(f"✓ Data loaded successfully")
print(f"  Training samples: {len(X_train_scaled)}")
print(f"  Testing samples:  {len(X_test_scaled)}")
print(f"  Number of features: {config['n_features']}")

# ========================================
# STEP 2: Define Models to Test
# ========================================
print("\n[STEP 2] Defining classical models...")

models_to_test = {
    # Support Vector Machines with different kernels
    'SVM-Linear': SVC(kernel='linear', random_state=42, probability=True),
    'SVM-RBF': SVC(kernel='rbf', random_state=42, probability=True, cache_size=2000),
    'SVM-Poly': SVC(kernel='poly', degree=3, random_state=42, probability=True),
    'SVM-Sigmoid': SVC(kernel='sigmoid', random_state=42, probability=True),
    
    # Ensemble Methods
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    
    # Linear Models
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    
    # Instance-based
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

print(f"✓ {len(models_to_test)} models defined")

# ========================================
# STEP 3: Train and Evaluate Models
# ========================================
print("\n[STEP 3] Training and evaluating models...")
print("-"*80)

results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': [],
    'Training Time (s)': [],
    'Prediction Time (s)': [],
    'Total Time (s)': []
}

# Store predictions for later analysis
predictions = {}
probabilities = {}

for model_name, model in models_to_test.items():
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Training
        print("  Training...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Prediction
        print("  Predicting...")
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        prediction_time = time.time() - start_time
        
        # Get probability scores for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test_scaled)
        else:
            y_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
        
        total_time = training_time + prediction_time
        
        # Store results
        results['Model'].append(model_name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1)
        results['ROC-AUC'].append(roc_auc)
        results['Training Time (s)'].append(training_time)
        results['Prediction Time (s)'].append(prediction_time)
        results['Total Time (s)'].append(total_time)
        
        # Store predictions
        predictions[model_name] = y_pred
        probabilities[model_name] = y_proba
        
        # Print results
        print(f"\n  Results:")
        print(f"    Accuracy:        {accuracy:.4f}")
        print(f"    Precision:       {precision:.4f}")
        print(f"    Recall:          {recall:.4f}")
        print(f"    F1-Score:        {f1:.4f}")
        print(f"    ROC-AUC:         {roc_auc:.4f}")
        print(f"    Training Time:   {training_time:.3f}s")
        print(f"    Prediction Time: {prediction_time:.3f}s")
        print(f"    Total Time:      {total_time:.3f}s")
        
    except Exception as e:
        print(f"  ✗ Error training {model_name}: {str(e)}")
        continue

# ========================================
# STEP 4: Results Summary
# ========================================
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("CLASSICAL BASELINE RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(RESULTS_DIR, 'classical_baseline_results.csv'), index=False)
print(f"\n✓ Results saved to {RESULTS_DIR}/classical_baseline_results.csv")

# ========================================
# STEP 5: Best Model Analysis
# ========================================
print("\n" + "="*80)
print("BEST MODEL ANALYSIS")
print("="*80)

# Find best model by F1-score
best_idx = results_df['F1-Score'].idxmax()
best_model_info = results_df.iloc[best_idx]
best_model_name = best_model_info['Model']

print(f"\nBest Model: {best_model_name}")
print(f"  Accuracy:  {best_model_info['Accuracy']:.4f}")
print(f"  Precision: {best_model_info['Precision']:.4f}")
print(f"  Recall:    {best_model_info['Recall']:.4f}")
print(f"  F1-Score:  {best_model_info['F1-Score']:.4f}")
print(f"  ROC-AUC:   {best_model_info['ROC-AUC']:.4f}")

# Confusion Matrix
y_pred_best = predictions[best_model_name]
cm = confusion_matrix(y_test, y_pred_best)

print(f"\nConfusion Matrix:")
print(cm)
print(f"\n  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Attack']))

# ========================================
# STEP 6: Comprehensive Visualization
# ========================================
print("\n[STEP 6] Creating visualizations...")

fig = plt.figure(figsize=(20, 14))

# 1. Performance Metrics Comparison
ax1 = plt.subplot(3, 3, 1)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1.1])

# 2. Training Time Comparison
ax2 = plt.subplot(3, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
bars = ax2.barh(range(len(results_df)), results_df['Training Time (s)'], color=colors)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['Model'], fontsize=9)
ax2.set_xlabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Highlight best model
best_bar_idx = results_df[results_df['Model'] == best_model_name].index[0]
bars[best_bar_idx].set_edgecolor('red')
bars[best_bar_idx].set_linewidth(3)

# 3. Confusion Matrix Heatmap (Best Model)
ax3 = plt.subplot(3, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. ROC Curves
ax4 = plt.subplot(3, 3, 4)

for model_name in results_df['Model']:
    if model_name in probabilities:
        try:
            y_proba = probabilities[model_name]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            linestyle = '-' if model_name == best_model_name else '--'
            linewidth = 3 if model_name == best_model_name else 1
            
            ax4.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})',
                    linestyle=linestyle, linewidth=linewidth)
        except:
            continue

ax4.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curves - All Models', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. F1-Score vs Training Time
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(results_df['Training Time (s)'], results_df['F1-Score'],
                     s=300, c=results_df['ROC-AUC'], cmap='viridis',
                     alpha=0.7, edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax5.annotate(row['Model'], 
                (row['Training Time (s)'], row['F1-Score']),
                fontsize=7, ha='center', va='bottom')

ax5.set_xlabel('Training Time (seconds)')
ax5.set_ylabel('F1-Score')
ax5.set_title('F1-Score vs Training Time', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('ROC-AUC', rotation=270, labelpad=15)

# 6. Radar Chart for Best Model
ax6 = plt.subplot(3, 3, 6, projection='polar')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [best_model_info[m] for m in metrics]
values += values[:1]  # Complete the circle

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

ax6.plot(angles, values, 'o-', linewidth=2, color='blue', label=best_model_name)
ax6.fill(angles, values, alpha=0.25, color='blue')
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics, fontsize=9)
ax6.set_ylim(0, 1)
ax6.set_title(f'Performance Radar\n{best_model_name}', fontsize=12, fontweight='bold', pad=20)
ax6.grid(True)

# 7. Model Comparison Heatmap
ax7 = plt.subplot(3, 3, 7)
metrics_matrix = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].T
sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7,
            xticklabels=results_df['Model'], yticklabels=metrics_matrix.index,
            cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
ax7.set_title('Performance Heatmap', fontsize=12, fontweight='bold')
ax7.set_xlabel('')
plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# 8. Precision-Recall Scatter
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(results_df['Recall'], results_df['Precision'], s=200, 
           c=results_df['F1-Score'], cmap='plasma', alpha=0.7,
           edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax8.annotate(row['Model'], 
                (row['Recall'], row['Precision']),
                fontsize=7, ha='center', va='top')

ax8.set_xlabel('Recall')
ax8.set_ylabel('Precision')
ax8.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_xlim([0, 1.05])
ax8.set_ylim([0, 1.05])

# 9. Total Time vs F1-Score
ax9 = plt.subplot(3, 3, 9)
colors_f1 = results_df['F1-Score']
scatter = ax9.scatter(results_df['Total Time (s)'], results_df['Accuracy'],
                     s=300, c=colors_f1, cmap='coolwarm',
                     alpha=0.7, edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax9.annotate(row['Model'], 
                (row['Total Time (s)'], row['Accuracy']),
                fontsize=7, ha='center', va='bottom')

ax9.set_xlabel('Total Time (seconds)')
ax9.set_ylabel('Accuracy')
ax9.set_title('Efficiency vs Performance', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax9)
cbar.set_label('F1-Score', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'classical_baseline_comprehensive.png'), 
           dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive visualization saved")

# Additional individual plots for paper
# Confusion matrices for top 3 models
top_3_models = results_df.nlargest(3, 'F1-Score')['Model'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, model_name in enumerate(top_3_models):
    cm_model = confusion_matrix(y_test, predictions[model_name])
    sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    axes[idx].set_title(f'{model_name}')
    axes[idx].set_ylabel('True Label' if idx == 0 else '')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'top_3_confusion_matrices.png'),
           dpi=300, bbox_inches='tight')
print(f"✓ Top 3 models confusion matrices saved")

plt.close('all')

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("CLASSICAL BASELINE TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"  F1-Score: {best_model_info['F1-Score']:.4f}")
print(f"  Accuracy: {best_model_info['Accuracy']:.4f}")
print(f"  Training Time: {best_model_info['Training Time (s)']:.2f}s")
print(f"\nFiles saved in:")
print(f"  Results: {RESULTS_DIR}/classical_baseline_results.csv")
print(f"  Plots:   {FIGURES_DIR}/")
print(f"\nNext step: Run 03_quantum_kernel.py")
print("="*80)