"""
Quantum IDS Project - Quantum Kernel Implementation
File: src/03_quantum_kernel.py
Purpose: Implement and evaluate quantum kernel methods with GPU acceleration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve, auc)

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS PROJECT - QUANTUM KERNEL IMPLEMENTATION")
print("="*80)

# ========================================
# STEP 1: Load Processed Data
# ========================================
print("\n[STEP 1] Loading processed data...")

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Load data
X_train_quantum = np.load(os.path.join(PROCESSED_DIR, 'X_train_quantum.npy'))
X_test_quantum = np.load(os.path.join(PROCESSED_DIR, 'X_test_quantum.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

# Load configuration
with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

n_qubits = config['n_features']

print(f"✓ Data loaded successfully")
print(f"  Training samples: {len(X_train_quantum)}")
print(f"  Testing samples:  {len(X_test_quantum)}")
print(f"  Number of qubits: {n_qubits}")

# ========================================
# STEP 2: Setup Quantum Backend with GPU
# ========================================
print("\n[STEP 2] Setting up quantum backend...")

try:
    # Try GPU backend first
    backend_gpu = AerSimulator(method='statevector', device='GPU')
    backend = backend_gpu
    print("✓ GPU backend initialized (using RTX 3050)")
    using_gpu = True
except:
    # Fallback to CPU
    backend = AerSimulator(method='statevector', device='CPU')
    print("✓ CPU backend initialized (GPU not available)")
    using_gpu = False

# ========================================
# STEP 3: Define Quantum Feature Maps
# ========================================
print("\n[STEP 3] Defining quantum feature maps...")

feature_maps = {
    'ZZ-Linear-R1': ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear'),
    'ZZ-Linear-R2': ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear'),
    'ZZ-Full-R2': ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full'),
    'Pauli-Z-R2': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z']),
    'Pauli-ZZ-R2': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ']),
    'Pauli-ZZZ-R2': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ', 'ZZZ']),
}

print(f"✓ {len(feature_maps)} feature maps defined")

# Print circuit statistics
print("\nFeature Map Statistics:")
print("-" * 60)
for name, fm in feature_maps.items():
    print(f"  {name:20s} | Depth: {fm.depth():3d} | Gates: {fm.size():4d}")

# ========================================
# STEP 4: Visualize Feature Maps
# ========================================
print("\n[STEP 4] Visualizing quantum circuits...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (name, fm) in enumerate(feature_maps.items()):
    if idx < 6:
        fm.decompose().draw('mpl', ax=axes[idx], style='iqp')
        axes[idx].set_title(f'{name}\nDepth={fm.depth()}, Gates={fm.size()}')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'quantum_feature_maps.png'), 
           dpi=300, bbox_inches='tight')
print(f"✓ Feature map circuits saved")
plt.close()

# ========================================
# STEP 5: Quantum Kernel Training Function
# ========================================
def train_quantum_kernel_svm(X_train, X_test, y_train, y_test,
                             feature_map, feature_map_name, dataset_size, backend):
    """
    Train SVM with quantum kernel and return metrics
    """
    print(f"\n{'='*80}")
    print(f"Training: {feature_map_name} | Dataset size: {dataset_size}")
    print(f"{'='*80}")
    
    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Compute kernel matrices
    print("  Computing training kernel matrix...")
    start_time = time.time()
    try:
        kernel_train = quantum_kernel.evaluate(x_vec=X_train)
        kernel_train_time = time.time() - start_time
        print(f"  ✓ Training kernel computed in {kernel_train_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Error computing training kernel: {str(e)}")
        return None
    
    print("  Computing test kernel matrix...")
    start_time = time.time()
    try:
        kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        kernel_test_time = time.time() - start_time
        print(f"  ✓ Test kernel computed in {kernel_test_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Error computing test kernel: {str(e)}")
        return None
    
    # Train SVM with precomputed quantum kernel
    print("  Training SVM...")
    start_time = time.time()
    try:
        svm = SVC(kernel='precomputed')
        svm.fit(kernel_train, y_train)
        svm_training_time = time.time() - start_time
        print(f"  ✓ SVM trained in {svm_training_time:.3f}s")
    except Exception as e:
        print(f"  ✗ Error training SVM: {str(e)}")
        return None
    
    # Make predictions
    print("  Making predictions...")
    start_time = time.time()
    try:
        y_pred = svm.predict(kernel_test)
        prediction_time = time.time() - start_time
    except Exception as e:
        print(f"  ✗ Error making predictions: {str(e)}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Total time
    total_time = kernel_train_time + kernel_test_time + svm_training_time
    
    print(f"\n  Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Total Time: {total_time:.2f}s")
    
    # Get circuit statistics
    circuit_depth = feature_map.depth()
    num_gates = feature_map.size()
    
    return {
        'Feature Map': feature_map_name,
        'Dataset Size': dataset_size,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Kernel Train Time (s)': kernel_train_time,
        'Kernel Test Time (s)': kernel_test_time,
        'SVM Training Time (s)': svm_training_time,
        'Prediction Time (s)': prediction_time,
        'Total Time (s)': total_time,
        'Circuit Depth': circuit_depth,
        'Number of Gates': num_gates,
        'Number of Qubits': n_qubits,
        'Backend': 'GPU' if using_gpu else 'CPU'
    }, y_pred

# ========================================
# STEP 6: Run Quantum Experiments
# ========================================
print("\n[STEP 6] Running quantum kernel experiments...")

# Dataset sizes to test (start small, can increase with 16GB RAM)
dataset_sizes = [100, 250, 500, 1000]  # Can add 1500, 2000 if time permits

quantum_results = []
quantum_predictions = {}

for size in dataset_sizes:
    print(f"\n{'#'*80}")
    print(f"DATASET SIZE: {size}")
    print(f"{'#'*80}")
    
    # Sample data
    if size < len(X_train_quantum):
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train_quantum, y_train, 
            train_size=size, 
            stratify=y_train, 
            random_state=42
        )
        test_size = min(size // 2, len(X_test_quantum))
        X_test_sub, _, y_test_sub, _ = train_test_split(
            X_test_quantum, y_test,
            train_size=test_size,
            stratify=y_test,
            random_state=42
        )
    else:
        X_train_sub, y_train_sub = X_train_quantum, y_train
        X_test_sub, y_test_sub = X_test_quantum, y_test
    
    print(f"  Train: {len(X_train_sub)} samples, Test: {len(X_test_sub)} samples")
    
    # Test each feature map
    for fm_name, fm in feature_maps.items():
        try:
            result = train_quantum_kernel_svm(
                X_train_sub, X_test_sub, y_train_sub, y_test_sub,
                fm, fm_name, size, backend
            )
            
            if result is not None:
                metrics, y_pred = result
                quantum_results.append(metrics)
                quantum_predictions[f"{fm_name}_{size}"] = (y_pred, y_test_sub)
            
        except Exception as e:
            print(f"  ✗ Error with {fm_name}: {str(e)}")
            continue

# ========================================
# STEP 7: Save Quantum Results
# ========================================
print("\n[STEP 7] Saving quantum results...")

quantum_results_df = pd.DataFrame(quantum_results)

print("\n" + "="*80)
print("QUANTUM KERNEL RESULTS SUMMARY")
print("="*80)
print(quantum_results_df.to_string(index=False))

# Save results
quantum_results_df.to_csv(os.path.join(RESULTS_DIR, 'quantum_kernel_results.csv'), index=False)
print(f"\n✓ Results saved to {RESULTS_DIR}/quantum_kernel_results.csv")

# ========================================
# STEP 8: Quantum vs Classical Comparison
# ========================================
print("\n[STEP 8] Comparing quantum vs classical...")

# Load classical results
classical_results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classical_baseline_results.csv'))

# Get best classical result
best_classical = classical_results_df.loc[classical_results_df['F1-Score'].idxmax()]

print("\n" + "="*80)
print("QUANTUM vs CLASSICAL COMPARISON")
print("="*80)

print(f"\nBest Classical Model: {best_classical['Model']}")
print(f"  F1-Score: {best_classical['F1-Score']:.4f}")
print(f"  Accuracy: {best_classical['Accuracy']:.4f}")
print(f"  Training Time: {best_classical['Training Time (s)']:.2f}s")

print("\nBest Quantum Results by Dataset Size:")
for size in dataset_sizes:
    size_results = quantum_results_df[quantum_results_df['Dataset Size'] == size]
    if len(size_results) > 0:
        best_quantum = size_results.loc[size_results['F1-Score'].idxmax()]
        print(f"\n  Dataset Size: {size}")
        print(f"    Feature Map: {best_quantum['Feature Map']}")
        print(f"    F1-Score: {best_quantum['F1-Score']:.4f}")
        print(f"    Accuracy: {best_quantum['Accuracy']:.4f}")
        print(f"    Total Time: {best_quantum['Total Time (s)']:.2f}s")
        print(f"    Circuit Depth: {best_quantum['Circuit Depth']}")
        print(f"    Circuit Gates: {best_quantum['Number of Gates']}")
        
        # Compare with classical
        f1_diff = best_quantum['F1-Score'] - best_classical['F1-Score']
        acc_diff = best_quantum['Accuracy'] - best_classical['Accuracy']
        print(f"    vs Classical:")
        print(f"      F1 Difference: {f1_diff:+.4f}")
        print(f"      Acc Difference: {acc_diff:+.4f}")

# ========================================
# STEP 9: Comprehensive Visualization
# ========================================
print("\n[STEP 9] Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 15))

# 1. F1-Score across dataset sizes and feature maps
ax1 = plt.subplot(3, 3, 1)
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax1.plot(fm_data['Dataset Size'], fm_data['F1-Score'],
            marker='o', label=fm_name, linewidth=2, markersize=8)

# Add classical baseline
ax1.axhline(y=best_classical['F1-Score'], color='red', linestyle='--',
           linewidth=3, label=f"Classical Best: {best_classical['Model']}")
ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('F1-Score')
ax1.set_title('F1-Score: Quantum vs Classical', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.05])

# 2. Accuracy comparison
ax2 = plt.subplot(3, 3, 2)
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax2.plot(fm_data['Dataset Size'], fm_data['Accuracy'],
            marker='s', label=fm_name, linewidth=2, markersize=8)

ax2.axhline(y=best_classical['Accuracy'], color='red', linestyle='--',
           linewidth=3, label=f"Classical Best")
ax2.set_xlabel('Dataset Size')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy: Quantum vs Classical', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.05])

# 3. Total training time comparison
ax3 = plt.subplot(3, 3, 3)
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax3.plot(fm_data['Dataset Size'], fm_data['Total Time (s)'],
            marker='^', label=fm_name, linewidth=2, markersize=8)

ax3.axhline(y=best_classical['Training Time (s)'], color='red', linestyle='--',
           linewidth=3, label=f"Classical Best")
ax3.set_xlabel('Dataset Size')
ax3.set_ylabel('Total Time (seconds)')
ax3.set_title('Training Time: Quantum vs Classical', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# 4. Precision-Recall trade-off
ax4 = plt.subplot(3, 3, 4)
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax4.scatter(fm_data['Recall'], fm_data['Precision'],
               s=100, label=fm_name, alpha=0.7)

ax4.scatter(best_classical['Recall'], best_classical['Precision'],
           s=400, color='red', marker='*', edgecolors='black', linewidth=2,
           label=f"Classical Best", zorder=10)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1.05])
ax4.set_ylim([0, 1.05])

# 5. Circuit Complexity vs Performance
ax5 = plt.subplot(3, 3, 5)
# Get one entry per feature map (largest dataset)
largest_size = quantum_results_df['Dataset Size'].max()
largest_data = quantum_results_df[quantum_results_df['Dataset Size'] == largest_size]

scatter = ax5.scatter(largest_data['Circuit Depth'], largest_data['F1-Score'],
                     s=largest_data['Number of Gates'], c=largest_data['Total Time (s)'],
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)

for idx, row in largest_data.iterrows():
    ax5.annotate(row['Feature Map'],
                (row['Circuit Depth'], row['F1-Score']),
                fontsize=7, ha='center', va='bottom')

ax5.set_xlabel('Circuit Depth')
ax5.set_ylabel('F1-Score')
ax5.set_title(f'Circuit Complexity vs Performance\n(bubble size = gate count, color = time)',
             fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Training Time (s)', rotation=270, labelpad=15)

# 6. Feature Map Performance Heatmap
ax6 = plt.subplot(3, 3, 6)
heatmap_data = quantum_results_df.pivot_table(
    values='F1-Score',
    index='Feature Map',
    columns='Dataset Size'
)
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6,
           cbar_kws={'label': 'F1-Score'}, vmin=0.5, vmax=1.0)
ax6.set_title('F1-Score Heatmap', fontsize=12, fontweight='bold')
ax6.set_xlabel('Dataset Size')
ax6.set_ylabel('Feature Map')

# 7. Scalability Analysis
ax7 = plt.subplot(3, 3, 7)
# Time per sample
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    time_per_sample = fm_data['Total Time (s)'] / fm_data['Dataset Size']
    ax7.plot(fm_data['Dataset Size'], time_per_sample,
            marker='o', label=fm_name, linewidth=2, markersize=6)

ax7.set_xlabel('Dataset Size')
ax7.set_ylabel('Time per Sample (seconds)')
ax7.set_title('Scalability Analysis', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8, loc='best')
ax7.grid(True, alpha=0.3)
ax7.set_yscale('log')

# 8. Best Quantum vs Classical Bar Chart
ax8 = plt.subplot(3, 3, 8)
# Get best quantum for each dataset size
best_quantum_results = []
for size in dataset_sizes:
    size_data = quantum_results_df[quantum_results_df['Dataset Size'] == size]
    if len(size_data) > 0:
        best = size_data.loc[size_data['F1-Score'].idxmax()]
        best_quantum_results.append({
            'Size': size,
            'F1': best['F1-Score'],
            'FM': best['Feature Map']
        })

best_quantum_df = pd.DataFrame(best_quantum_results)

x = np.arange(len(best_quantum_df))
width = 0.35

bars1 = ax8.bar(x - width/2, best_quantum_df['F1'], width,
               label='Best Quantum', color='blue', alpha=0.7)
bars2 = ax8.bar(x + width/2, [best_classical['F1-Score']]*len(x), width,
               label='Best Classical', color='red', alpha=0.7)

ax8.set_xlabel('Dataset Size')
ax8.set_ylabel('F1-Score')
ax8.set_title('Best Quantum vs Best Classical', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(best_quantum_df['Size'])
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')
ax8.set_ylim([0.5, 1.05])

# Annotate quantum feature maps
for i, (idx, row) in enumerate(best_quantum_df.iterrows()):
    ax8.text(i - width/2, row['F1'] + 0.01, row['FM'].split('-')[0],
            ha='center', va='bottom', fontsize=7, rotation=90)

# 9. Performance Improvement Matrix
ax9 = plt.subplot(3, 3, 9)
improvement_matrix = []
for size in dataset_sizes:
    size_data = quantum_results_df[quantum_results_df['Dataset Size'] == size]
    improvements = []
    for fm_name in quantum_results_df['Feature Map'].unique():
        fm_data = size_data[size_data['Feature Map'] == fm_name]
        if len(fm_data) > 0:
            improvement = fm_data.iloc[0]['F1-Score'] - best_classical['F1-Score']
            improvements.append(improvement)
        else:
            improvements.append(np.nan)
    improvement_matrix.append(improvements)

improvement_df = pd.DataFrame(improvement_matrix,
                             columns=quantum_results_df['Feature Map'].unique(),
                             index=dataset_sizes)

sns.heatmap(improvement_df.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
           ax=ax9, cbar_kws={'label': 'F1-Score Difference'})
ax9.set_title('Quantum Advantage/Disadvantage\n(vs Best Classical)',
             fontsize=11, fontweight='bold')
ax9.set_xlabel('Dataset Size')
ax9.set_ylabel('Feature Map')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'quantum_comprehensive_analysis.png'),
           dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive analysis visualization saved")

plt.close('all')

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("QUANTUM KERNEL TRAINING COMPLETE!")
print("="*80)

# Find overall best quantum result
best_quantum_overall = quantum_results_df.loc[quantum_results_df['F1-Score'].idxmax()]

print(f"\nBest Quantum Result:")
print(f"  Feature Map: {best_quantum_overall['Feature Map']}")
print(f"  Dataset Size: {best_quantum_overall['Dataset Size']}")
print(f"  F1-Score: {best_quantum_overall['F1-Score']:.4f}")
print(f"  Accuracy: {best_quantum_overall['Accuracy']:.4f}")
print(f"  Total Time: {best_quantum_overall['Total Time (s)']:.2f}s")
print(f"  Backend: {best_quantum_overall['Backend']}")

print(f"\nComparison with Classical:")
f1_advantage = best_quantum_overall['F1-Score'] - best_classical['F1-Score']
acc_advantage = best_quantum_overall['Accuracy'] - best_classical['Accuracy']
time_ratio = best_quantum_overall['Total Time (s)'] / best_classical['Training Time (s)']

print(f"  F1-Score Difference: {f1_advantage:+.4f} ({f1_advantage/best_classical['F1-Score']*100:+.2f}%)")
print(f"  Accuracy Difference: {acc_advantage:+.4f} ({acc_advantage/best_classical['Accuracy']*100:+.2f}%)")
print(f"  Time Ratio: {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'}")

print(f"\nFiles saved in:")
print(f"  Results: {RESULTS_DIR}/quantum_kernel_results.csv")
print(f"  Plots:   {FIGURES_DIR}/")
print(f"\nNext step: Run 04_comparative_analysis.py for final analysis")
print("="*80)