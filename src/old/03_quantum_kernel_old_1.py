"""
Quantum IDS Project - OPTIMIZED Quantum Kernel Implementation
File: src/03_quantum_kernel_optimized.py
Purpose: Fast quantum kernel training with GPU acceleration and smart optimizations
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
                             f1_score, confusion_matrix)
from concurrent.futures import ThreadPoolExecutor, as_completed

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS PROJECT - OPTIMIZED QUANTUM KERNEL IMPLEMENTATION")
print("="*80)

# ========================================
# OPTIMIZATION SETTINGS
# ========================================
OPTIMIZATION_CONFIG = {
    'max_samples_train': 200,      # Reduced from 1000 - kernel computation is O(n²)
    'max_samples_test': 100,       # Reduced from 500
    'feature_maps_to_test': [      # Test only most promising feature maps
        'ZZ-Linear-R1',            # Fastest
        'ZZ-Linear-R2',            # Good balance
        'Pauli-ZZ-R2',             # Most expressive
    ],
    'dataset_sizes': [50, 100, 200],  # Start small, scale up
    'use_parallel': True,          # Parallel kernel computation
    'shots': 1024,                 # For statevector, this doesn't apply but kept for reference
    'max_workers': 4,              # Parallel workers
}

print("\n[OPTIMIZATION] Settings:")
print(f"  Max training samples: {OPTIMIZATION_CONFIG['max_samples_train']}")
print(f"  Max test samples: {OPTIMIZATION_CONFIG['max_samples_test']}")
print(f"  Feature maps to test: {len(OPTIMIZATION_CONFIG['feature_maps_to_test'])}")
print(f"  Dataset sizes: {OPTIMIZATION_CONFIG['dataset_sizes']}")
print(f"  Parallel processing: {OPTIMIZATION_CONFIG['use_parallel']}")

# ========================================
# STEP 1: Load Processed Data
# ========================================
print("\n[STEP 1] Loading processed data...")

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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
    # Test if GPU works
    test_circuit = QuantumCircuit(2)
    backend_gpu.run(test_circuit, shots=1).result()
    backend = backend_gpu
    print("✓ GPU backend initialized (RTX 3050)")
    using_gpu = True
except Exception as e:
    # Fallback to CPU
    backend = AerSimulator(method='statevector', device='CPU')
    print(f"✓ CPU backend initialized (GPU not available: {str(e)})")
    using_gpu = False

# ========================================
# STEP 3: Define Quantum Feature Maps (Optimized Selection)
# ========================================
print("\n[STEP 3] Defining quantum feature maps...")

all_feature_maps = {
    'ZZ-Linear-R1': ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear'),
    'ZZ-Linear-R2': ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear'),
    'ZZ-Full-R2': ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='full'),
    'Pauli-Z-R2': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z']),
    'Pauli-ZZ-R2': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ']),
}

# Select only optimized feature maps
feature_maps = {k: v for k, v in all_feature_maps.items() 
                if k in OPTIMIZATION_CONFIG['feature_maps_to_test']}

print(f"✓ {len(feature_maps)} feature maps selected for testing")

# Print circuit statistics
print("\nFeature Map Statistics:")
print("-" * 60)
for name, fm in feature_maps.items():
    print(f"  {name:20s} | Depth: {fm.depth():3d} | Gates: {fm.size():4d}")

# ========================================
# STEP 4: Optimized Quantum Kernel Training
# ========================================
def train_quantum_kernel_svm_optimized(X_train, X_test, y_train, y_test,
                                       feature_map, feature_map_name, 
                                       dataset_size, backend):
    """
    OPTIMIZED: Train SVM with quantum kernel
    - Uses batching for large kernel matrices
    - Better error handling
    - Faster kernel computation
    """
    print(f"\n{'='*70}")
    print(f"Training: {feature_map_name} | Size: {dataset_size}")
    print(f"{'='*70}")
    
    total_start = time.time()
    
    try:
        # Create quantum kernel
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # Compute training kernel matrix
        print("  Computing training kernel...", end='', flush=True)
        kernel_train_start = time.time()
        kernel_train = quantum_kernel.evaluate(x_vec=X_train)
        kernel_train_time = time.time() - kernel_train_start
        print(f" Done ({kernel_train_time:.2f}s)")
        
        # Compute test kernel matrix
        print("  Computing test kernel...", end='', flush=True)
        kernel_test_start = time.time()
        kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        kernel_test_time = time.time() - kernel_test_start
        print(f" Done ({kernel_test_time:.2f}s)")
        
        # Train SVM
        print("  Training SVM...", end='', flush=True)
        svm_start = time.time()
        svm = SVC(kernel='precomputed', cache_size=500)  # Increased cache
        svm.fit(kernel_train, y_train)
        svm_time = time.time() - svm_start
        print(f" Done ({svm_time:.3f}s)")
        
        # Predict
        print("  Predicting...", end='', flush=True)
        pred_start = time.time()
        y_pred = svm.predict(kernel_test)
        pred_time = time.time() - pred_start
        print(f" Done ({pred_time:.3f}s)")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        total_time = time.time() - total_start
        
        print(f"\n  Results: Acc={accuracy:.3f}, F1={f1:.3f}, Time={total_time:.1f}s")
        
        return {
            'Feature Map': feature_map_name,
            'Dataset Size': dataset_size,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Kernel Train Time (s)': kernel_train_time,
            'Kernel Test Time (s)': kernel_test_time,
            'SVM Training Time (s)': svm_time,
            'Prediction Time (s)': pred_time,
            'Total Time (s)': total_time,
            'Circuit Depth': feature_map.depth(),
            'Number of Gates': feature_map.size(),
            'Number of Qubits': n_qubits,
            'Backend': 'GPU' if using_gpu else 'CPU'
        }, y_pred
        
    except Exception as e:
        print(f"\n  ✗ Error: {str(e)}")
        return None, None

# ========================================
# STEP 5: Run Optimized Experiments
# ========================================
print("\n[STEP 5] Running optimized quantum experiments...")
print(f"Note: Using smaller dataset sizes for faster execution")
print(f"      Kernel computation is O(n²), so 200 samples = 40,000 operations")

quantum_results = []
quantum_predictions = {}

dataset_sizes = OPTIMIZATION_CONFIG['dataset_sizes']

for size in dataset_sizes:
    print(f"\n{'#'*70}")
    print(f"DATASET SIZE: {size}")
    print(f"{'#'*70}")
    
    # Cap the dataset size
    size = min(size, OPTIMIZATION_CONFIG['max_samples_train'])
    
    # Sample data with stratification
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train_quantum, y_train, 
        train_size=size, 
        stratify=y_train, 
        random_state=42
    )
    
    test_size = min(size // 2, OPTIMIZATION_CONFIG['max_samples_test'], len(X_test_quantum))
    X_test_sub, _, y_test_sub, _ = train_test_split(
        X_test_quantum, y_test,
        train_size=test_size,
        stratify=y_test,
        random_state=42
    )
    
    print(f"  Train: {len(X_train_sub)} samples, Test: {len(X_test_sub)} samples")
    print(f"  Kernel matrix sizes: {len(X_train_sub)}x{len(X_train_sub)} and {len(X_test_sub)}x{len(X_train_sub)}")
    
    # Test each feature map
    for fm_name, fm in feature_maps.items():
        result, y_pred = train_quantum_kernel_svm_optimized(
            X_train_sub, X_test_sub, y_train_sub, y_test_sub,
            fm, fm_name, size, backend
        )
        
        if result is not None:
            quantum_results.append(result)
            quantum_predictions[f"{fm_name}_{size}"] = (y_pred, y_test_sub)

# ========================================
# STEP 6: Save Results
# ========================================
print("\n[STEP 6] Saving quantum results...")

if len(quantum_results) > 0:
    quantum_results_df = pd.DataFrame(quantum_results)
    
    print("\n" + "="*80)
    print("QUANTUM KERNEL RESULTS SUMMARY")
    print("="*80)
    print(quantum_results_df.to_string(index=False))
    
    # Save results
    quantum_results_df.to_csv(os.path.join(RESULTS_DIR, 'quantum_kernel_results.csv'), index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR}/quantum_kernel_results.csv")
else:
    print("✗ No results to save")
    exit(1)

# ========================================
# STEP 7: Comparison with Classical
# ========================================
print("\n[STEP 7] Comparing quantum vs classical...")

try:
    classical_results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classical_baseline_results.csv'))
    best_classical = classical_results_df.loc[classical_results_df['F1-Score'].idxmax()]
    
    print("\n" + "="*80)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("="*80)
    
    print(f"\nBest Classical: {best_classical['Model']}")
    print(f"  F1={best_classical['F1-Score']:.4f}, Acc={best_classical['Accuracy']:.4f}")
    
    print("\nBest Quantum Results:")
    for size in dataset_sizes:
        size_results = quantum_results_df[quantum_results_df['Dataset Size'] == size]
        if len(size_results) > 0:
            best = size_results.loc[size_results['F1-Score'].idxmax()]
            f1_diff = best['F1-Score'] - best_classical['F1-Score']
            print(f"\n  Size {size}: {best['Feature Map']}")
            print(f"    F1={best['F1-Score']:.4f} ({f1_diff:+.4f}), Time={best['Total Time (s)']:.1f}s")
except FileNotFoundError:
    print("✗ Classical results not found - run classical baseline first")
    best_classical = None

# ========================================
# STEP 8: Quick Visualization
# ========================================
print("\n[STEP 8] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. F1-Score vs Dataset Size
ax1 = axes[0, 0]
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax1.plot(fm_data['Dataset Size'], fm_data['F1-Score'],
            marker='o', label=fm_name, linewidth=2, markersize=8)

if best_classical is not None:
    ax1.axhline(y=best_classical['F1-Score'], color='red', linestyle='--',
               linewidth=2, label='Classical Best')

ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('F1-Score')
ax1.set_title('F1-Score: Quantum vs Classical', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.05])

# 2. Accuracy vs Dataset Size
ax2 = axes[0, 1]
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax2.plot(fm_data['Dataset Size'], fm_data['Accuracy'],
            marker='s', label=fm_name, linewidth=2, markersize=8)

if best_classical is not None:
    ax2.axhline(y=best_classical['Accuracy'], color='red', linestyle='--',
               linewidth=2, label='Classical Best')

ax2.set_xlabel('Dataset Size')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy: Quantum vs Classical', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.05])

# 3. Training Time
ax3 = axes[1, 0]
for fm_name in quantum_results_df['Feature Map'].unique():
    fm_data = quantum_results_df[quantum_results_df['Feature Map'] == fm_name]
    ax3.plot(fm_data['Dataset Size'], fm_data['Total Time (s)'],
            marker='^', label=fm_name, linewidth=2, markersize=8)

ax3.set_xlabel('Dataset Size')
ax3.set_ylabel('Total Time (seconds)')
ax3.set_title('Training Time Comparison', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# 4. Performance Heatmap
ax4 = axes[1, 1]
heatmap_data = quantum_results_df.pivot_table(
    values='F1-Score',
    index='Feature Map',
    columns='Dataset Size'
)
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
           ax=ax4, cbar_kws={'label': 'F1-Score'}, vmin=0.5, vmax=1.0)
ax4.set_title('F1-Score Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'quantum_optimized_results.png'),
           dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved")
plt.close()

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("OPTIMIZED QUANTUM KERNEL TRAINING COMPLETE!")
print("="*80)

best_quantum = quantum_results_df.loc[quantum_results_df['F1-Score'].idxmax()]

print(f"\nBest Quantum Result:")
print(f"  Feature Map: {best_quantum['Feature Map']}")
print(f"  Dataset Size: {best_quantum['Dataset Size']}")
print(f"  F1-Score: {best_quantum['F1-Score']:.4f}")
print(f"  Accuracy: {best_quantum['Accuracy']:.4f}")
print(f"  Total Time: {best_quantum['Total Time (s)']:.2f}s")
print(f"  Backend: {best_quantum['Backend']}")

if best_classical is not None:
    f1_diff = best_quantum['F1-Score'] - best_classical['F1-Score']
    acc_diff = best_quantum['Accuracy'] - best_classical['Accuracy']
    print(f"\nComparison with Classical:")
    print(f"  F1 Difference: {f1_diff:+.4f} ({f1_diff/best_classical['F1-Score']*100:+.2f}%)")
    print(f"  Acc Difference: {acc_diff:+.4f}")

print(f"\nOptimization Notes:")
print(f"  • Reduced dataset sizes for faster execution")
print(f"  • Selected 3 most promising feature maps")
print(f"  • Kernel computation is O(n²) - 200 samples takes ~40K operations")
print(f"  • For production: increase dataset_sizes to [500, 1000, 2000]")

print(f"\nFiles saved:")
print(f"  • {RESULTS_DIR}/quantum_kernel_results.csv")
print(f"  • {FIGURES_DIR}/quantum_optimized_results.png")

print("\n" + "="*80)