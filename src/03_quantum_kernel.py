"""
Quantum IDS Project - DEBUG VERSION
File: src/03_quantum_kernel_debug.py

This version adds extensive debugging to identify why experiments are failing.
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
                             f1_score, confusion_matrix, roc_auc_score)

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM IDS - DEBUG MODE")
print("="*80)

# ========================================
# CONFIGURATION
# ========================================
FAST_CONFIG = {
    'dataset_sizes': [50, 100, 200],
    'feature_maps_to_test': ['ZZ-R1', 'ZZ-R2', 'Pauli-ZZ'],
    'n_repetitions': 1,
    'use_gpu': True,
    'batch_kernel': True,
    'max_test_samples': 150,
}

# ========================================
# STEP 1: Load Data with Validation
# ========================================
print("\n[STEP 1] Loading data...")

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

try:
    X_train_quantum = np.load(os.path.join(PROCESSED_DIR, 'X_train_quantum.npy'))
    X_test_quantum = np.load(os.path.join(PROCESSED_DIR, 'X_test_quantum.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    
    print(f"✓ X_train_quantum: shape={X_train_quantum.shape}, dtype={X_train_quantum.dtype}")
    print(f"✓ X_test_quantum: shape={X_test_quantum.shape}, dtype={X_test_quantum.dtype}")
    print(f"✓ y_train: shape={y_train.shape}, unique values={np.unique(y_train)}")
    print(f"✓ y_test: shape={y_test.shape}, unique values={np.unique(y_test)}")
    
    # Check for NaN or Inf
    if np.any(np.isnan(X_train_quantum)) or np.any(np.isinf(X_train_quantum)):
        print("⚠️  WARNING: X_train_quantum contains NaN or Inf values!")
    if np.any(np.isnan(X_test_quantum)) or np.any(np.isinf(X_test_quantum)):
        print("⚠️  WARNING: X_test_quantum contains NaN or Inf values!")
    
    with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    n_qubits = config['n_features']
    print(f"✓ Config loaded: {n_qubits} qubits")
    
except Exception as e:
    print(f"✗ ERROR loading data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========================================
# STEP 2: Quantum Backend Setup
# ========================================
print("\n[STEP 2] Setting up quantum backend...")

using_gpu = False
try:
    if FAST_CONFIG['use_gpu']:
        print("  Attempting GPU backend...")
        backend = AerSimulator(method='statevector', device='GPU')
        
        # Test with a simple circuit
        test_circuit = QuantumCircuit(2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        test_result = backend.run(test_circuit, shots=100).result()
        
        using_gpu = True
        print("  ✓ GPU backend active")
    else:
        raise Exception("GPU disabled in config")
        
except Exception as e:
    print(f"  ⚠️  GPU failed ({e}), falling back to CPU")
    backend = AerSimulator(method='statevector', device='CPU')
    using_gpu = False
    print("  ✓ CPU backend active")

print(f"  Backend: {backend}")
print(f"  Device: {'GPU' if using_gpu else 'CPU'}")

# ========================================
# STEP 3: Feature Maps
# ========================================
print("\n[STEP 3] Defining feature maps...")

try:
    all_feature_maps = {
        'ZZ-R1': ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear'),
        'ZZ-R2': ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear'),
        'Pauli-ZZ': PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ']),
    }
    
    feature_maps = {k: v for k, v in all_feature_maps.items() 
                    if k in FAST_CONFIG['feature_maps_to_test']}
    
    print(f"✓ Testing {len(feature_maps)} feature maps:")
    for name, fm in feature_maps.items():
        print(f"  • {name}: Depth={fm.depth()}, Gates={fm.size()}, Qubits={fm.num_qubits}")
        
except Exception as e:
    print(f"✗ ERROR creating feature maps: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========================================
# STEP 4: Training Function with Debug
# ========================================
def train_quantum_debug(X_train, X_test, y_train, y_test, feature_map, fm_name, size):
    """Training function with extensive debugging"""
    print(f"\n{'='*70}")
    print(f"[{fm_name}] Size: {size}")
    print(f"{'='*70}")
    
    start_total = time.time()
    
    try:
        # Validate inputs
        print(f"  Input validation:")
        print(f"    X_train: {X_train.shape}, range=[{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"    X_test: {X_test.shape}, range=[{X_test.min():.3f}, {X_test.max():.3f}]")
        print(f"    y_train: {y_train.shape}, values={np.unique(y_train)}")
        print(f"    y_test: {y_test.shape}, values={np.unique(y_test)}")
        
        # Create kernel
        print(f"  Creating kernel...")
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        print(f"    ✓ Kernel created")
        
        # Compute training kernel
        print(f"  Computing K_train ({len(X_train)}x{len(X_train)})...", end='', flush=True)
        t1 = time.time()
        K_train = kernel.evaluate(x_vec=X_train)
        kernel_train_time = time.time() - t1
        print(f" {kernel_train_time:.1f}s")
        print(f"    K_train: shape={K_train.shape}, range=[{K_train.min():.3f}, {K_train.max():.3f}]")
        
        # Check for issues
        if np.any(np.isnan(K_train)):
            print(f"    ⚠️  K_train contains NaN!")
        if np.any(np.isinf(K_train)):
            print(f"    ⚠️  K_train contains Inf!")
        
        # Compute test kernel
        print(f"  Computing K_test ({len(X_test)}x{len(X_train)})...", end='', flush=True)
        t2 = time.time()
        K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
        kernel_test_time = time.time() - t2
        print(f" {kernel_test_time:.1f}s")
        print(f"    K_test: shape={K_test.shape}, range=[{K_test.min():.3f}, {K_test.max():.3f}]")
        
        # Train SVM
        print(f"  Training SVM...", end='', flush=True)
        t3 = time.time()
        svm = SVC(kernel='precomputed', cache_size=500, probability=True)
        svm.fit(K_train, y_train)
        train_time = time.time() - t3
        print(f" {train_time:.2f}s")
        print(f"    Support vectors: {svm.n_support_}")
        
        # Predict
        print(f"  Predicting...", end='', flush=True)
        t4 = time.time()
        y_pred = svm.predict(K_test)
        y_proba = svm.predict_proba(K_test)[:, 1]
        pred_time = time.time() - t4
        print(f" {pred_time:.2f}s")
        print(f"    Predictions: {np.unique(y_pred, return_counts=True)}")
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception as auc_e:
            print(f"    ⚠️  ROC-AUC error: {auc_e}")
            auc = np.nan
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        total_time = time.time() - start_total
        
        print(f"\n  ✓ METRICS:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    Time:      {total_time:.1f}s")
        
        result = {
            'Feature Map': fm_name,
            'Dataset Size': size,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Kernel Time (s)': kernel_train_time + kernel_test_time,
            'SVM Time (s)': train_time,
            'Prediction Time (s)': pred_time,
            'Total Time (s)': total_time,
            'Circuit Depth': feature_map.depth(),
            'Gates': feature_map.size(),
            'Qubits': n_qubits,
            'Backend': 'GPU' if using_gpu else 'CPU',
        }
        
        return result
        
    except Exception as e:
        print(f"\n  ✗ EXCEPTION CAUGHT: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# STEP 5: Run Single Test First
# ========================================
print("\n[STEP 4] Running single test experiment...")

try:
    # Use smallest size for initial test
    test_size = FAST_CONFIG['dataset_sizes'][0]
    
    print(f"\n  Sampling {test_size} training samples...")
    X_tr, _, y_tr, _ = train_test_split(
        X_train_quantum, y_train,
        train_size=min(test_size, len(X_train_quantum)),
        stratify=y_train,
        random_state=42
    )
    
    test_samples = min(test_size // 2, FAST_CONFIG['max_test_samples'], len(X_test_quantum))
    print(f"  Sampling {test_samples} test samples...")
    X_te, _, y_te, _ = train_test_split(
        X_test_quantum, y_test,
        train_size=test_samples,
        stratify=y_test,
        random_state=42
    )
    
    # Test with first feature map
    fm_name = list(feature_maps.keys())[0]
    fm = feature_maps[fm_name]
    
    print(f"\n  Testing with {fm_name}...")
    result = train_quantum_debug(X_tr, X_te, y_tr, y_te, fm, fm_name, test_size)
    
    if result:
        print("\n  ✓ SINGLE TEST SUCCESSFUL!")
        print(f"    Result: {result}")
    else:
        print("\n  ✗ SINGLE TEST FAILED!")
        print("    Check the error messages above.")
        exit(1)
        
except Exception as e:
    print(f"\n✗ Test experiment failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========================================
# STEP 6: Run All Experiments
# ========================================
print("\n[STEP 5] Running all experiments...")

results = []
experiment_start = time.time()

for i, size in enumerate(FAST_CONFIG['dataset_sizes']):
    print(f"\n{'#'*70}")
    print(f"DATASET SIZE: {size} ({i+1}/{len(FAST_CONFIG['dataset_sizes'])})")
    print(f"{'#'*70}")
    
    try:
        # Sample data
        X_tr, _, y_tr, _ = train_test_split(
            X_train_quantum, y_train,
            train_size=min(size, len(X_train_quantum)),
            stratify=y_train,
            random_state=42
        )
        
        test_size = min(size // 2, FAST_CONFIG['max_test_samples'], len(X_test_quantum))
        X_te, _, y_te, _ = train_test_split(
            X_test_quantum, y_test,
            train_size=test_size,
            stratify=y_test,
            random_state=42
        )
        
        # Test each feature map
        for j, (fm_name, fm) in enumerate(feature_maps.items()):
            result = train_quantum_debug(X_tr, X_te, y_tr, y_te, fm, fm_name, size)
            if result:
                results.append(result)
                print(f"  ✓ Result added to list (total: {len(results)})")
            else:
                print(f"  ✗ Result was None, not added")
                
    except Exception as e:
        print(f"\n✗ Error in size {size}: {e}")
        import traceback
        traceback.print_exc()
        continue

total_experiment_time = time.time() - experiment_start

print(f"\n{'='*70}")
print(f"EXPERIMENTS COMPLETE!")
print(f"Total experiments: {len(results)}")
print(f"Total time: {total_experiment_time/60:.1f} minutes")
print(f"{'='*70}")

# ========================================
# STEP 7: Save Results
# ========================================
print("\n[STEP 6] Saving results...")

if len(results) == 0:
    print("✗ NO RESULTS TO SAVE!")
    print("  All experiments failed. Check the error messages above.")
    exit(1)

df = pd.DataFrame(results)

print("\n" + "="*80)
print("QUANTUM KERNEL RESULTS")
print("="*80)
print(df.to_string(index=False))

df.to_csv(os.path.join(RESULTS_DIR, 'quantum_kernel_results.csv'), index=False)
print(f"\n✓ Results saved to {RESULTS_DIR}/quantum_kernel_results.csv")

print("\n" + "="*80)
print("DEBUG COMPLETE!")
print("="*80)
print(f"\n✓ {len(results)} experiments completed successfully")
print(f"✓ Results saved")
print("\nIf you see this message, the quantum kernel code is working!")
print("You can now run the full version (03_quantum_kernel_fast.py)")