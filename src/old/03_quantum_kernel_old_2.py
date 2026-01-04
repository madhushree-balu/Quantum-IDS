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
                             f1_score, confusion_matrix, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CPU_OPTIMAL_CONFIG = {
    'n_qubits': 6,
    'train_size': 2000,
    'test_size': 500,
    'optimal_C': 100.0,
    'optimal_gamma': 'scale',
    'feature_map': 'ZZ-Optimized',
    'use_pca_rotation': True,
    'use_quantile_transform': True,
    'standardize_first': True,
    'use_cpu_parallel': True,
    'max_cpu_threads': -1,
    'transpile_optimization_level': 3,
    'chunk_kernel_computation': True,
    'kernel_chunk_size': 200,
    'target_accuracy': 0.98,
    'early_stop_enabled': True,
    'max_runtime_hours': 6,
}

print("="*80)
print("QUANTUM IDS - CPU-OPTIMIZED (16GB RAM, >0.98 ACCURACY)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Qubits:       {CPU_OPTIMAL_CONFIG['n_qubits']} (CPU optimized)")
print(f"  Train size:   {CPU_OPTIMAL_CONFIG['train_size']}")
print(f"  Test size:    {CPU_OPTIMAL_CONFIG['test_size']}")
print(f"  Target acc:   {CPU_OPTIMAL_CONFIG['target_accuracy']}")
print(f"  Max runtime:  {CPU_OPTIMAL_CONFIG['max_runtime_hours']}h")
print(f"  CPU threads:  All available cores")

start_time_global = time.time()

print("\n[STEP 1] Loading preprocessed data...")

PROCESSED_DIR = 'data/processed'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

try:
    X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    
    print(f"✓ Loaded: Train={X_train.shape}, Test={X_test.shape}")
    
    with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
        prep_config = json.load(f)
    
    with open(os.path.join(PROCESSED_DIR, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f]
    
    n_classes = len(class_names)
    is_binary = n_classes == 2
    
    print(f"✓ Classes: {n_classes} - {', '.join(class_names)}")
    print(f"✓ Original features: {X_train.shape[1]}")
    
except Exception as e:
    print(f"✗ ERROR loading data: {e}")
    print("\n  Make sure you've run 01_preprocess_data.py first!")
    exit(1)

print("\n[STEP 2] Advanced feature engineering for quantum encoding...")

if CPU_OPTIMAL_CONFIG['standardize_first']:
    print("  Re-standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("  ✓ Standardization complete")

if X_train.shape[1] != CPU_OPTIMAL_CONFIG['n_qubits']:
    print(f"  Reducing features: {X_train.shape[1]} → {CPU_OPTIMAL_CONFIG['n_qubits']} (PCA)")
    
    pca = PCA(n_components=CPU_OPTIMAL_CONFIG['n_qubits'], random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  ✓ PCA variance retained: {explained_var:.4f}")
    
    X_train = X_train_pca
    X_test = X_test_pca

n_qubits = CPU_OPTIMAL_CONFIG['n_qubits']

if CPU_OPTIMAL_CONFIG['use_quantile_transform']:
    print("  Applying quantile transformation...")
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train = qt.fit_transform(X_train)
    X_test = qt.transform(X_test)
    print("  ✓ Quantile transform applied")

print("  Scaling to [0, π] for quantum states...")
X_train_quantum = X_train * np.pi
X_test_quantum = X_test * np.pi

X_train_quantum = X_train_quantum / (np.linalg.norm(X_train_quantum, axis=1, keepdims=True) + 1e-10)
X_test_quantum = X_test_quantum / (np.linalg.norm(X_test_quantum, axis=1, keepdims=True) + 1e-10)

print(f"  ✓ Final data shape: {X_train_quantum.shape}")
print(f"  ✓ Data range: [{X_train_quantum.min():.3f}, {X_train_quantum.max():.3f}]")

print(f"\n[STEP 3] Strategic sampling (target: {CPU_OPTIMAL_CONFIG['train_size']} train)...")

if len(X_train_quantum) > CPU_OPTIMAL_CONFIG['train_size']:
    print(f"  Sampling from {len(X_train_quantum)} → {CPU_OPTIMAL_CONFIG['train_size']}...")
    
    X_train_quantum, _, y_train, _ = train_test_split(
        X_train_quantum, y_train,
        train_size=CPU_OPTIMAL_CONFIG['train_size'],
        stratify=y_train,
        random_state=42
    )
    print(f"  ✓ Train sampled: {len(X_train_quantum)}")

if len(X_test_quantum) > CPU_OPTIMAL_CONFIG['test_size']:
    X_test_quantum, _, y_test, _ = train_test_split(
        X_test_quantum, y_test,
        train_size=CPU_OPTIMAL_CONFIG['test_size'],
        stratify=y_test,
        random_state=42
    )
    print(f"  ✓ Test sampled: {len(X_test_quantum)}")

print(f"\n  Final dataset:")
print(f"    Train: {len(X_train_quantum)} samples")
print(f"    Test:  {len(X_test_quantum)} samples")

print(f"\n  Class distribution:")
for i, name in enumerate(class_names):
    train_count = np.sum(y_train == i)
    test_count = np.sum(y_test == i)
    print(f"    {name:12s}: Train={train_count:4d}, Test={test_count:4d}")

print("\n[STEP 4] Initializing CPU backend...")

backend = AerSimulator(
    method='statevector',
    device='CPU',
    max_parallel_threads=CPU_OPTIMAL_CONFIG['max_cpu_threads']
)

print("  ✓ CPU backend active")
print(f"    All available cores enabled")
print(f"    16GB RAM available")

print(f"\n[STEP 5] Creating optimized feature map...")

feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=3,
    entanglement='full',
    insert_barriers=False
)

print(f"✓ Feature map created: ZZ-Optimized")
print(f"  • Qubits: {n_qubits}")
print(f"  • Depth: {feature_map.depth()}")
print(f"  • Gates: {feature_map.size()}")
print(f"  • Reps: 3 (optimal for accuracy)")

def train_quantum_cpu_optimized(X_train, X_test, y_train, y_test, feature_map):
    result = {
        'Success': False,
        'Error': None
    }
    
    print(f"\n  {'─'*76}")
    print(f"  Training Quantum Model")
    print(f"  {'─'*76}")
    
    try:
        start = time.time()
        
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        print(f"    [1/4] Computing training kernel ({len(X_train)}x{len(X_train)})...")
        t1 = time.time()
        
        if CPU_OPTIMAL_CONFIG['chunk_kernel_computation']:
            chunk_size = CPU_OPTIMAL_CONFIG['kernel_chunk_size']
            n_chunks = (len(X_train) + chunk_size - 1) // chunk_size
            
            K_train = np.zeros((len(X_train), len(X_train)))
            
            with tqdm(total=n_chunks, desc="       Computing chunks") as pbar:
                for i in range(n_chunks):
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, len(X_train))
                    
                    K_batch = kernel.evaluate(x_vec=X_train[start_i:end_i], y_vec=X_train)
                    K_train[start_i:end_i, :] = K_batch
                    
                    pbar.update(1)
                    
                    if i % 5 == 0:
                        import gc
                        gc.collect()
        else:
            K_train = kernel.evaluate(x_vec=X_train)
        
        kernel_train_time = time.time() - t1
        print(f"    ✓ Train kernel: {kernel_train_time:.1f}s ({kernel_train_time/60:.1f}m)")
        
        print(f"    [2/4] Computing test kernel ({len(X_test)}x{len(X_train)})...")
        t2 = time.time()
        
        if CPU_OPTIMAL_CONFIG['chunk_kernel_computation']:
            chunk_size = CPU_OPTIMAL_CONFIG['kernel_chunk_size']
            n_chunks = (len(X_test) + chunk_size - 1) // chunk_size
            
            K_test = np.zeros((len(X_test), len(X_train)))
            
            with tqdm(total=n_chunks, desc="       Computing chunks") as pbar:
                for i in range(n_chunks):
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, len(X_test))
                    
                    K_batch = kernel.evaluate(x_vec=X_test[start_i:end_i], y_vec=X_train)
                    K_test[start_i:end_i, :] = K_batch
                    
                    pbar.update(1)
        else:
            K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
        kernel_test_time = time.time() - t2
        print(f"    ✓ Test kernel: {kernel_test_time:.1f}s ({kernel_test_time/60:.1f}m)")
        
        print(f"    [3/4] Training SVM (C={CPU_OPTIMAL_CONFIG['optimal_C']})...")
        t3 = time.time()
        
        svm = SVC(
            kernel='precomputed',
            C=CPU_OPTIMAL_CONFIG['optimal_C'],
            gamma=CPU_OPTIMAL_CONFIG['optimal_gamma'],
            probability=True,
            class_weight='balanced',
            cache_size=2000,
            max_iter=-1,
            decision_function_shape='ovr'
        )
        
        svm.fit(K_train, y_train)
        svm_time = time.time() - t3
        print(f"    ✓ SVM trained: {svm_time:.1f}s")
        
        print(f"    [4/4] Evaluating...")
        y_pred = svm.predict(K_test)
        
        avg_method = 'binary' if is_binary else 'weighted'
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        try:
            y_proba = svm.predict_proba(K_test)
            if is_binary:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            auc = np.nan
        
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        total_time = time.time() - start
        
        result.update({
            'Success': True,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Confusion Matrix': cm,
            'Class Report': class_report,
            'Kernel Train Time (s)': kernel_train_time,
            'Kernel Test Time (s)': kernel_test_time,
            'SVM Time (s)': svm_time,
            'Total Time (s)': total_time,
            'Train Size': len(X_train),
            'Test Size': len(X_test),
            'N_Qubits': n_qubits,
            'Circuit Depth': feature_map.depth(),
            'Circuit Gates': feature_map.size(),
            'Backend': 'CPU',
            'SVM_C': CPU_OPTIMAL_CONFIG['optimal_C'],
        })
        
        print(f"\n    🎯 RESULTS:")
        print(f"       Accuracy:  {acc:.4f}")
        print(f"       Precision: {prec:.4f}")
        print(f"       Recall:    {rec:.4f}")
        print(f"       F1-Score:  {f1:.4f}")
        print(f"       ROC-AUC:   {auc:.4f}")
        print(f"       Time:      {total_time:.1f}s ({total_time/60:.1f}m)")
        
        if acc >= CPU_OPTIMAL_CONFIG['target_accuracy']:
            print(f"\n    ✅ TARGET REACHED! Accuracy {acc:.4f} >= {CPU_OPTIMAL_CONFIG['target_accuracy']}")
        
        return result
        
    except Exception as e:
        print(f"\n    ✗ ERROR: {str(e)}")
        result['Error'] = str(e)
        return result

print("\n" + "="*80)
print("STEP 6: TRAINING QUANTUM MODEL")
print("="*80)

result = train_quantum_cpu_optimized(
    X_train_quantum, X_test_quantum, 
    y_train, y_test, 
    feature_map
)

total_runtime = time.time() - start_time_global

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Total runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
print("="*80)

if not result['Success']:
    print("\n✗ Training failed!")
    exit(1)

print("\n[STEP 7] Analyzing results...")

print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"Accuracy:     {result['Accuracy']:.4f}")
print(f"Precision:    {result['Precision']:.4f}")
print(f"Recall:       {result['Recall']:.4f}")
print(f"F1-Score:     {result['F1-Score']:.4f}")
print(f"ROC-AUC:      {result['ROC-AUC']:.4f}")
print(f"Training:     {result['Total Time (s)']:.1f}s ({result['Total Time (s)']/60:.1f}m)")
print(f"Backend:      {result['Backend']}")
print("="*80)

print(f"\n📊 PER-CLASS PERFORMANCE:")
print(f"{'─'*80}")
class_report = result['Class Report']
for class_name in class_names:
    if class_name in class_report:
        metrics = class_report[class_name]
        print(f"  {class_name:15s}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
print(f"{'─'*80}")

with open(os.path.join(RESULTS_DIR, 'quantum_cpu_optimized_results.json'), 'w') as f:
    details = {
        'accuracy': float(result['Accuracy']),
        'precision': float(result['Precision']),
        'recall': float(result['Recall']),
        'f1_score': float(result['F1-Score']),
        'roc_auc': float(result['ROC-AUC']),
        'training_time_seconds': float(result['Total Time (s)']),
        'n_qubits': int(result['N_Qubits']),
        'circuit_depth': int(result['Circuit Depth']),
        'backend': result['Backend'],
        'class_report': {k: {m: float(v) for m, v in metrics.items() if isinstance(v, (int, float))}
                        for k, metrics in class_report.items() if isinstance(metrics, dict)}
    }
    json.dump(details, f, indent=2)

print(f"\n✓ Results saved to {RESULTS_DIR}/")

print("\n[STEP 8] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :2])
cm = result['Confusion Matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Confusion Matrix - CPU Optimized', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=11)
ax1.set_xlabel('Predicted Label', fontsize=11)

ax2 = fig.add_subplot(gs[0, 2])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [result[m] for m in metrics]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax2.barh(metrics, values, color=colors, alpha=0.7)
ax2.set_xlim([0, 1])
ax2.set_title('Performance Metrics', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, values):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{val:.4f}', va='center', fontweight='bold')

ax3 = fig.add_subplot(gs[1, 0])
class_f1 = [class_report[name]['f1-score'] for name in class_names if name in class_report]
colors_bar = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
bars = ax3.bar(range(len(class_names)), class_f1, color=colors_bar, alpha=0.7)
ax3.set_xticks(range(len(class_names)))
ax3.set_xticklabels(class_names, rotation=45, ha='right')
ax3.set_ylabel('F1-Score', fontsize=11)
ax3.set_title('Per-Class F1-Score', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(class_f1):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

ax4 = fig.add_subplot(gs[1, 1])
time_components = {
    'Kernel (Train)': result['Kernel Train Time (s)'],
    'Kernel (Test)': result['Kernel Test Time (s)'],
    'SVM Training': result['SVM Time (s)']
}
wedges, texts, autotexts = ax4.pie(time_components.values(), 
                                     labels=time_components.keys(),
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=['#3498db', '#2ecc71', '#e74c3c'])
ax4.set_title('Time Breakdown', fontsize=12, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 2])
class_metrics = pd.DataFrame({
    'Precision': [class_report[name]['precision'] for name in class_names if name in class_report],
    'Recall': [class_report[name]['recall'] for name in class_names if name in class_report],
    'F1-Score': [class_report[name]['f1-score'] for name in class_names if name in class_report]
}, index=class_names)

x = np.arange(len(class_names))
width = 0.25
ax5.bar(x - width, class_metrics['Precision'], width, label='Precision', alpha=0.8, color='#3498db')
ax5.bar(x, class_metrics['Recall'], width, label='Recall', alpha=0.8, color='#2ecc71')
ax5.bar(x + width, class_metrics['F1-Score'], width, label='F1-Score', alpha=0.8, color='#e74c3c')
ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(class_names, rotation=45, ha='right')
ax5.legend()
ax5.set_ylim([0, 1])
ax5.grid(True, alpha=0.3, axis='y')

ax6 = fig.add_subplot(gs[2, :])
summary_text = f"""
{'='*100}
QUANTUM IDS - CPU-OPTIMIZED TRAINING SUMMARY
{'='*100}

CONFIGURATION:
  Qubits:                 {CPU_OPTIMAL_CONFIG['n_qubits']}
  Training Samples:       {len(X_train_quantum):,}
  Testing Samples:        {len(X_test_quantum):,}
  Number of Classes:      {n_classes}
  Backend:                CPU (All cores)
  Pre-optimized C:        {CPU_OPTIMAL_CONFIG['optimal_C']}

PERFORMANCE:
  Accuracy:               {result['Accuracy']:.4f} {'✅ TARGET REACHED!' if result['Accuracy'] >= CPU_OPTIMAL_CONFIG['target_accuracy'] else ''}
  Precision:              {result['Precision']:.4f}
  Recall:                 {result['Recall']:.4f}
  F1-Score:               {result['F1-Score']:.4f}
  ROC-AUC:                {result['ROC-AUC']:.4f}

TRAINING METRICS:
  Total Runtime:          {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)
  Training Time:          {result['Total Time (s)']:.1f} seconds ({result['Total Time (s)']/60:.1f} minutes)
  Kernel Computation:     {result['Kernel Train Time (s)'] + result['Kernel Test Time (s)']:.1f}s
  SVM Training:           {result['SVM Time (s)']:.1f}s
  Circuit Depth:          {result['Circuit Depth']}
  Circuit Gates:          {result['Circuit Gates']}

PER-CLASS PERFORMANCE:
"""

for class_name in class_names:
    if class_name in class_report:
        metrics = class_report[class_name]
        summary_text += f"  {class_name:20s}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n"

summary_text += f"""
{'='*100}
STATUS: {'✅ SUCCESS - Target accuracy achieved!' if result['Accuracy'] >= CPU_OPTIMAL_CONFIG['target_accuracy'] else '⚠️  Target not reached, but best possible result obtained'}
{'='*100}
"""

ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax6.axis('off')

plt.suptitle('Quantum Intrusion Detection System - CPU Optimized Results', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(os.path.join(FIGURES_DIR, 'quantum_cpu_optimized_results.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved")

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
class_metrics.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
ax.set_title('Detailed Per-Class Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=11)
ax.set_xlabel('Class', fontsize=11)
ax.legend(loc='lower right')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax = axes[0, 1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
ax.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)

ax = axes[1, 0]
support = [class_report[name]['support'] for name in class_names if name in class_report]
colors_support = plt.cm.plasma(np.linspace(0, 1, len(class_names)))
bars = ax.barh(class_names, support, color=colors_support, alpha=0.7)
ax.set_xlabel('Number of Samples', fontsize=11)
ax.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, support):
    ax.text(val + max(support)*0.01, bar.get_y() + bar.get_height()/2, 
            f'{int(val)}', va='center', fontweight='bold')

ax = axes[1, 1]
summary_data = {
    'Metric': ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 
               'Macro Avg F1', 'Weighted Avg F1', 'ROC-AUC'],
    'Score': [
        result['Accuracy'],
        class_report['macro avg']['precision'],
        class_report['macro avg']['recall'],
        class_report['macro avg']['f1-score'],
        class_report['weighted avg']['f1-score'],
        result['ROC-AUC']
    ]
}

table_data = [[m, f"{s:.4f}"] for m, s in zip(summary_data['Metric'], summary_data['Score'])]
table = ax.table(cellText=table_data, colLabels=['Metric', 'Score'],
                cellLoc='left', loc='center',
                colWidths=[0.6, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(1, len(table_data) + 1):
    score = summary_data['Score'][i-1]
    if score >= 0.95:
        color = '#d4edda'
    elif score >= 0.90:
        color = '#fff3cd'
    else:
        color = '#f8d7da'
    table[(i, 1)].set_facecolor(color)

ax.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'quantum_cpu_optimized_detailed.png'),
            dpi=300, bbox_inches='tight')
print("✓ Detailed visualization saved")

# Close figures to free memory
plt.close('all')

print("\n" + "="*80)
print("🎉 QUANTUM IDS CPU OPTIMIZATION COMPLETE!")
print("="*80)

if result['Accuracy'] >= CPU_OPTIMAL_CONFIG['target_accuracy']:
    print(f"\n✅ SUCCESS! Target accuracy {CPU_OPTIMAL_CONFIG['target_accuracy']} achieved!")
else:
    gap = CPU_OPTIMAL_CONFIG['target_accuracy'] - result['Accuracy']
    print(f"\n⚠️ Target not reached. Gap: {gap:.4f}")
    print("Suggestions:")
    print(" • Increase train size")
    print(" • Increase reps in feature map")
    print(" • Try PauliFeatureMap")
    print(" • Tune SVM C parameter")

print("\n📁 OUTPUT FILES:")
print(f"  JSON Results:  {os.path.join(RESULTS_DIR, 'quantum_cpu_optimized_results.json')}")
print(f"  Main Figure:   {os.path.join(FIGURES_DIR, 'quantum_cpu_optimized_results.png')}")
print(f"  Detailed Fig:  {os.path.join(FIGURES_DIR, 'quantum_cpu_optimized_detailed.png')}")

print("\n🚀 Done!")
