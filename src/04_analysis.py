"""
Quantum IDS Project - Enhanced Comprehensive Comparative Analysis
File: src/04_comparative_analysis_enhanced.py

Purpose: Deep analysis and visualization of quantum vs classical performance
Features:
- Statistical significance testing
- Multiple visualization styles
- Detailed per-class analysis
- Publication-ready figures
- Automated report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUANTUM vs CLASSICAL - ENHANCED COMPREHENSIVE ANALYSIS")
print("="*80)

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    'results_dir': 'results',
    'figures_dir': 'results/figures',
    'processed_dir': 'data/processed',
    'dpi': 300,
    'fig_format': 'png',
    'include_detailed_stats': True,
    'confidence_level': 0.95,
}

RESULTS_DIR = CONFIG['results_dir']
FIGURES_DIR = CONFIG['figures_dir']
PROCESSED_DIR = CONFIG['processed_dir']

os.makedirs(FIGURES_DIR, exist_ok=True)

# ========================================
# STEP 1: Load All Available Results
# ========================================
print("\n[STEP 1] Loading all available results...")

def safe_load_csv(directory, filenames):
    """Safely load CSV files, trying multiple filenames."""
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"  ✓ Loaded: {filename}")
                return df, filename
            except Exception as e:
                print(f"  ⚠ Error loading {filename}: {e}")
    return None, None

def safe_load_json(directory, filenames):
    """Safely load JSON files, trying multiple filenames."""
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"  ✓ Loaded: {filename}")
                return data, filename
            except Exception as e:
                print(f"  ⚠ Error loading {filename}: {e}")
    return None, None

# Load classical results
classical_files = [
    'classical_baseline_final_results.csv',
    'classical_baseline_improved_results.csv',
    'classical_baseline_results.csv'
]
classical_df, classical_file = safe_load_csv(RESULTS_DIR, classical_files)

if classical_df is None:
    print("\n✗ ERROR: No classical results found!")
    print("  Please run 02_classical_baseline_final.py first")
    exit(1)

# Standardize column names
def standardize_columns(df):
    """Standardize column names across different result files."""
    column_mapping = {
        'Training Time (s)': 'Total Time (s)',
        'Training Time': 'Total Time (s)',
        'Time (s)': 'Total Time (s)',
        'Total Training Time (s)': 'Total Time (s)',
    }
    df = df.rename(columns=column_mapping)
    return df

classical_df = standardize_columns(classical_df)
print(f"  Classical columns: {list(classical_df.columns)}")

quantum_files = [
    'quantum_kernel_final_results.csv',
    'quantum_kernel_improved_results.csv',
    'quantum_kernel_optimized_results.csv'
]
quantum_df, quantum_file = safe_load_csv(RESULTS_DIR, quantum_files)

# Also try loading from JSON if CSV not available
if quantum_df is None:
    print("  No quantum CSV found, checking JSON...")
    quantum_json_files = [
        'ultra_novel_results.json',  # ✅ ADD THIS for ultra-novel script
        'quantum_cpu_optimized_results.json',
        'quantum_kernel_results.json'
    ]
    quantum_json, json_file = safe_load_json(RESULTS_DIR, quantum_json_files)
    
    if quantum_json:
        # ✅ NEW: Handle ultra-novel JSON format
        if 'individual_models' in quantum_json:
            # Ultra-novel format with multiple models
            print(f"  ✓ Found ultra-novel results format")
            
            models_data = []
            for model_name, model_data in quantum_json['individual_models'].items():
                models_data.append({
                    'Model': model_name,
                    'Accuracy': model_data.get('accuracy', 0),
                    'Precision': quantum_json['final_metrics'].get('precision', 0),
                    'Recall': quantum_json['final_metrics'].get('recall', 0),
                    'F1-Score': model_data.get('f1_score', 0),
                    'ROC-AUC': quantum_json['final_metrics'].get('roc_auc', 0),
                    'Total Time (s)': quantum_json.get('training_time_hours', 0) * 3600,
                })
            
            # Add ensemble results
            models_data.append({
                'Model': 'Ultra-Novel Ensemble',
                'Accuracy': quantum_json['final_metrics']['accuracy'],
                'Precision': quantum_json['final_metrics']['precision'],
                'Recall': quantum_json['final_metrics']['recall'],
                'F1-Score': quantum_json['final_metrics']['f1_score'],
                'ROC-AUC': quantum_json['final_metrics']['roc_auc'],
                'Total Time (s)': quantum_json['training_time_hours'] * 3600,
            })
            
            quantum_df = pd.DataFrame(models_data)
            print(f"  ✓ Converted {len(models_data)} models from ultra-novel JSON")
            
        else:
            # ✅ EXISTING: Old format (single model)
            quantum_df = pd.DataFrame([{
                'Model': 'Quantum Kernel',
                'Accuracy': quantum_json.get('accuracy', 0),
                'Precision': quantum_json.get('precision', 0),
                'Recall': quantum_json.get('recall', 0),
                'F1-Score': quantum_json.get('f1_score', 0),
                'ROC-AUC': quantum_json.get('roc_auc', 0),
                'Total Time (s)': quantum_json.get('training_time_seconds', 0),
            }])
            print(f"  ✓ Converted single model from JSON: {json_file}")

if quantum_df is None:
    print("\n⚠ WARNING: No quantum results found!")
    print("  Analysis will only show classical results")
    quantum_available = False
else:
    quantum_df = standardize_columns(quantum_df)
    print(f"  Quantum columns: {list(quantum_df.columns)}")
    quantum_available = True


# Load configuration and metadata
try:
    with open(os.path.join(PROCESSED_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(PROCESSED_DIR, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f]
    
    print(f"\n✓ Configuration loaded")
    print(f"  Classification mode: {config['mode']}")
    print(f"  Number of classes:   {len(class_names)}")
    print(f"  Classes:             {', '.join(class_names)}")
    
except Exception as e:
    print(f"\n⚠ Warning: Could not load config: {e}")
    config = {'mode': 'unknown', 'sample_size': 0}
    class_names = ['Unknown']

# Summary
print(f"\n{'='*80}")
print(f"DATA SUMMARY:")
print(f"  Classical models:    {len(classical_df)}")
if quantum_available:
    # ✅ FIX: Handle both 'Feature Map' and 'Model' columns
    model_col = 'Model' if 'Model' in quantum_df.columns else 'Feature Map'
    
    best_quantum = quantum_df.loc[quantum_df['F1-Score'].idxmax()]
    print(f"\n🔴 Best Quantum Model: {best_quantum[model_col]}")
    print(f"   F1-Score: {best_quantum['F1-Score']:.4f}")
    print(f"   Accuracy: {best_quantum['Accuracy']:.4f}")


else:
    print(f"  Quantum models:      0 (not available)")
print(f"  Classification:      {config['mode']} ({len(class_names)} classes)")
print(f"{'='*80}")

# ========================================
# STEP 2: Enhanced Statistical Analysis
# ========================================
print("\n[STEP 2] Performing enhanced statistical analysis...")

# Metrics to analyze
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

# Best models
best_classical = classical_df.loc[classical_df['F1-Score'].idxmax()]
print(f"\n🔵 Best Classical Model: {best_classical['Model']}")
print(f"   F1-Score: {best_classical['F1-Score']:.4f}")
print(f"   Accuracy: {best_classical['Accuracy']:.4f}")

if quantum_available:
    model_col = 'Feature Map' if 'Feature Map' in quantum_df.columns else 'Model'
    best_quantum = quantum_df.loc[quantum_df['F1-Score'].idxmax()]
    print(f"\n🔴 Best Quantum Model: {best_quantum[model_col]}")
    print(f"   F1-Score: {best_quantum['F1-Score']:.4f}")
    print(f"   Accuracy: {best_quantum['Accuracy']:.4f}")

# Detailed comparison
print("\n" + "="*80)
print("DETAILED PERFORMANCE COMPARISON")
print("="*80)

results_summary = {}

for metric in metrics:
    classical_vals = classical_df[metric].dropna()
    
    stats_dict = {
        'classical_mean': classical_vals.mean(),
        'classical_std': classical_vals.std(),
        'classical_min': classical_vals.min(),
        'classical_max': classical_vals.max(),
        'classical_median': classical_vals.median(),
    }
    
    if quantum_available:
        quantum_vals = quantum_df[metric].dropna()
        stats_dict.update({
            'quantum_mean': quantum_vals.mean(),
            'quantum_std': quantum_vals.std(),
            'quantum_min': quantum_vals.min(),
            'quantum_max': quantum_vals.max(),
            'quantum_median': quantum_vals.median(),
        })
        
        # Statistical test
        if len(classical_vals) > 1 and len(quantum_vals) > 1:
            t_stat, p_value = stats.ttest_ind(quantum_vals, classical_vals)
            stats_dict.update({
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    results_summary[metric] = stats_dict
    
    # Print summary
    print(f"\n{metric}:")
    print(f"  Classical: μ={stats_dict['classical_mean']:.4f} ± {stats_dict['classical_std']:.4f}, "
          f"range=[{stats_dict['classical_min']:.4f}, {stats_dict['classical_max']:.4f}]")
    
    if quantum_available:
        print(f"  Quantum:   μ={stats_dict['quantum_mean']:.4f} ± {stats_dict['quantum_std']:.4f}, "
              f"range=[{stats_dict['quantum_min']:.4f}, {stats_dict['quantum_max']:.4f}]")
        
        if 'p_value' in stats_dict:
            significance = "***" if stats_dict['p_value'] < 0.001 else \
                          "**" if stats_dict['p_value'] < 0.01 else \
                          "*" if stats_dict['p_value'] < 0.05 else "ns"
            print(f"  t-test:    t={stats_dict['t_statistic']:.4f}, p={stats_dict['p_value']:.4f} {significance}")

# ========================================
# STEP 3: Comprehensive Visualizations
# ========================================
print("\n[STEP 3] Creating comprehensive visualizations...")

# Main comparison figure
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# Color scheme
classical_color = '#3498db'
quantum_color = '#e74c3c'

# 1. Head-to-Head Best Models
ax1 = fig.add_subplot(gs[0, 0])
if quantum_available:
    x = np.arange(len(metrics))
    width = 0.35
    classical_scores = [best_classical[m] for m in metrics]
    quantum_scores = [best_quantum[m] for m in metrics]
    
    bars1 = ax1.bar(x - width/2, classical_scores, width, label='Classical', 
                   color=classical_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, quantum_scores, width, label='Quantum', 
                   color=quantum_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right', fontsize=10)
else:
    classical_scores = [best_classical[m] for m in metrics]
    bars = ax1.bar(range(len(metrics)), classical_scores, color=classical_color, alpha=0.8)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)

ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('Best Models Comparison', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1.1])

# 2. Distribution Comparison (Box Plot)
ax2 = fig.add_subplot(gs[0, 1])
if quantum_available:
    box_data = []
    labels = []
    colors_box = []
    
    for metric in ['Accuracy', 'F1-Score']:
        box_data.append(classical_df[metric].dropna())
        labels.append(f'{metric}\nClassical')
        colors_box.append(classical_color)
        
        box_data.append(quantum_df[metric].dropna())
        labels.append(f'{metric}\nQuantum')
        colors_box.append(quantum_color)
    
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
else:
    box_data = [classical_df['Accuracy'].dropna(), classical_df['F1-Score'].dropna()]
    bp = ax2.boxplot(box_data, labels=['Accuracy', 'F1-Score'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(classical_color)

ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Score Distributions', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.get_xticklabels(), fontsize=8)

# 3. Radar Chart
ax3 = fig.add_subplot(gs[0, 2], projection='polar')
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

classical_radar = [best_classical[m] for m in metrics] + [best_classical[metrics[0]]]
ax3.plot(angles, classical_radar, 'o-', linewidth=2.5, color=classical_color, 
        label='Classical', markersize=8)
ax3.fill(angles, classical_radar, alpha=0.25, color=classical_color)

if quantum_available:
    quantum_radar = [best_quantum[m] for m in metrics] + [best_quantum[metrics[0]]]
    ax3.plot(angles, quantum_radar, 's-', linewidth=2.5, color=quantum_color, 
            label='Quantum', markersize=8)
    ax3.fill(angles, quantum_radar, alpha=0.25, color=quantum_color)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_ylim(0, 1)
ax3.set_title('Performance Radar', fontsize=12, fontweight='bold', pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax3.grid(True, linewidth=1.5, alpha=0.5)

# 4. Improvement Analysis
ax4 = fig.add_subplot(gs[0, 3])
if quantum_available:
    improvements = []
    for metric in metrics:
        classical_val = best_classical[metric]
        quantum_val = best_quantum[metric]
        improvement = ((quantum_val - classical_val) / classical_val * 100) if classical_val > 0 else 0
        improvements.append(improvement)
    
    colors_imp = [quantum_color if x > 0 else classical_color for x in improvements]
    bars = ax4.barh(range(len(metrics)), improvements, color=colors_imp, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    for i, v in enumerate(improvements):
        ax4.text(v + (1 if v > 0 else -1), i, f'{v:+.1f}%', 
                va='center', ha='left' if v > 0 else 'right', fontsize=9, fontweight='bold')
    
    ax4.set_yticks(range(len(metrics)))
    ax4.set_yticklabels(metrics, fontsize=9)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax4.set_title('Quantum Improvement (%)', fontsize=12, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Quantum results\nnot available', 
            ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')

ax4.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. All Models Ranking
ax5 = fig.add_subplot(gs[1, :2])
if quantum_available:
    model_col = 'Feature Map' if 'Feature Map' in quantum_df.columns else 'Model'
    all_results = pd.concat([
        classical_df[['Model', 'F1-Score']].rename(columns={'Model': 'Name'}).assign(Type='Classical'),
        quantum_df[[model_col, 'F1-Score']].rename(columns={model_col: 'Name'}).assign(Type='Quantum')
    ]).sort_values('F1-Score', ascending=True).tail(12)
    
    colors_rank = [classical_color if t == 'Classical' else quantum_color for t in all_results['Type']]
    bars = ax5.barh(range(len(all_results)), all_results['F1-Score'], 
                   color=colors_rank, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax5.set_yticks(range(len(all_results)))
    ax5.set_yticklabels(all_results['Name'], fontsize=8)
    
    for i, (idx, row) in enumerate(all_results.iterrows()):
        ax5.text(row['F1-Score'] + 0.005, i, f"{row['F1-Score']:.3f}", 
                va='center', fontsize=8, fontweight='bold')
else:
    top_models = classical_df.nlargest(12, 'F1-Score').sort_values('F1-Score')
    bars = ax5.barh(range(len(top_models)), top_models['F1-Score'], 
                   color=classical_color, alpha=0.7)
    ax5.set_yticks(range(len(top_models)))
    ax5.set_yticklabels(top_models['Model'], fontsize=8)

ax5.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax5.set_title('Top Models Ranking', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.set_xlim([0, 1.05])

# 6. Training Time Analysis
ax6 = fig.add_subplot(gs[1, 2:])

# Check if time column exists
has_time_classical = 'Total Time (s)' in classical_df.columns
has_time_quantum = quantum_available and 'Total Time (s)' in quantum_df.columns

if has_time_classical and has_time_quantum:
    # Scatter plot
    ax6.scatter(classical_df['Total Time (s)'], classical_df['F1-Score'], 
               s=200, alpha=0.6, color=classical_color, label='Classical', 
               marker='o', edgecolors='black', linewidth=1.5)
    ax6.scatter(quantum_df['Total Time (s)'], quantum_df['F1-Score'], 
               s=200, alpha=0.6, color=quantum_color, label='Quantum', 
               marker='s', edgecolors='black', linewidth=1.5)
    
    # Add best model annotations
    ax6.annotate(f"{best_classical['Model'][:15]}", 
                (best_classical['Total Time (s)'], best_classical['F1-Score']),
                xytext=(10, 10), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=classical_color, alpha=0.3))
    
    if quantum_available:
        model_name = best_quantum[model_col][:15]
        ax6.annotate(f"{model_name}", 
                    (best_quantum['Total Time (s)'], best_quantum['F1-Score']),
                    xytext=(10, -10), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=quantum_color, alpha=0.3))
    
    ax6.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax6.set_title('Performance vs Training Time', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
elif has_time_classical:
    # Only classical has time data
    ax6.scatter(classical_df['Total Time (s)'], classical_df['F1-Score'], 
               s=200, alpha=0.6, color=classical_color, marker='o', edgecolors='black', linewidth=1.5)
    ax6.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax6.set_title('Performance vs Training Time (Classical Only)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
else:
    # No time data available
    ax6.text(0.5, 0.5, 'Training time\ndata not available', 
            ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    ax6.set_title('Performance vs Training Time', fontsize=12, fontweight='bold')
    ax6.axis('off')

# 7. Statistical Significance Heatmap
ax7 = fig.add_subplot(gs[2, :2])
if quantum_available and CONFIG['include_detailed_stats']:
    sig_data = []
    for metric in metrics:
        row = []
        classical_vals = classical_df[metric].dropna()
        quantum_vals = quantum_df[metric].dropna()
        
        if len(classical_vals) > 1 and len(quantum_vals) > 1:
            _, p_val = stats.ttest_ind(quantum_vals, classical_vals)
            row.append(p_val)
        else:
            row.append(1.0)
        
        sig_data.append(row)
    
    sig_matrix = np.array(sig_data)
    
    sns.heatmap(sig_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r', 
               ax=ax7, cbar_kws={'label': 'p-value'},
               xticklabels=['Quantum vs Classical'],
               yticklabels=metrics,
               vmin=0, vmax=0.1, center=0.05,
               linewidths=2, linecolor='black')
    
    ax7.set_title('Statistical Significance (t-test p-values)', 
                 fontsize=12, fontweight='bold')
    
    # Add significance markers
    for i in range(len(metrics)):
        p_val = sig_matrix[i, 0]
        if p_val < 0.001:
            marker = "***"
        elif p_val < 0.01:
            marker = "**"
        elif p_val < 0.05:
            marker = "*"
        else:
            marker = "ns"
        ax7.text(0.5, i + 0.7, marker, ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
else:
    ax7.text(0.5, 0.5, 'Statistical analysis\nrequires both\nclassical and quantum\nresults', 
            ha='center', va='center', fontsize=12, transform=ax7.transAxes)
    ax7.axis('off')

# 8. Performance Consistency
ax8 = fig.add_subplot(gs[2, 2:])
std_metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
x = np.arange(len(std_metrics))
width = 0.35

classical_stds = [classical_df[m].std() for m in std_metrics]

if quantum_available:
    quantum_stds = [quantum_df[m].std() for m in std_metrics]
    
    bars1 = ax8.bar(x - width/2, classical_stds, width, label='Classical', 
                   color=classical_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax8.bar(x + width/2, quantum_stds, width, label='Quantum', 
                   color=quantum_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
else:
    bars = ax8.bar(x, classical_stds, color=classical_color, alpha=0.8)

ax8.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
ax8.set_title('Performance Consistency (Lower is Better)', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(std_metrics, fontsize=9)
if quantum_available:
    ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, axis='y')

# 9. Summary Statistics Table
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

summary_text = f"""
{'='*100}
COMPREHENSIVE ANALYSIS SUMMARY
{'='*100}

DATASET INFORMATION:
  • Classification Mode:     {config['mode']}
  • Number of Classes:       {len(class_names)}
  • Classes:                 {', '.join(class_names)}
  • Sample Size:             {config.get('sample_size', 'N/A')}

BEST CLASSICAL MODEL:
  • Model:                   {best_classical['Model']}
  • Accuracy:                {best_classical['Accuracy']:.4f}
  • Precision:               {best_classical['Precision']:.4f}
  • Recall:                  {best_classical['Recall']:.4f}
  • F1-Score:                {best_classical['F1-Score']:.4f}
  • ROC-AUC:                 {best_classical['ROC-AUC']:.4f}
  • Training Time:           {best_classical['Total Time (s)']:.2f}s
"""

if quantum_available:
    model_name = best_quantum[model_col]
    # Get time safely
    best_quantum_time = best_quantum.get('Total Time (s)', 'N/A')
    best_classical_time = best_classical.get('Total Time (s)', 'N/A')
    
    summary_text += f"""
BEST QUANTUM MODEL:
  • Model:                   {model_name}
  • Accuracy:                {best_quantum['Accuracy']:.4f}
  • Precision:               {best_quantum['Precision']:.4f}
  • Recall:                  {best_quantum['Recall']:.4f}
  • F1-Score:                {best_quantum['F1-Score']:.4f}
  • ROC-AUC:                 {best_quantum['ROC-AUC']:.4f}
  • Training Time:           {best_quantum_time if isinstance(best_quantum_time, str) else f'{best_quantum_time:.2f}s'}

COMPARATIVE ANALYSIS:
"""
    
    for metric in metrics:
        classical_val = best_classical[metric]
        quantum_val = best_quantum[metric]
        improvement = ((quantum_val - classical_val) / classical_val * 100) if classical_val > 0 else 0
        winner = "🟢 Quantum" if improvement > 0 else "🔵 Classical" if improvement < 0 else "⚪ Tie"
        summary_text += f"  • {metric:12s}:         {winner} ({improvement:+.2f}%)\n"
    
    avg_classical = classical_df['F1-Score'].mean()
    avg_quantum = quantum_df['F1-Score'].mean()
    summary_text += f"""
OVERALL STATISTICS:
  • Avg Classical F1:        {avg_classical:.4f}
  • Avg Quantum F1:          {avg_quantum:.4f}
  • Overall Winner:          {'🟢 Quantum' if avg_quantum > avg_classical else '🔵 Classical'}
"""

summary_text += f"""
{'='*100}
"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=2))

plt.suptitle('Quantum vs Classical IDS - Comprehensive Analysis', 
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig(os.path.join(FIGURES_DIR, f'quantum_vs_classical_comprehensive.{CONFIG["fig_format"]}'), 
           dpi=CONFIG['dpi'], bbox_inches='tight')
print(f"✓ Comprehensive visualization saved")

# ========================================
# STEP 4: Generate Detailed Report
# ========================================
print("\n[STEP 4] Generating detailed report...")

report_path = os.path.join(RESULTS_DIR, 'comparative_analysis_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("QUANTUM vs CLASSICAL IDS - DETAILED ANALYSIS REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("1. DATASET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Classification Mode:  {config['mode']}\n")
    f.write(f"Number of Classes:    {len(class_names)}\n")
    f.write(f"Classes:              {', '.join(class_names)}\n")
    f.write(f"Sample Size:          {config.get('sample_size', 'N/A')}\n\n")
    
    f.write("2. CLASSICAL RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Number of Models:     {len(classical_df)}\n")
    f.write(f"Best Model:           {best_classical['Model']}\n\n")
    
    f.write("Best Model Performance:\n")
    for metric in metrics:
        f.write(f"  {metric:12s}: {best_classical[metric]:.4f}\n")
    
    f.write("\nAverage Performance:\n")
    for metric in metrics:
        f.write(f"  {metric:12s}: {results_summary[metric]['classical_mean']:.4f} ± {results_summary[metric]['classical_std']:.4f}\n")
    
    if quantum_available:
        f.write("\n3. QUANTUM RESULTS\n")
        f.write("-" * 80 + "\n")
        
        # Overall assessment
        quantum_wins = sum(1 for metric in metrics 
                          if best_quantum[metric] > best_classical[metric])
        classical_wins = len(metrics) - quantum_wins
        
        f.write(f"\nHead-to-Head (Best Models):\n")
        f.write(f"  Quantum Wins:    {quantum_wins}/{len(metrics)} metrics\n")
        f.write(f"  Classical Wins:  {classical_wins}/{len(metrics)} metrics\n")
        
        if quantum_wins > classical_wins:
            f.write(f"\n✓ The best quantum model outperforms the best classical model.\n")
        elif classical_wins > quantum_wins:
            f.write(f"\n✓ The best classical model outperforms the best quantum model.\n")
        else:
            f.write(f"\n✓ The best quantum and classical models show similar performance.\n")
        
        if avg_quantum > avg_classical:
            f.write(f"✓ On average, quantum models show superior performance.\n")
        else:
            f.write(f"✓ On average, classical models show superior performance.\n")
        
        # Performance vs Time tradeoff
        best_quantum_time = best_quantum['Total Time (s)']
        best_classical_time = best_classical['Total Time (s)']
        
        f.write(f"\nPerformance vs Time Tradeoff:\n")
        
        # Check if time data exists
        best_quantum_time = best_quantum.get('Total Time (s)', None)
        best_classical_time = best_classical.get('Total Time (s)', None)
        
        if best_classical_time is not None and best_quantum_time is not None:
            f.write(f"  Best Classical: F1={best_classical['F1-Score']:.4f} in {best_classical_time:.2f}s\n")
            f.write(f"  Best Quantum:   F1={best_quantum['F1-Score']:.4f} in {best_quantum_time:.2f}s\n")
            
            if best_quantum['F1-Score'] > best_classical['F1-Score']:
                perf_gain = ((best_quantum['F1-Score'] - best_classical['F1-Score']) / best_classical['F1-Score'] * 100)
                time_cost = ((best_quantum_time - best_classical_time) / best_classical_time * 100) if best_classical_time > 0 else 0
                f.write(f"  Performance Gain: {perf_gain:+.2f}%\n")
                f.write(f"  Time Cost:        {time_cost:+.2f}%\n")
        else:
            f.write(f"  Best Classical: F1={best_classical['F1-Score']:.4f}\n")
            f.write(f"  Best Quantum:   F1={best_quantum['F1-Score']:.4f}\n")
            f.write(f"  (Training time data not available)\n")
        
        f.write("\nRecommendation:\n")
        if best_quantum['F1-Score'] > best_classical['F1-Score'] + 0.01:  # 1% better
            f.write("  → Quantum kernel methods show meaningful performance improvements.\n")
            f.write("    Consider using quantum approaches for this intrusion detection task.\n")
        elif best_classical['F1-Score'] > best_quantum['F1-Score'] + 0.01:
            f.write("  → Classical methods show superior performance for this task.\n")
            f.write("    Classical methods are recommended for practical deployment.\n")
        else:
            f.write("  → Quantum and classical methods show comparable performance.\n")
            f.write("    Choice depends on computational resources and deployment constraints.\n")
    
    else:
        # Only classical results available
        f.write("\n3. QUANTUM RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("No quantum results available for comparison.\n")
        f.write("Please run quantum kernel training script to generate quantum results.\n")
        
        f.write("\n4. CONCLUSION\n")
        f.write("-" * 80 + "\n")
        f.write("Analysis limited to classical models only.\n")
        f.write(f"Best classical model achieved F1-Score of {best_classical['F1-Score']:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ Detailed report saved to {report_path}")

# ========================================
# STEP 5: Additional Detailed Visualizations
# ========================================
print("\n[STEP 5] Creating additional detailed visualizations...")

# Create individual detailed plots
if quantum_available:
    # 1. Detailed Comparison Matrix
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    comparison_data = []
    for metric in metrics:
        comparison_data.append([
            best_classical[metric],
            best_quantum[metric],
            results_summary[metric]['classical_mean'],
            results_summary[metric]['quantum_mean']
        ])
    
    comparison_df = pd.DataFrame(
        comparison_data,
        columns=['Classical Best', 'Quantum Best', 'Classical Avg', 'Quantum Avg'],
        index=metrics
    )
    
    sns.heatmap(comparison_df, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax,
               cbar_kws={'label': 'Score'}, linewidths=1, linecolor='black',
               vmin=0, vmax=1)
    ax.set_title('Detailed Performance Comparison Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Comparison Type', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'comparison_matrix.{CONFIG["fig_format"]}'), 
               dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"✓ Comparison matrix saved")
    
    # 2. Performance Improvement Detailed
    fig3, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top subplot: Absolute values
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    
    classical_scores = [best_classical[m] for m in metrics]
    quantum_scores = [best_quantum[m] for m in metrics]
    
    bars1 = ax.bar(x - width/2, classical_scores, width, label='Classical', 
                  color=classical_color, alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, quantum_scores, width, label='Quantum', 
                  color=quantum_color, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bottom subplot: Relative improvement
    ax = axes[1]
    improvements = []
    for metric in metrics:
        improvement = ((best_quantum[metric] - best_classical[metric]) / best_classical[metric] * 100) if best_classical[metric] > 0 else 0
        improvements.append(improvement)
    
    colors_imp = [quantum_color if x > 0 else classical_color for x in improvements]
    bars = ax.bar(range(len(metrics)), improvements, color=colors_imp, alpha=0.8, 
                 edgecolor='black', linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quantum Relative Improvement over Classical', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(bar.get_x() + bar.get_width()/2., val + (0.5 if val > 0 else -0.5),
               f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'performance_improvement_detailed.{CONFIG["fig_format"]}'), 
               dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"✓ Performance improvement plot saved")
    
    plt.close('all')

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "="*80)
print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"   Best Classical:  {best_classical['Model']}")
print(f"   F1-Score:        {best_classical['F1-Score']:.4f}")

if quantum_available:
    print(f"\n   Best Quantum:    {best_quantum[model_col]}")
    print(f"   F1-Score:        {best_quantum['F1-Score']:.4f}")
    
    improvement = ((best_quantum['F1-Score'] - best_classical['F1-Score']) / best_classical['F1-Score'] * 100)
    if improvement > 0:
        print(f"\n   🟢 Quantum advantage: {improvement:+.2f}%")
    elif improvement < 0:
        print(f"\n   🔵 Classical advantage: {abs(improvement):.2f}%")
    else:
        print(f"\n   ⚪ Tie: Equal performance")

print(f"\n📁 OUTPUT FILES:")
print(f"   Main visualization:  {FIGURES_DIR}/quantum_vs_classical_comprehensive.{CONFIG['fig_format']}")
print(f"   Detailed report:     {report_path}")

if quantum_available:
    print(f"   Comparison matrix:   {FIGURES_DIR}/comparison_matrix.{CONFIG['fig_format']}")
    print(f"   Improvement plot:    {FIGURES_DIR}/performance_improvement_detailed.{CONFIG['fig_format']}")

print("\n" + "="*80)
print("Analysis complete! Review the report and visualizations for insights.")
print("="*80)