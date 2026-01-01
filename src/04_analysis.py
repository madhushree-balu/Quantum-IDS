"""
Quantum IDS Project - Comparative Analysis and Paper Figures (FIXED)
File: src/04_comparative_analysis_fixed.py
Purpose: Generate publication-ready figures with correct column names
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Set publication-quality plot settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'

print("="*80)
print("QUANTUM IDS PROJECT - COMPARATIVE ANALYSIS & PAPER FIGURES")
print("="*80)

# ========================================
# STEP 1: Load All Results
# ========================================
print("\n[STEP 1] Loading all results...")

RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
PAPER_DIR = 'paper/figures'

# Create paper figures directory
os.makedirs(PAPER_DIR, exist_ok=True)

# Load results
classical_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classical_baseline_results.csv'))
quantum_df = pd.read_csv(os.path.join(RESULTS_DIR, 'quantum_kernel_results.csv'))

print(f"✓ Classical results: {len(classical_df)} models")
print(f"✓ Quantum results: {len(quantum_df)} experiments")

# Debug: Print column names
print(f"\nQuantum DataFrame columns: {quantum_df.columns.tolist()}")

# Standardize column names (handle both old and new naming)
column_mapping = {
    'Gates': 'Number of Gates',
    'Qubits': 'Number of Qubits'
}
quantum_df.rename(columns=column_mapping, inplace=True)

# Get best models
best_classical = classical_df.loc[classical_df['F1-Score'].idxmax()]
best_quantum = quantum_df.loc[quantum_df['F1-Score'].idxmax()]

print(f"\nBest Classical: {best_classical['Model']}")
print(f"  F1-Score: {best_classical['F1-Score']:.4f}")

print(f"\nBest Quantum: {best_quantum['Feature Map']}")
print(f"  F1-Score: {best_quantum['F1-Score']:.4f}")
print(f"  Dataset Size: {best_quantum['Dataset Size']}")

# ========================================
# STEP 2: Statistical Analysis
# ========================================
print("\n[STEP 2] Performing statistical analysis...")

# Get quantum results for different dataset sizes
dataset_sizes = sorted(quantum_df['Dataset Size'].unique())

statistical_results = []
for size in dataset_sizes:
    size_data = quantum_df[quantum_df['Dataset Size'] == size]
    
    # T-test: Quantum vs Classical
    quantum_f1 = size_data['F1-Score'].values
    classical_f1 = best_classical['F1-Score']
    
    # Handle case where all quantum values are identical
    if len(set(quantum_f1)) == 1:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = stats.ttest_1samp(quantum_f1, classical_f1)
    
    statistical_results.append({
        'Dataset Size': size,
        'Mean Quantum F1': quantum_f1.mean(),
        'Std Quantum F1': quantum_f1.std(),
        'Classical F1': classical_f1,
        'Difference': quantum_f1.mean() - classical_f1,
        't-statistic': t_stat,
        'p-value': p_value,
        'Significant (alpha=0.05)': 'Yes' if (not np.isnan(p_value) and p_value < 0.05) else 'No'
    })

stats_df = pd.DataFrame(statistical_results)
print("\nStatistical Comparison:")
print(stats_df.to_string(index=False))

stats_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_analysis.csv'), index=False)

# ========================================
# STEP 3: Paper Figure 1 - Performance Comparison
# ========================================
print("\n[STEP 3] Creating Figure 1: Performance Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Get top 3 feature maps by average performance
top_feature_maps = quantum_df.groupby('Feature Map')['F1-Score'].mean().nlargest(3).index.tolist()

# (a) F1-Score Comparison
ax = axes[0, 0]
for fm in top_feature_maps:
    data = quantum_df[quantum_df['Feature Map'] == fm].sort_values('Dataset Size')
    ax.plot(data['Dataset Size'], data['F1-Score'], 
           marker='o', label=fm, linewidth=2, markersize=8)

ax.axhline(y=best_classical['F1-Score'], color='red', linestyle='--',
          linewidth=2.5, label=f'Classical ({best_classical["Model"]})')
ax.set_xlabel('Training Dataset Size')
ax.set_ylabel('F1-Score')
ax.set_title('(a) F1-Score Comparison')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.85, 1.05])

# (b) Accuracy Comparison
ax = axes[0, 1]
for fm in top_feature_maps:
    data = quantum_df[quantum_df['Feature Map'] == fm].sort_values('Dataset Size')
    ax.plot(data['Dataset Size'], data['Accuracy'],
           marker='s', label=fm, linewidth=2, markersize=8)

ax.axhline(y=best_classical['Accuracy'], color='red', linestyle='--',
          linewidth=2.5, label=f'Classical ({best_classical["Model"]})')
ax.set_xlabel('Training Dataset Size')
ax.set_ylabel('Accuracy')
ax.set_title('(b) Accuracy Comparison')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.85, 1.05])

# (c) Training Time Comparison
ax = axes[1, 0]
for fm in top_feature_maps:
    data = quantum_df[quantum_df['Feature Map'] == fm].sort_values('Dataset Size')
    ax.plot(data['Dataset Size'], data['Total Time (s)'],
           marker='^', label=fm, linewidth=2, markersize=8)

ax.axhline(y=best_classical['Training Time (s)'], color='red', linestyle='--',
          linewidth=2.5, label=f'Classical ({best_classical["Model"]})')
ax.set_xlabel('Training Dataset Size')
ax.set_ylabel('Total Training Time (s)')
ax.set_title('(c) Computational Time Comparison')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (d) Performance vs Time Trade-off
ax = axes[1, 1]
# Best quantum per dataset size
best_quantum_per_size = quantum_df.groupby('Dataset Size', as_index=False).apply(
    lambda x: x.loc[x['F1-Score'].idxmax()], include_groups=False
)

scatter = ax.scatter(best_quantum_per_size['Total Time (s)'], 
                    best_quantum_per_size['F1-Score'],
                    s=200, c=best_quantum_per_size['Dataset Size'],
                    cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2,
                    label='Quantum (Best per size)')

ax.scatter(best_classical['Training Time (s)'], best_classical['F1-Score'],
          s=300, color='red', marker='*', edgecolors='black', linewidth=2,
          label='Classical (Best)', zorder=10)

ax.set_xlabel('Training Time (s)')
ax.set_ylabel('F1-Score')
ax.set_title('(d) Performance-Efficiency Trade-off')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Dataset Size', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'figure1_performance_comparison.png'),
           dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(PAPER_DIR, 'figure1_performance_comparison.pdf'),
           bbox_inches='tight')
print("✓ Figure 1 saved (PNG + PDF)")
plt.close()

# ========================================
# STEP 4: Paper Figure 2 - Quantum Circuit Analysis
# ========================================
print("\n[STEP 4] Creating Figure 2: Quantum Circuit Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Get one entry per feature map (use largest dataset or average)
circuit_data = quantum_df.groupby('Feature Map', as_index=False).agg({
    'Circuit Depth': 'first',
    'Number of Gates': 'first',
    'F1-Score': 'mean',
    'Total Time (s)': 'mean',
    'Dataset Size': 'max'
})

# (a) Circuit Complexity vs Performance
ax = axes[0]
scatter = ax.scatter(circuit_data['Circuit Depth'], circuit_data['F1-Score'],
                    s=circuit_data['Number of Gates']*5, 
                    c=circuit_data['Total Time (s)'],
                    cmap='plasma', alpha=0.6, edgecolors='black', linewidth=1.5)

for idx, row in circuit_data.iterrows():
    ax.annotate(row['Feature Map'].replace('ZZ-', '').replace('Pauli-', 'P-'), 
               (row['Circuit Depth'], row['F1-Score']),
               fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Circuit Depth')
ax.set_ylabel('Average F1-Score')
ax.set_title('(a) Circuit Complexity vs Performance')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Avg Time (s)', rotation=270, labelpad=15)

# (b) Feature Map Performance
ax = axes[1]
fm_performance = quantum_df.groupby('Feature Map')['F1-Score'].mean().sort_values(ascending=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(fm_performance)))
bars = ax.barh(range(len(fm_performance)), fm_performance.values, color=colors)
ax.set_yticks(range(len(fm_performance)))
ax.set_yticklabels([fm.replace('ZZ-', '').replace('Pauli-', 'P-') for fm in fm_performance.index], fontsize=9)
ax.set_xlabel('Average F1-Score')
ax.set_title('(b) Feature Map Performance')
ax.axvline(x=best_classical['F1-Score'], color='red', linestyle='--',
          linewidth=2, label='Classical Best')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([0.95, 1.01])

# (c) Scalability: Time per Sample
ax = axes[2]
for fm in top_feature_maps:
    data = quantum_df[quantum_df['Feature Map'] == fm].sort_values('Dataset Size')
    time_per_sample = data['Total Time (s)'] / data['Dataset Size']
    ax.plot(data['Dataset Size'], time_per_sample,
           marker='o', label=fm, linewidth=2, markersize=6)

ax.set_xlabel('Dataset Size')
ax.set_ylabel('Time per Sample (s)')
ax.set_title('(c) Scalability Analysis')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'figure2_circuit_analysis.png'),
           dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(PAPER_DIR, 'figure2_circuit_analysis.pdf'),
           bbox_inches='tight')
print("✓ Figure 2 saved (PNG + PDF)")
plt.close()

# ========================================
# STEP 5: Paper Table 1 - Results Summary
# ========================================
print("\n[STEP 5] Creating Table 1: Results Summary...")

# Top 3 classical models
top_classical = classical_df.nlargest(3, 'F1-Score')[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)']]

# Best quantum for each dataset size
best_quantum_table = []
for size in sorted(dataset_sizes):
    size_data = quantum_df[quantum_df['Dataset Size'] == size]
    best = size_data.loc[size_data['F1-Score'].idxmax()]
    best_quantum_table.append({
        'Model': f"Quantum ({best['Feature Map']}, n={size})",
        'Accuracy': best['Accuracy'],
        'Precision': best['Precision'],
        'Recall': best['Recall'],
        'F1-Score': best['F1-Score'],
        'Training Time (s)': best['Total Time (s)']
    })

best_quantum_df = pd.DataFrame(best_quantum_table)

# Combine tables
results_table = pd.concat([top_classical, best_quantum_df], ignore_index=True)

# Format for LaTeX
results_table_latex = results_table.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    results_table_latex[col] = results_table_latex[col].apply(lambda x: f"{x:.4f}")
results_table_latex['Training Time (s)'] = results_table_latex['Training Time (s)'].apply(lambda x: f"{x:.2f}")

print("\nTable 1: Model Performance Comparison")
print(results_table_latex.to_string(index=False))

# Save as CSV and LaTeX
results_table.to_csv(os.path.join(RESULTS_DIR, 'table1_results_summary.csv'), index=False)
with open(os.path.join(PAPER_DIR, 'table1_results_summary.tex'), 'w') as f:
    f.write(results_table_latex.to_latex(index=False, escape=False))

print(f"\n✓ Table 1 saved")

# ========================================
# STEP 6: Paper Figure 3 - Quantum Advantage Analysis
# ========================================
print("\n[STEP 6] Creating Figure 3: Quantum Advantage Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Performance Improvement Heatmap
ax = axes[0]
improvement_matrix = []
feature_maps = sorted(quantum_df['Feature Map'].unique())

for fm in feature_maps:
    improvements = []
    for size in sorted(dataset_sizes):
        data = quantum_df[(quantum_df['Feature Map'] == fm) & 
                         (quantum_df['Dataset Size'] == size)]
        if len(data) > 0:
            improvement = (data.iloc[0]['F1-Score'] - best_classical['F1-Score']) * 100
            improvements.append(improvement)
        else:
            improvements.append(np.nan)
    improvement_matrix.append(improvements)

improvement_df = pd.DataFrame(improvement_matrix, 
                             columns=[f'{s}' for s in sorted(dataset_sizes)],
                             index=[fm.replace('ZZ-', '').replace('Pauli-', 'P-') for fm in feature_maps])

sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
           ax=ax, cbar_kws={'label': 'Improvement (%)'}, vmin=-5, vmax=5)
ax.set_title('(a) F1-Score Improvement over Classical (%)')
ax.set_xlabel('Training Dataset Size')
ax.set_ylabel('Feature Map')

# (b) Statistical Significance
ax = axes[1]
x_pos = np.arange(len(stats_df))
colors = ['green' if (not np.isnan(p) and p < 0.05) else 'gray' 
          for p in stats_df['p-value']]

bars = ax.bar(x_pos, stats_df['Difference']*100, color=colors, alpha=0.7,
             edgecolor='black', linewidth=1.5)

# Add significance markers
for i, (idx, row) in enumerate(stats_df.iterrows()):
    if not np.isnan(row['p-value']) and row['p-value'] < 0.05:
        ax.text(i, row['Difference']*100 + 0.2, '*', 
               ha='center', va='bottom', fontsize=20, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Dataset Size')
ax.set_ylabel('F1-Score Difference (%)')
ax.set_title('(b) Statistical Significance of Performance Difference')
ax.set_xticks(x_pos)
ax.set_xticklabels(stats_df['Dataset Size'])
ax.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label='p < 0.05 (Significant)'),
                  Patch(facecolor='gray', alpha=0.7, label='p ≥ 0.05 or N/A')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, 'figure3_quantum_advantage.png'),
           dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(PAPER_DIR, 'figure3_quantum_advantage.pdf'),
           bbox_inches='tight')
print("✓ Figure 3 saved (PNG + PDF)")
plt.close()

# ========================================
# STEP 7: Summary Report
# ========================================
print("\n[STEP 7] Generating summary report...")

report = f"""
{'='*80}
QUANTUM IDS PROJECT - FINAL REPORT
{'='*80}

1. DATASET INFORMATION
   - Number of Qubits: {quantum_df.iloc[0]['Number of Qubits']}
   - Training Sizes Tested: {', '.join(map(str, sorted(dataset_sizes)))}
   - Feature Maps Tested: {len(quantum_df['Feature Map'].unique())}

2. CLASSICAL BASELINE RESULTS
   Best Model: {best_classical['Model']}
   - Accuracy:  {best_classical['Accuracy']:.4f}
   - Precision: {best_classical['Precision']:.4f}
   - Recall:    {best_classical['Recall']:.4f}
   - F1-Score:  {best_classical['F1-Score']:.4f}
   - Training Time: {best_classical['Training Time (s)']:.2f}s

3. QUANTUM KERNEL RESULTS
   Best Configuration: {best_quantum['Feature Map']} (n={best_quantum['Dataset Size']})
   - Accuracy:  {best_quantum['Accuracy']:.4f}
   - Precision: {best_quantum['Precision']:.4f}
   - Recall:    {best_quantum['Recall']:.4f}
   - F1-Score:  {best_quantum['F1-Score']:.4f}
   - Training Time: {best_quantum['Total Time (s)']:.2f}s
   - Circuit Depth: {best_quantum['Circuit Depth']}
   - Number of Gates: {best_quantum['Number of Gates']}
   - Backend: {best_quantum['Backend']}

4. COMPARATIVE ANALYSIS
   Performance Difference:
   - F1-Score: {(best_quantum['F1-Score'] - best_classical['F1-Score'])*100:+.2f}%
   - Accuracy: {(best_quantum['Accuracy'] - best_classical['Accuracy'])*100:+.2f}%
   
   Time Comparison:
   - Quantum Time: {best_quantum['Total Time (s)']:.2f}s
   - Classical Time: {best_classical['Training Time (s)']:.2f}s
   - Ratio: {best_quantum['Total Time (s)'] / best_classical['Training Time (s)']:.2f}x

5. STATISTICAL SIGNIFICANCE
   Valid Tests: {sum(~stats_df['p-value'].isna())}
   Significant Results (alpha=0.05): {sum(stats_df['p-value'] < 0.05)}/{len(stats_df)}
   
6. KEY FINDINGS
   - Quantum kernels {'achieved equivalent performance' if abs(best_quantum['F1-Score'] - best_classical['F1-Score']) < 0.01 else ('outperformed' if best_quantum['F1-Score'] > best_classical['F1-Score'] else 'underperformed')} compared to classical methods
   - Best quantum feature map: {best_quantum['Feature Map']}
   - Average F1-Score across sizes: {quantum_df.groupby('Feature Map')['F1-Score'].mean().max():.4f}
   - Computational overhead: {best_quantum['Total Time (s)'] / best_classical['Training Time (s)']:.1f}x slower than classical
   - Most efficient feature map: {quantum_df.groupby('Feature Map')['Total Time (s)'].mean().idxmin()}

7. RECOMMENDATIONS FOR PAPER
   - Focus on feature map comparison: {', '.join(top_feature_maps)}
   - Highlight: {'quantum achieves competitive performance' if abs(best_quantum['F1-Score'] - best_classical['F1-Score']) < 0.02 else 'quantum shows promise but needs optimization'}
   - Discuss trade-offs: Performance ({best_quantum['F1-Score']:.3f}) vs Time ({best_quantum['Total Time (s)']:.1f}s)
   - Future work: Hardware implementation, noise analysis, larger qubit counts

8. PUBLICATION-READY OUTPUTS
   ✓ Figure 1: Performance Comparison (4 subplots)
   ✓ Figure 2: Circuit Analysis (3 subplots)
   ✓ Figure 3: Quantum Advantage Analysis (2 subplots)
   ✓ Table 1: Model Performance Summary (LaTeX + CSV)
   ✓ Statistical Analysis (CSV)
   ✓ Final Report (TXT)

{'='*80}
"""

print(report)

# Save report with UTF-8 encoding
with open(os.path.join(RESULTS_DIR, 'final_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✓ Final report saved to {RESULTS_DIR}/final_report.txt")

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated Files:")
print(f"  📊 Paper Figures (PNG @ 600 DPI + PDF):")
print(f"     {PAPER_DIR}/")
print(f"       - figure1_performance_comparison")
print(f"       - figure2_circuit_analysis")
print(f"       - figure3_quantum_advantage")
print(f"  📋 Tables:")
print(f"       - {PAPER_DIR}/table1_results_summary.tex")
print(f"       - {RESULTS_DIR}/table1_results_summary.csv")
print(f"  📄 Reports:")
print(f"       - {RESULTS_DIR}/final_report.txt")
print(f"       - {RESULTS_DIR}/statistical_analysis.csv")
print(f"\n🎓 All results ready for paper writing!")
print("="*80)