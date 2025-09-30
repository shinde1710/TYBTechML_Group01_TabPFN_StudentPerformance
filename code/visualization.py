# Visualization of Model Comparison Results

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Load results
results_df = pd.read_csv('results/new_dataset_results.csv', index_col=0)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
results_df['Accuracy'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# ROC-AUC
results_df['ROC-AUC'].plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
axes[1].set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('ROC-AUC', fontsize=12)
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Time
results_df['Time(s)'].plot(kind='bar', ax=axes[2], color='salmon', edgecolor='black')
axes[2].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Time (seconds)', fontsize=12)
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved to figures/comparison.png")
