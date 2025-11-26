"""Simplified visualization comparing Boston vs Kingston quantum data"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import BitArray
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'boston': '#1f77b4', 'kingston': '#ff7f0e'}

# Load data
print("Loading data...")
boston_data = np.load('bit_array_boston.npz')
boston_samples = boston_data['samples']
boston_bits = int(boston_data['num_bits'])

kingston_data = np.load('bit_array_kingston.npz')
kingston_samples = kingston_data['samples']
kingston_bits = int(kingston_data['num_bits'])

# Reconstruct BitArray
boston_array = BitArray(boston_samples, num_bits=boston_bits)
kingston_array = BitArray(kingston_samples, num_bits=kingston_bits)

# Unpack bits
boston_unpacked = np.unpackbits(boston_array.array, axis=-1)[..., -boston_bits:].astype(bool)
kingston_unpacked = np.unpackbits(kingston_array.array, axis=-1)[..., -kingston_bits:].astype(bool)

# Calculate metrics
boston_hamming = np.sum(boston_unpacked, axis=1)
kingston_hamming = np.sum(kingston_unpacked, axis=1)

boston_unique = len(np.unique(boston_unpacked, axis=0))
kingston_unique = len(np.unique(kingston_unpacked, axis=0))

boston_tuples = [tuple(row) for row in boston_unpacked]
kingston_tuples = [tuple(row) for row in kingston_unpacked]
boston_counts = Counter(boston_tuples)
kingston_counts = Counter(kingston_tuples)

def shannon_entropy(counts, total):
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

boston_entropy = shannon_entropy(boston_counts, len(boston_unpacked))
kingston_entropy = shannon_entropy(kingston_counts, len(kingston_unpacked))

# Create figure with 2 subplots and extra space for text below
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax_text1 = fig.add_subplot(gs[1, 0])
ax_text2 = fig.add_subplot(gs[1, 1])

fig.suptitle('IBM Quantum Backend Comparison: Boston vs Kingston', 
             fontsize=18, fontweight='bold')

# 1. Particle Number Distribution
bins = np.arange(9, 40, 1)
ax1.hist(boston_hamming, bins=bins, alpha=0.6, label='Boston', color=colors['boston'], density=True)
ax1.hist(kingston_hamming, bins=bins, alpha=0.6, label='Kingston', color=colors['kingston'], density=True)
ax1.axvline(24, color='red', linestyle='--', linewidth=2.5, label='Expected (24)', alpha=0.8)
ax1.set_xlabel('Particle Number', fontsize=13, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
ax1.set_title('Particle Number Distribution', fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=11)

# 2. Quality Metrics Bar Chart
metrics = ['Diversity (%)', 'Entropy (bits)', 'Mean Particles']
boston_vals = [
    boston_unique/len(boston_unpacked)*100,
    boston_entropy,
    np.mean(boston_hamming)
]
kingston_vals = [
    kingston_unique/len(kingston_unpacked)*100,
    kingston_entropy,
    np.mean(kingston_hamming)
]

# Normalize for visualization
boston_norm = [boston_vals[0]/100, boston_vals[1]/15.322, boston_vals[2]/24]
kingston_norm = [kingston_vals[0]/100, kingston_vals[1]/15.322, kingston_vals[2]/24]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax2.bar(x - width/2, boston_norm, width, label='Boston', color=colors['boston'], alpha=0.85)
bars2 = ax2.bar(x + width/2, kingston_norm, width, label='Kingston', color=colors['kingston'], alpha=0.85)

ax2.set_ylabel('Normalized Score', fontsize=13, fontweight='bold')
ax2.set_title('Quality Metrics (Normalized to 1.0)', fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax2.legend(fontsize=12)
ax2.set_ylim([0.90, 1.01])
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(labelsize=11)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

# Text annotations below charts
ax_text1.axis('off')
ax_text2.axis('off')

# Stats text below particle distribution
stats_text = "Statistics:\n\n"
stats_text += f"Boston:     μ = {np.mean(boston_hamming):5.2f},   σ = {np.std(boston_hamming):4.2f}\n"
stats_text += f"Kingston:   μ = {np.mean(kingston_hamming):5.2f},   σ = {np.std(kingston_hamming):4.2f}"
ax_text1.text(0.5, 0.5, stats_text, transform=ax_text1.transAxes, 
              fontsize=13, verticalalignment='center', horizontalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1),
              family='monospace', weight='normal')

# Raw values as table below quality metrics
table_data = [
    ['Diversity (%)', f'{boston_vals[0]:.2f}', f'{kingston_vals[0]:.2f}'],
    ['Entropy (bits)', f'{boston_vals[1]:.3f}', f'{kingston_vals[1]:.3f}'],
    ['Mean Particles', f'{boston_vals[2]:.2f}', f'{kingston_vals[2]:.2f}']
]

table = ax_text2.table(cellText=table_data,
                       colLabels=['Metric', 'Boston', 'Kingston'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.4, 0.3, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#4a5f7f')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 4):
    table[(i, 0)].set_facecolor('#e8e8e8')
    table[(i, 0)].set_text_props(weight='bold')
    table[(i, 1)].set_facecolor('#d6e9f8')
    table[(i, 2)].set_facecolor('#ffe4cc')

plt.tight_layout()
plt.savefig('backend_comparison_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Visualization saved as 'backend_comparison_simple.png'")
plt.show()
