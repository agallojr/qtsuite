"""Create visualization comparing Boston vs Kingston quantum data"""

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

boston_concentration = sum([(count/len(boston_unpacked))**2 for count in boston_counts.values()])
kingston_concentration = sum([(count/len(kingston_unpacked))**2 for count in kingston_counts.values()])

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Title
fig.suptitle('IBM Quantum Backend Comparison: Boston vs Kingston', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Particle Number Distribution (large, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
bins = np.arange(9, 40, 1)
ax1.hist(boston_hamming, bins=bins, alpha=0.6, label='Boston', color=colors['boston'], density=True)
ax1.hist(kingston_hamming, bins=bins, alpha=0.6, label='Kingston', color=colors['kingston'], density=True)
ax1.axvline(24, color='red', linestyle='--', linewidth=2, label='Expected (24)', alpha=0.7)
ax1.set_xlabel('Particle Number', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('Particle Number Conservation', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add stats as text
stats_text = f"Boston: μ={np.mean(boston_hamming):.2f}, σ={np.std(boston_hamming):.2f}\n"
stats_text += f"Kingston: μ={np.mean(kingston_hamming):.2f}, σ={np.std(kingston_hamming):.2f}"
ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Quality Metrics Bar Chart
ax2 = fig.add_subplot(gs[0, 2])
metrics = ['Diversity\n(%)', 'Entropy\n(bits)', 'Mean\nParticles']
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
ax2.bar(x - width/2, boston_norm, width, label='Boston', color=colors['boston'], alpha=0.8)
ax2.bar(x + width/2, kingston_norm, width, label='Kingston', color=colors['kingston'], alpha=0.8)
ax2.set_ylabel('Normalized Score', fontsize=10)
ax2.set_title('Quality Metrics', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=9)
ax2.legend(fontsize=9)
ax2.set_ylim([0.93, 1.01])
ax2.grid(True, alpha=0.3, axis='y')

# 3. State Concentration - Top 10
ax3 = fig.add_subplot(gs[1, :2])
top_n = 10
boston_top = boston_counts.most_common(top_n)
kingston_top = kingston_counts.most_common(top_n)

ranks = np.arange(1, top_n+1)
boston_probs = [count/len(boston_unpacked)*100 for _, count in boston_top]
kingston_probs = [count/len(kingston_unpacked)*100 for _, count in kingston_top]

ax3.plot(ranks, boston_probs, 'o-', label='Boston', color=colors['boston'], 
         linewidth=2, markersize=8, alpha=0.8)
ax3.plot(ranks, kingston_probs, 's-', label='Kingston', color=colors['kingston'], 
         linewidth=2, markersize=8, alpha=0.8)
ax3.set_xlabel('State Rank', fontsize=11)
ax3.set_ylabel('Probability (%)', fontsize=11)
ax3.set_title('Top 10 Most Frequent States (Lower = Better Distribution)', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(ranks)

# 4. Concentration Metric Comparison
ax4 = fig.add_subplot(gs[1, 2])
conc_data = [boston_concentration*1000, kingston_concentration*1000]
bars = ax4.bar(['Boston', 'Kingston'], conc_data, 
               color=[colors['boston'], colors['kingston']], alpha=0.8)
ax4.set_ylabel('Concentration (×10⁻³)', fontsize=10)
ax4.set_title('Sampling Concentration\n(Lower = Better)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, conc_data)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Summary Scorecard (spans all 3 columns)
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Calculate scores
expected_particles = 24
metrics_comparison = {
    'Unique States': (boston_unique, kingston_unique, 'higher'),
    'Particle Conservation': (abs(np.mean(boston_hamming) - expected_particles), 
                              abs(np.mean(kingston_hamming) - expected_particles), 'lower'),
    'Particle Stability (σ)': (np.std(boston_hamming), np.std(kingston_hamming), 'lower'),
    'Shannon Entropy': (boston_entropy, kingston_entropy, 'higher'),
    'Uniformity': (boston_concentration, kingston_concentration, 'lower')
}

boston_score = 0
kingston_score = 0
results = []

for metric, (b_val, k_val, better) in metrics_comparison.items():
    if better == 'higher':
        winner = 'Kingston' if k_val > b_val else 'Boston'
    else:
        winner = 'Kingston' if k_val < b_val else 'Boston'
    
    if winner == 'Boston':
        boston_score += 1
    else:
        kingston_score += 1
    
    results.append((metric, winner))

# Create summary table
table_data = []
for metric, winner in results:
    boston_mark = '✓' if winner == 'Boston' else '○'
    kingston_mark = '✓' if winner == 'Kingston' else '○'
    table_data.append([metric, boston_mark, kingston_mark])

table = ax5.table(cellText=table_data, 
                  colLabels=['Metric', 'Boston', 'Kingston'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.5, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#40466e')
    cell.set_text_props(weight='bold', color='white')

# Style cells with winners
for i in range(1, len(table_data)+1):
    winner = results[i-1][1]
    
    table[(i, 0)].set_facecolor('#f0f0f0')
    
    if winner == 'Boston':
        table[(i, 1)].set_facecolor('#cfe2f3')
        table[(i, 1)].set_text_props(weight='bold', color='green', size=14)
    else:
        table[(i, 1)].set_facecolor('#f0f0f0')
        
    if winner == 'Kingston':
        table[(i, 2)].set_facecolor('#fff2cc')
        table[(i, 2)].set_text_props(weight='bold', color='green', size=14)
    else:
        table[(i, 2)].set_facecolor('#f0f0f0')

# Add title and score
title_text = f"Overall Score:  Boston {boston_score}/5  |  Kingston {kingston_score}/5"
if kingston_score > boston_score:
    winner_text = "★ Kingston shows superior data quality ★"
elif boston_score > kingston_score:
    winner_text = "★ Boston shows superior data quality ★"
else:
    winner_text = "★ Both backends produce comparable quality ★"

ax5.text(0.5, 0.95, title_text, transform=ax5.transAxes,
         fontsize=13, fontweight='bold', ha='center')
ax5.text(0.5, 0.05, winner_text, transform=ax5.transAxes,
         fontsize=12, ha='center', style='italic', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.savefig('backend_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Visualization saved as 'backend_comparison.png'")
plt.show()
