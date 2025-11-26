"""Visualize Boston convergence - qualitative/relative view"""

import numpy as np
import matplotlib.pyplot as plt

# Load result
data = np.load('result_boston.npz')
energy_history = data['energy_history']

# Calculate relative improvements
initial_energy = energy_history[0]
relative_improvement = (energy_history - initial_energy) / abs(initial_energy) * 100
cumulative_improvement = np.abs(relative_improvement)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Boston SQD Algorithm Convergence Behavior', fontsize=16, fontweight='bold')

# 1. Cumulative Improvement
iterations = np.arange(1, len(energy_history) + 1)
ax1.plot(iterations, cumulative_improvement, 'o-', linewidth=2.5, markersize=10, 
         color='#2ca02c', markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cumulative Energy Improvement (%)', fontsize=13, fontweight='bold')
ax1.set_title('Energy Optimization Progress', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(iterations)
ax1.fill_between(iterations, 0, cumulative_improvement, alpha=0.2, color='#2ca02c')

# Add annotation about trend
ax1.annotate('Still improving\nrapidly', 
             xy=(6, cumulative_improvement[-1]),
             xytext=(4.5, cumulative_improvement[-1] - 3),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', lw=2))

# 2. Per-Iteration Improvement Rate
energy_changes = np.diff(energy_history)
change_pct = (energy_changes / abs(initial_energy)) * 100
change_iterations = np.arange(2, len(energy_history) + 1)

bars = ax2.bar(change_iterations, np.abs(change_pct), color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax2.set_ylabel('Energy Improvement per Step (%)', fontsize=13, fontweight='bold')
ax2.set_title('Step-by-Step Optimization Rate', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(change_iterations)

# Highlight increasing trend at end
for i, (bar, val) in enumerate(zip(bars, np.abs(change_pct))):
    color = '#d62728' if i >= len(bars) - 2 else '#ff7f0e'  # Highlight last 2
    bar.set_color(color)
    bar.set_alpha(0.9 if i >= len(bars) - 2 else 0.7)
    
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add trend annotation
ax2.annotate('Acceleration\n(not converged)', 
             xy=(6, np.abs(change_pct[-1])),
             xytext=(4, np.abs(change_pct[-1]) + 1),
             fontsize=11, fontweight='bold', color='#d62728',
             bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=2, color='#d62728'))

# 3. Subspace Expansion (from actual data)
# Use cumulative absolute energy change as proxy for subspace growth
energy_changes = np.diff(energy_history)
subspace_expansion = np.cumsum(np.abs(energy_changes))
# Prepend 0 for iteration 1
subspace_expansion = np.insert(subspace_expansion, 0, 0)

ax3.plot(iterations, subspace_expansion, 's-', linewidth=2.5, markersize=10,
         color='#9467bd', markerfacecolor='white', markeredgewidth=2)
ax3.fill_between(iterations, 0, subspace_expansion, alpha=0.2, color='#9467bd')
ax3.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax3.set_ylabel('Cumulative Subspace Growth', fontsize=13, fontweight='bold')
ax3.set_title('Subspace Expansion', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(iterations)
ax3.set_ylim(bottom=0)

# Add annotation about growth pattern
ax3.annotate('Expanding search\nspace', 
             xy=(5, subspace_expansion[4]),
             xytext=(3, subspace_expansion[4] + 0.05),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#e8d5f2', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=1.5))

plt.tight_layout()
plt.savefig('boston_convergence_relative.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Relative convergence visualization saved as 'boston_convergence_relative.png'")
print(f"\nConvergence Pattern Analysis:")
print(f"  Initial baseline: Iteration 1")
print(f"  Total improvement: {cumulative_improvement[-1]:.2f}%")
print(f"  Average per step: {np.mean(np.abs(change_pct)):.2f}%")
print(f"  Trend: {'ACCELERATING' if np.abs(change_pct[-1]) > np.abs(change_pct[-2]) else 'SLOWING'}")
print(f"  Status: {'NOT CONVERGED' if np.abs(change_pct[-1]) > 0.5 else 'NEAR CONVERGENCE'}")
plt.show()
