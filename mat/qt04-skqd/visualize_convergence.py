"""Visualize Boston energy convergence"""

import numpy as np
import matplotlib.pyplot as plt

# Load result
data = np.load('result_boston.npz')
energy_history = data['energy_history']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Boston Post-Processing Convergence Analysis', fontsize=16, fontweight='bold')

# 1. Energy vs Iteration
iterations = np.arange(1, len(energy_history) + 1)
ax1.plot(iterations, energy_history, 'o-', linewidth=2.5, markersize=10, 
         color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Energy (Hartree)', fontsize=13, fontweight='bold')
ax1.set_title('Energy Convergence', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(iterations)

# Add final energy annotation
final_energy = energy_history[-1]
ax1.annotate(f'Final: {final_energy:.4f} Ha', 
             xy=(len(energy_history), final_energy),
             xytext=(len(energy_history) - 1.5, final_energy + 0.15),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', lw=2))

# 2. Energy Change per Iteration
energy_changes = np.diff(energy_history)
change_iterations = np.arange(2, len(energy_history) + 1)

bars = ax2.bar(change_iterations, energy_changes, color='#ff7f0e', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax2.set_ylabel('Energy Change (Hartree)', fontsize=13, fontweight='bold')
ax2.set_title('Energy Decrease per Iteration', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(change_iterations)
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Add value labels on bars
for bar, val in zip(bars, energy_changes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height - 0.01 if height < 0 else height + 0.005,
            f'{val:.3f}',
            ha='center', va='top' if height < 0 else 'bottom', 
            fontsize=9, fontweight='bold')

# Summary statistics
total_change = energy_history[-1] - energy_history[0]
avg_change = total_change / (len(energy_history) - 1)
summary_text = f"Total Energy Drop: {total_change:.4f} Ha\n"
summary_text += f"Average per Iteration: {avg_change:.4f} Ha\n"
summary_text += f"Largest Single Drop: {min(energy_changes):.4f} Ha (Iter {change_iterations[np.argmin(energy_changes)]})"

fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=0.8),
         family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('boston_convergence.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Convergence visualization saved as 'boston_convergence.png'")
print(f"\nConvergence Summary:")
print(f"  Starting energy: {energy_history[0]:.6f} Ha")
print(f"  Final energy:    {energy_history[-1]:.6f} Ha")
print(f"  Total change:    {total_change:.6f} Ha")
print(f"  Iterations:      {len(energy_history)}")
plt.show()
