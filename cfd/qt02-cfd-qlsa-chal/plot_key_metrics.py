#!/usr/bin/env python3
"""Plot key metrics: K, qubits, depth"""

import matplotlib.pyplot as plt
from pathlib import Path

# Data from scaling study
data = {
    '2x2': {'grid_points': 4, 'kappa': 1, 'qubits': 7, 'depth': 243},
    '3x3': {'grid_points': 9, 'kappa': 43, 'qubits': 13, 'depth': 2156878},
    '4x4': {'grid_points': 16, 'kappa': 160, 'qubits': 15, 'depth': 8592323},
    '5x5': {'grid_points': 25, 'kappa': 460, 'qubits': 17, 'depth': 72522301}
}

mesh_labels = list(data.keys())
grid_points = [data[m]['grid_points'] for m in mesh_labels]
kappas = [data[m]['kappa'] for m in mesh_labels]
qubits = [data[m]['qubits'] for m in mesh_labels]
depths = [data[m]['depth'] for m in mesh_labels]

# Create figure with single plot and dual y-axes
fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('HHL Circuit Scaling: Condition Number, Qubits, and Depth', 
             fontsize=16, fontweight='bold', y=0.98)

# Left y-axis: Condition Number and Circuit Depth (log scale)
ax1.set_xlabel('Mesh Size', fontweight='bold', fontsize=14)
ax1.set_ylabel('Condition Number (κ) and Circuit Depth (log scale)', fontweight='bold', fontsize=13, color='black')
ax1.set_xticks(grid_points)
ax1.set_xticklabels(mesh_labels, fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Plot Condition Number (log scale)
line1 = ax1.semilogy(grid_points, kappas, 's-', linewidth=3, markersize=12, 
                     color='#A23B72', label='Condition Number (κ)', zorder=3)

# Plot Circuit Depth (log scale)
line2 = ax1.semilogy(grid_points, depths, 'o-', linewidth=3, markersize=12, 
                     color='#06A77D', label='Circuit Depth', zorder=3)

ax1.tick_params(axis='y', labelsize=11)

# Right y-axis: Qubit Count (linear scale)
ax2 = ax1.twinx()
ax2.set_ylabel('Number of Qubits', fontweight='bold', fontsize=13, color='#F18F01')
line3 = ax2.plot(grid_points, qubits, '^-', linewidth=3, markersize=12, 
                 color='#F18F01', label='Qubits', zorder=3)
ax2.tick_params(axis='y', labelcolor='#F18F01', labelsize=11)
ax2.set_ylim([5, 19])

# Add annotations for all three metrics
for i, (x, k, d, q) in enumerate(zip(grid_points, kappas, depths, qubits)):
    # Condition number annotation
    ax1.annotate(f'κ={k:.0f}', xy=(x, k), xytext=(-35, -20), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='#A23B72',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#A23B72', alpha=0.9))
    
    # Depth annotation
    if d < 1000:
        depth_label = f'{d:.0f}'
    elif d < 1e6:
        depth_label = f'{d/1e6:.2f}M'
    else:
        depth_label = f'{d/1e6:.1f}M'
    ax1.annotate(depth_label, xy=(x, d), xytext=(35, 20), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='#06A77D',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#06A77D', alpha=0.9))
    
    # Qubit annotation
    ax2.annotate(f'{q}', xy=(x, q), xytext=(0, -25), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold', color='#F18F01',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#F18F01', alpha=0.9))

# Combine legends
lines1 = line1 + line2
labels1 = [l.get_label() for l in lines1]
lines2 = line3
labels2 = [l.get_label() for l in lines2]
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
output_path = Path("/Users/agallojr/.lwfm/out/qt02-cfd/0bbb27f2") / "key_metrics.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved plot to: {output_path}")
