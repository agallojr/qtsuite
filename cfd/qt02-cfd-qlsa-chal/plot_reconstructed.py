#!/usr/bin/env python3
"""Generate plots from reconstructed results"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python plot_reconstructed.py <workflow_id>")
    sys.exit(1)

workflow_id = sys.argv[1]
workflow_dir = Path(f"/Users/agallojr/.lwfm/out/qt02-cfd/{workflow_id}")
pkl_file = workflow_dir / "results_reconstructed.pkl"

if not pkl_file.exists():
    print(f"Results file not found: {pkl_file}")
    sys.exit(1)

# Load data
with open(pkl_file, 'rb') as f:
    case_data = pickle.load(f)

print(f"Loaded {len(case_data)} cases")

# Group by mesh size
data_by_mesh = {}
for case_info in case_data:
    params = case_info['params']
    nx = params.get('nx')
    ny = params.get('ny')
    shots = params.get('qc_shots')
    
    q_sol = case_info.get('quantum_solution')
    c_sol = case_info.get('classical_solution')
    
    if q_sol is None or c_sol is None:
        continue
    
    # Calculate fidelity
    fidelity = float(np.abs(np.dot(q_sol, c_sol)))
    
    mesh_key = f"{nx}x{ny}"
    if mesh_key not in data_by_mesh:
        data_by_mesh[mesh_key] = {'shots': [], 'fidelity': []}
    
    data_by_mesh[mesh_key]['shots'].append(shots)
    data_by_mesh[mesh_key]['fidelity'].append(fidelity)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

colors = {'2x2': 'blue', '3x3': 'orange', '4x4': 'green'}
markers = {'2x2': 'o', '3x3': 's', '4x4': '^'}

for mesh_key in sorted(data_by_mesh.keys()):
    data = data_by_mesh[mesh_key]
    # Sort by shots
    sorted_indices = np.argsort(data['shots'])
    shots = np.array(data['shots'])[sorted_indices]
    fidelity = np.array(data['fidelity'])[sorted_indices]
    
    color = colors.get(mesh_key, 'gray')
    marker = markers.get(mesh_key, 'o')
    
    ax.plot(shots, fidelity, marker=marker, linestyle='-', linewidth=2, 
            markersize=10, label=f"{mesh_key} mesh", color=color)

ax.set_xscale('log')
ax.set_xlabel('Number of Shots', fontsize=14, fontweight='bold')
ax.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
ax.set_title('Quantum Solution Fidelity vs Shot Count\n(Hele-Shaw, Statevector Simulator)', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')
ax.set_ylim([0, 1.05])

# Add horizontal line at 1.0 for reference
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
output_path = workflow_dir / "fidelity_vs_shots.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

# Print summary statistics
print("\nFidelity Summary:")
for mesh_key in sorted(data_by_mesh.keys()):
    data = data_by_mesh[mesh_key]
    fidelities = data['fidelity']
    print(f"\n{mesh_key} mesh:")
    print(f"  Mean: {np.mean(fidelities):.4f}")
    print(f"  Std:  {np.std(fidelities):.4f}")
    print(f"  Min:  {np.min(fidelities):.4f}")
    print(f"  Max:  {np.max(fidelities):.4f}")
