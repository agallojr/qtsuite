#!/usr/bin/env python3
"""Analyze circuit scaling study results from checkpoints"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python analyze_scaling.py <workflow_id>")
    sys.exit(1)

workflow_id = sys.argv[1]
workflow_dir = Path(f"/Users/agallojr/.lwfm/out/qt02-cfd/{workflow_id}")

# Load checkpoint files
checkpoint_files = sorted(workflow_dir.glob("checkpoint_*.json"))

if not checkpoint_files:
    print(f"No checkpoint files found in {workflow_dir}")
    sys.exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files")

# Collect data
results = []
for ckpt_file in checkpoint_files:
    with open(ckpt_file, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
        params = ckpt['params']
        
        nx = params.get('nx')
        ny = params.get('ny')
        matrix_size_orig = params.get('_matrix_size_original')
        matrix_size_herm = params.get('_matrix_size')
        cond_orig = params.get('_matrix_condition_number_original')
        cond_herm = params.get('_matrix_condition_number')
        circuit_qubits = params.get('_circuit_qubits')
        circuit_depth = params.get('_circuit_depth')
        circuit_gates = params.get('_circuit_gates')
        circuit_depth_trans = params.get('_circuit_depth_transpiled')
        circuit_gates_trans = params.get('_circuit_gates_transpiled')
        t_construct = params.get('_time_circuit_construction')
        t_gen = params.get('_time_circuit_generation')
        
        results.append({
            'mesh': f"{nx}x{ny}",
            'nx': nx,
            'ny': ny,
            'grid_points': nx * ny,
            'matrix_size_original': matrix_size_orig,
            'matrix_size_hermitian': matrix_size_herm,
            'condition_number_original': cond_orig,
            'condition_number_hermitian': cond_herm,
            'circuit_qubits': circuit_qubits,
            'circuit_depth': circuit_depth,
            'circuit_gates': circuit_gates,
            'circuit_depth_transpiled': circuit_depth_trans,
            'circuit_gates_transpiled': circuit_gates_trans,
            'time_construction_sec': t_construct,
            'time_generation_sec': t_gen
        })

# Sort by grid points
results.sort(key=lambda x: x['grid_points'])

# Print table
print("\n" + "="*120)
print("Circuit Scaling Study Results")
print("="*120)
print(f"{'Mesh':<8} {'Grid':<6} {'Matrix':<12} {'Hermitian':<12} {'κ_orig':<12} {'κ_herm':<12} {'Qubits':<8} {'Depth':<8} {'Gates':<8} {'T_const(s)':<12}")
print("-"*120)

for r in results:
    print(f"{r['mesh']:<8} "
          f"{r['grid_points']:<6} "
          f"{r['matrix_size_original'] or 'N/A':<12} "
          f"{r['matrix_size_hermitian'] or 'N/A':<12} "
          f"{r['condition_number_original']:<12.2e} " if r['condition_number_original'] else f"{'N/A':<12} "
          f"{r['condition_number_hermitian']:<12.2e} " if r['condition_number_hermitian'] else f"{'N/A':<12} "
          f"{r['circuit_qubits'] or 'N/A':<8} "
          f"{r['circuit_depth'] or 'N/A':<8} "
          f"{r['circuit_gates'] or 'N/A':<8} "
          f"{r['time_construction_sec']:<12.2f}" if r['time_construction_sec'] else f"{'N/A':<12}")

print("="*120)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

grid_points = [r['grid_points'] for r in results]
matrix_sizes_herm = [r['matrix_size_hermitian'] for r in results if r['matrix_size_hermitian']]
cond_numbers = [r['condition_number_hermitian'] for r in results if r['condition_number_hermitian']]
circuit_depths = [r['circuit_depth'] for r in results if r['circuit_depth']]
circuit_qubits = [r['circuit_qubits'] for r in results if r['circuit_qubits']]
mesh_labels = [r['mesh'] for r in results]

# Plot 1: Matrix size vs grid points
ax = axes[0, 0]
if matrix_sizes_herm:
    ax.plot(grid_points[:len(matrix_sizes_herm)], matrix_sizes_herm, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold')
    ax.set_ylabel('Hermitian Matrix Size', fontweight='bold')
    ax.set_title('Matrix Size Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grid_points[:len(matrix_sizes_herm)])
    ax.set_xticklabels([mesh_labels[i] for i in range(len(matrix_sizes_herm))])

# Plot 2: Condition number vs grid points
ax = axes[0, 1]
if cond_numbers:
    ax.semilogy(grid_points[:len(cond_numbers)], cond_numbers, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold')
    ax.set_ylabel('Condition Number (κ)', fontweight='bold')
    ax.set_title('Matrix Conditioning', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grid_points[:len(cond_numbers)])
    ax.set_xticklabels([mesh_labels[i] for i in range(len(cond_numbers))])

# Plot 3: Circuit depth vs grid points
ax = axes[1, 0]
if circuit_depths:
    ax.plot(grid_points[:len(circuit_depths)], circuit_depths, '^-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold')
    ax.set_ylabel('Circuit Depth', fontweight='bold')
    ax.set_title('Circuit Depth Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grid_points[:len(circuit_depths)])
    ax.set_xticklabels([mesh_labels[i] for i in range(len(circuit_depths))])

# Plot 4: Circuit qubits vs grid points
ax = axes[1, 1]
if circuit_qubits:
    ax.plot(grid_points[:len(circuit_qubits)], circuit_qubits, 'd-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold')
    ax.set_ylabel('Number of Qubits', fontweight='bold')
    ax.set_title('Qubit Count Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grid_points[:len(circuit_qubits)])
    ax.set_xticklabels([mesh_labels[i] for i in range(len(circuit_qubits))])

plt.tight_layout()
output_path = workflow_dir / "scaling_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Save summary JSON
summary = {
    'workflow_id': workflow_id,
    'num_cases': len(results),
    'results': results
}
json_path = workflow_dir / "scaling_summary.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to: {json_path}")
