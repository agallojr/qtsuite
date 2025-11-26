#!/usr/bin/env python3
"""Plot untranspiled circuit scaling metrics"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

workflow_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/0bbb27f2")

# Load checkpoint files
checkpoint_files = sorted(workflow_dir.glob("checkpoint_*.json"))

# Untranspiled circuit data from QPY files
untranspiled_data = {
    '2x2': {'qubits': 7, 'depth': 243, 'gates': 353},
    '3x3': {'qubits': 13, 'depth': 2156878, 'gates': 2797979},
    '4x4': {'qubits': 15, 'depth': 8592323, 'gates': 11167864},
    '5x5': {'qubits': 17, 'depth': 72522301, 'gates': 93916071}
}

results = []
for ckpt_file in checkpoint_files:
    with open(ckpt_file, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
        params = ckpt['params']
        mesh = f"{params['nx']}x{params['ny']}"
        
        # Get untranspiled data if available
        unt_data = untranspiled_data.get(mesh, {})
        
        results.append({
            'mesh': mesh,
            'nx': params['nx'],
            'ny': params['ny'],
            'grid_points': params['nx'] * params['ny'],
            'matrix_size_hermitian': params.get('_matrix_size'),
            'condition_number': params.get('_matrix_condition_number'),
            'circuit_qubits': unt_data.get('qubits'),
            'circuit_depth': unt_data.get('depth'),
            'circuit_gates': unt_data.get('gates'),
            'time_construction_sec': params.get('_time_circuit_construction')
        })

# Sort by grid points
results.sort(key=lambda x: x['grid_points'])

# Add 5x5 data manually from what we know
unt_5x5 = untranspiled_data.get('5x5', {})
results.append({
    'mesh': '5x5',
    'nx': 5,
    'ny': 5,
    'grid_points': 25,
    'matrix_size_hermitian': 32,  # Will pad to 32
    'condition_number': 460,  # From metadata file
    'circuit_qubits': unt_5x5.get('qubits'),
    'circuit_depth': unt_5x5.get('depth'),
    'circuit_gates': unt_5x5.get('gates'),
    'time_construction_sec': 1855.33  # From circuit pkl file
})

mesh_labels = [r['mesh'] for r in results]
grid_points = [r['grid_points'] for r in results]

# Create figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('HHL Circuit Scaling: Untranspiled Metrics (2×2 through 5×5)', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Hermitian Matrix Size
ax = axes[0, 0]
matrix_sizes = [r['matrix_size_hermitian'] for r in results if r['matrix_size_hermitian']]
x_vals = grid_points[:len(matrix_sizes)]
ax.plot(x_vals, matrix_sizes, 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold', fontsize=11)
ax.set_ylabel('Hermitian Matrix Size', fontweight='bold', fontsize=11)
ax.set_title('Matrix Size After Hermitian Transform', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(x_vals)
ax.set_xticklabels([mesh_labels[i] for i in range(len(matrix_sizes))], fontsize=10)
for i, (x, y) in enumerate(zip(x_vals, matrix_sizes)):
    ax.annotate(f'{y}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

# Plot 2: Condition Number
ax = axes[0, 1]
cond_nums = [r['condition_number'] for r in results if r['condition_number']]
x_vals = grid_points[:len(cond_nums)]
ax.semilogy(x_vals, cond_nums, 's-', linewidth=2.5, markersize=10, color='#A23B72')
ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold', fontsize=11)
ax.set_ylabel('Condition Number (κ)', fontweight='bold', fontsize=11)
ax.set_title('Matrix Conditioning', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(x_vals)
ax.set_xticklabels([mesh_labels[i] for i in range(len(cond_nums))], fontsize=10)
for i, (x, y) in enumerate(zip(x_vals, cond_nums)):
    ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

# Plot 3: Circuit Qubits (actual from QPY)
ax = axes[0, 2]
qubits = [r['circuit_qubits'] for r in results if r['circuit_qubits']]
x_vals = grid_points[:len(qubits)]
ax.plot(x_vals, qubits, '^-', linewidth=2.5, markersize=10, color='#F18F01')
ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold', fontsize=11)
ax.set_ylabel('Number of Qubits', fontweight='bold', fontsize=11)
ax.set_title('Circuit Qubit Count (Untranspiled)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(x_vals)
ax.set_xticklabels([mesh_labels[i] for i in range(len(qubits))], fontsize=10)
for i, (x, y) in enumerate(zip(x_vals, qubits)):
    ax.annotate(f'{y}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

# Plot 4: File Size (proxy for gates/depth)
ax = axes[1, 0]
file_sizes_mb = [0.026, 144, 572, 4700]  # MB
x_vals = grid_points[:len(file_sizes_mb)]
ax.semilogy(x_vals, file_sizes_mb, 'd-', linewidth=2.5, markersize=10, color='#C73E1D')
ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold', fontsize=11)
ax.set_ylabel('QPY File Size (MB)', fontweight='bold', fontsize=11)
ax.set_title('Untranspiled Circuit File Size', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(x_vals)
ax.set_xticklabels([mesh_labels[i] for i in range(len(file_sizes_mb))], fontsize=10)
for i, (x, y) in enumerate(zip(x_vals, file_sizes_mb)):
    if y < 1:
        label = f'{y*1000:.0f}KB'
    elif y < 1000:
        label = f'{y:.0f}MB'
    else:
        label = f'{y/1000:.1f}GB'
    ax.annotate(label, xy=(x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

# Plot 5: Construction Time
ax = axes[1, 1]
times = [r['time_construction_sec'] for r in results if r['time_construction_sec']]
x_vals = grid_points[:len(times)]
ax.semilogy(x_vals, times, 'p-', linewidth=2.5, markersize=10, color='#06A77D')
ax.set_xlabel('Grid Points (nx × ny)', fontweight='bold', fontsize=11)
ax.set_ylabel('Construction Time (seconds)', fontweight='bold', fontsize=11)
ax.set_title('Circuit Construction Time', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(x_vals)
ax.set_xticklabels([mesh_labels[i] for i in range(len(times))], fontsize=10)
for i, (x, y) in enumerate(zip(x_vals, times)):
    if y < 1:
        label = f'{y:.2f}s'
    elif y < 60:
        label = f'{y:.1f}s'
    else:
        label = f'{y/60:.1f}m'
    ax.annotate(label, xy=(x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

# Plot 6: Summary Table
ax = axes[1, 2]
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Mesh', 'Matrix', 'κ', 'Qubits', 'File Size', 'Time'])
for i, r in enumerate(results):
    mesh = r['mesh']
    mat = f"{r['matrix_size_hermitian']}" if r['matrix_size_hermitian'] else 'N/A'
    kappa = f"{r['condition_number']:.0f}" if r['condition_number'] else 'N/A'
    qubits = f"{r['circuit_qubits']}" if r['circuit_qubits'] else 'N/A'
    
    if i < len(file_sizes_mb):
        if file_sizes_mb[i] < 1:
            fsize = f"{file_sizes_mb[i]*1000:.0f}KB"
        elif file_sizes_mb[i] < 1000:
            fsize = f"{file_sizes_mb[i]:.0f}MB"
        else:
            fsize = f"{file_sizes_mb[i]/1000:.1f}GB"
    else:
        fsize = 'N/A'
    
    if r['time_construction_sec']:
        if r['time_construction_sec'] < 1:
            ttime = f"{r['time_construction_sec']:.2f}s"
        elif r['time_construction_sec'] < 60:
            ttime = f"{r['time_construction_sec']:.1f}s"
        else:
            ttime = f"{r['time_construction_sec']/60:.1f}m"
    else:
        ttime = 'N/A'
    
    table_data.append([mesh, mat, kappa, qubits, fsize, ttime])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#E8E8E8')
    cell.set_text_props(weight='bold')

ax.set_title('Summary Table', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
output_path = workflow_dir / "untranspiled_scaling.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved plot to: {output_path}")
