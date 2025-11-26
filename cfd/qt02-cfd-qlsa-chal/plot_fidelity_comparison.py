#!/usr/bin/env python3
"""Pretty fidelity comparison diagram: Tridiagonal vs Hele-Shaw with circuit metrics"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Workflow directories
# Tridiag fidelity: ccdbece7 (MPS simulator with shot noise)
# Tridiag circuit metrics: 320a98b0 (original, for right panel)
tridiag_fidelity_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/ccdbece7")
tridiag_metrics_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/320a98b0")
tridiag_nq4_metrics_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/3c1b8abb")
heleshaw_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/f2df9b40")
heleshaw_scaling_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/0bbb27f2")

# Load tridiagonal checkpoint data (fidelity from MPS run)
tridiag_data = []
for ckpt_file in sorted(tridiag_fidelity_dir.glob("checkpoint_*.json")):
    with open(ckpt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Calculate fidelity
        q_sol = np.array(data['quantum_solution'])
        c_sol = np.array(data['classical_solution'])
        fidelity = np.abs(np.dot(q_sol, c_sol) / (np.linalg.norm(q_sol) * np.linalg.norm(c_sol)))
        
        tridiag_data.append({
            'nq_matrix': data['params']['NQ_MATRIX'],
            'shots': data['params']['qc_shots'],
            'kappa': data['params']['_matrix_condition_number'],
            'fidelity': fidelity,
            'qubits': data['params'].get('_circuit_qubits'),
            'depth_transpiled': data['params'].get('_circuit_depth_transpiled'),
            'gates_transpiled': data['params'].get('_circuit_gates_transpiled'),
        })

# Load tridiagonal circuit metrics from original runs (for right panel)
tridiag_metrics = []
for ckpt_file in sorted(tridiag_metrics_dir.glob("checkpoint_*.json")):
    with open(ckpt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        tridiag_metrics.append({
            'nq_matrix': data['params']['NQ_MATRIX'],
            'qubits': data['params'].get('_circuit_qubits'),
            'depth_transpiled': data['params'].get('_circuit_depth_transpiled'),
        })
for ckpt_file in sorted(tridiag_nq4_metrics_dir.glob("checkpoint_*.json")):
    with open(ckpt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        tridiag_metrics.append({
            'nq_matrix': data['params']['NQ_MATRIX'],
            'qubits': data['params'].get('_circuit_qubits'),
            'depth_transpiled': data['params'].get('_circuit_depth_transpiled'),
        })

# Load Hele-Shaw reconstructed data (for fidelity)
with open(heleshaw_dir / "results_reconstructed.json", 'r') as f:
    heleshaw_data_raw = json.load(f)

heleshaw_data = []
for item in heleshaw_data_raw:
    heleshaw_data.append({
        'nx': item['input_parameters']['nx'],
        'ny': item['input_parameters']['ny'],
        'shots': item['input_parameters']['qc_shots'],
        'kappa': item['matrix_properties']['condition_number_original'],
        'fidelity': item['results']['fidelity']
    })

# Load Hele-Shaw scaling data (for circuit metrics from 0bbb27f2)
hs_circuit_data = {}
for grid in ['2x2', '3x3', '4x4']:
    ckpt_file = heleshaw_scaling_dir / f"checkpoint_scaling_{grid}.json"
    if ckpt_file.exists():
        with open(ckpt_file, 'r') as f:
            data = json.load(f)
            hs_circuit_data[grid] = {
                'depth_transpiled': data['params'].get('_circuit_depth_transpiled'),
                'gates_transpiled': data['params'].get('_circuit_gates_transpiled'),
            }

# Qubit counts from transpiled QPY files (pre-computed)
hs_circuit_data['2x2']['qubits'] = 7
hs_circuit_data['3x3']['qubits'] = 13
hs_circuit_data['4x4']['qubits'] = 15

# Organize data
shot_counts = [100, 1000, 10000, 100000]

# Extract fidelity for each case
tri_nq2 = {s: next((d['fidelity'] for d in tridiag_data if d['nq_matrix']==2 and d['shots']==s), None) 
           for s in shot_counts}
tri_nq4 = {s: next((d['fidelity'] for d in tridiag_data if d['nq_matrix']==4 and d['shots']==s), None) 
           for s in shot_counts}
tri_nq5 = {s: next((d['fidelity'] for d in tridiag_data if d['nq_matrix']==5 and d['shots']==s), None) 
           for s in shot_counts}
hs_2x2 = {s: next((d['fidelity'] for d in heleshaw_data if d['nx']==2 and d['ny']==2 and d['shots']==s), None) 
          for s in shot_counts}
hs_3x3 = {s: next((d['fidelity'] for d in heleshaw_data if d['nx']==3 and d['ny']==3 and d['shots']==s), None) 
          for s in shot_counts}
hs_4x4 = {s: next((d['fidelity'] for d in heleshaw_data if d['nx']==4 and d['ny']==4 and d['shots']==s), None) 
          for s in shot_counts}

# Get kappa values
kappa_tri_nq2 = next((d['kappa'] for d in tridiag_data if d['nq_matrix']==2), 0)
kappa_tri_nq4 = next((d['kappa'] for d in tridiag_data if d['nq_matrix']==4), 0)
kappa_tri_nq5 = next((d['kappa'] for d in tridiag_data if d['nq_matrix']==5), 0)
kappa_hs_2x2 = next((d['kappa'] for d in heleshaw_data if d['nx']==2 and d['ny']==2), 0)
kappa_hs_3x3 = next((d['kappa'] for d in heleshaw_data if d['nx']==3 and d['ny']==3), 0)
kappa_hs_4x4 = next((d['kappa'] for d in heleshaw_data if d['nx']==4 and d['ny']==4), 0)

# Get circuit metrics (from original runs with transpiled circuits)
# Tridiag - get from tridiag_metrics (original runs have circuit depth info)
tri_nq2_circuit = next((d for d in tridiag_metrics if d['nq_matrix']==2), {})
tri_nq4_circuit = next((d for d in tridiag_metrics if d['nq_matrix']==4), {})
tri_nq5_circuit = next((d for d in tridiag_metrics if d['nq_matrix']==5), {})

# Color scheme
colors = {
    'tri_nq2': '#06A77D',
    'hs_2x2': '#2E8B57',
    'tri_nq4': '#5B8FF9',
    'tri_nq5': '#F18F01',
    'hs_3x3': '#E67E22',
    'hs_4x4': '#C73E1D'
}

markers = {
    'tri_nq2': 'o',
    'hs_2x2': 's',
    'tri_nq4': 'p',
    'tri_nq5': '^',
    'hs_3x3': 'D',
    'hs_4x4': 'v'
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

fig.suptitle('HHL Algorithm: Fidelity & Circuit Complexity Comparison', 
             fontsize=16, fontweight='bold')

# ============ LEFT PLOT: Fidelity vs Shots ============
fid_tri_nq2 = [tri_nq2[s] for s in shot_counts]
fid_hs_2x2 = [hs_2x2[s] for s in shot_counts]
fid_tri_nq4 = [tri_nq4[s] for s in shot_counts]
fid_tri_nq5 = [tri_nq5[s] for s in shot_counts]
fid_hs_3x3 = [hs_3x3[s] for s in shot_counts]
fid_hs_4x4 = [hs_4x4[s] for s in shot_counts]

ax1.semilogx(shot_counts, fid_tri_nq2, marker=markers['tri_nq2'], color=colors['tri_nq2'],
            linewidth=2.5, markersize=10, label=f'Tridiag NQ=2 (κ={kappa_tri_nq2:.1f})')
ax1.semilogx(shot_counts, fid_hs_2x2, marker=markers['hs_2x2'], color=colors['hs_2x2'],
            linewidth=2.5, markersize=10, label=f'Hele-Shaw 2×2 (κ={kappa_hs_2x2:.1f})')
ax1.semilogx(shot_counts, fid_tri_nq4, marker=markers['tri_nq4'], color=colors['tri_nq4'],
            linewidth=2.5, markersize=10, label=f'Tridiag NQ=4 (κ={kappa_tri_nq4:.1f})')
ax1.semilogx(shot_counts, fid_tri_nq5, marker=markers['tri_nq5'], color=colors['tri_nq5'],
            linewidth=2.5, markersize=10, label=f'Tridiag NQ=5 (κ={kappa_tri_nq5:.1f})')
ax1.semilogx(shot_counts, fid_hs_3x3, marker=markers['hs_3x3'], color=colors['hs_3x3'],
            linewidth=2.5, markersize=10, label=f'Hele-Shaw 3×3 (κ={kappa_hs_3x3:.0f})')
ax1.semilogx(shot_counts, fid_hs_4x4, marker=markers['hs_4x4'], color=colors['hs_4x4'],
            linewidth=2.5, markersize=10, label=f'Hele-Shaw 4×4 (κ={kappa_hs_4x4:.0f})')

ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.6, label='95% threshold')
ax1.axhline(y=0.80, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='80% threshold')

ax1.set_xlabel('Number of Shots', fontweight='bold', fontsize=12)
ax1.set_ylabel('Fidelity', fontweight='bold', fontsize=12)
ax1.set_ylim([0.35, 1.02])
ax1.set_title('Fidelity vs Shot Count', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='center right', fontsize=9, framealpha=0.95)

# ============ RIGHT PLOT: Circuit Depth & Qubits ============
cases = ['Tridiag\nNQ=2', 'Hele-Shaw\n2×2', 'Tridiag\nNQ=4', 'Tridiag\nNQ=5', 'Hele-Shaw\n3×3', 'Hele-Shaw\n4×4']
case_keys = ['tri_nq2', 'hs_2x2', 'tri_nq4', 'tri_nq5', 'hs_3x3', 'hs_4x4']

qubits = [
    tri_nq2_circuit.get('qubits', 0),
    hs_circuit_data['2x2']['qubits'],
    tri_nq4_circuit.get('qubits', 0),
    tri_nq5_circuit.get('qubits', 0),
    hs_circuit_data['3x3']['qubits'],
    hs_circuit_data['4x4']['qubits'],
]

depths = [
    tri_nq2_circuit.get('depth_transpiled', 0),
    hs_circuit_data['2x2']['depth_transpiled'],
    tri_nq4_circuit.get('depth_transpiled', 0),
    tri_nq5_circuit.get('depth_transpiled', 0),
    hs_circuit_data['3x3']['depth_transpiled'],
    hs_circuit_data['4x4']['depth_transpiled'],
]

x = np.arange(len(cases))
width = 0.35

# Left Y-axis: Qubits (bar chart)
bars = ax2.bar(x, qubits, width, color=[colors[k] for k in case_keys], alpha=0.8, label='Qubits')
ax2.set_ylabel('Number of Qubits', fontweight='bold', fontsize=12, color='#333')
ax2.set_ylim([0, max(qubits) * 1.2])


# Right Y-axis: Circuit Depth (line with markers)
ax2_right = ax2.twinx()
ax2_right.semilogy(x, depths, 'k-', marker='D', markersize=10, linewidth=2.5, 
                   markerfacecolor='white', markeredgewidth=2, label='Circuit Depth')
ax2_right.set_ylabel('Circuit Depth (transpiled, log scale)', fontweight='bold', fontsize=12)
ax2_right.set_ylim([100, max(depths) * 5])


ax2.set_xticks(x)
ax2.set_xticklabels(cases, fontsize=10)
ax2.set_title('Circuit Complexity: Qubits & Depth', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
output_path = tridiag_metrics_dir / "fidelity_comparison_pretty.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved comparison to: {output_path}")
