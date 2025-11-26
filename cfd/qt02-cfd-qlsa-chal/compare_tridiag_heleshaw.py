#!/usr/bin/env python3
"""Compare Tridiagonal vs Hele-Shaw HHL results"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Workflow directories
tridiag_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/320a98b0")
heleshaw_dir = Path("/Users/agallojr/.lwfm/out/qt02-cfd/f2df9b40")

# Load tridiagonal checkpoint data
tridiag_data = []
for ckpt_file in sorted(tridiag_dir.glob("checkpoint_*.json")):
    with open(ckpt_file, 'r') as f:
        data = json.load(f)
        # Calculate fidelity
        q_sol = np.array(data['quantum_solution'])
        c_sol = np.array(data['classical_solution'])
        fidelity = np.abs(np.dot(q_sol, c_sol) / (np.linalg.norm(q_sol) * np.linalg.norm(c_sol)))
        
        tridiag_data.append({
            'case_id': data['case_id'],
            'nq_matrix': data['params']['NQ_MATRIX'],
            'shots': data['params']['qc_shots'],
            'kappa': data['params']['_matrix_condition_number'],
            'qubits': data['params']['_circuit_qubits'],
            'depth': data['params']['_circuit_depth'],
            'gates': data['params']['_circuit_gates'],
            'construction_time': data['params']['_time_circuit_construction'],
            'execution_time': data['params']['_time_execution'],
            'fidelity': fidelity
        })

# Load Hele-Shaw reconstructed data
with open(heleshaw_dir / "results_reconstructed.json", 'r') as f:
    heleshaw_data_raw = json.load(f)

heleshaw_data = []
for item in heleshaw_data_raw:
    heleshaw_data.append({
        'case_id': item['case_id'],
        'nx': item['input_parameters']['nx'],
        'ny': item['input_parameters']['ny'],
        'shots': item['input_parameters']['qc_shots'],
        'kappa': item['matrix_properties']['condition_number_original'],
        'fidelity': item['results']['fidelity'],
        'construction_time': item['timing']['circuit_construction_sec']
    })

# Organize data by matrix size
tridiag_nq2 = [d for d in tridiag_data if d['nq_matrix'] == 2]
tridiag_nq5 = [d for d in tridiag_data if d['nq_matrix'] == 5]
hs_2x2 = [d for d in heleshaw_data if d['nx'] == 2 and d['ny'] == 2]
hs_3x3 = [d for d in heleshaw_data if d['nx'] == 3 and d['ny'] == 3]
hs_4x4 = [d for d in heleshaw_data if d['nx'] == 4 and d['ny'] == 4]

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Tridiagonal vs Hele-Shaw: HHL Performance Comparison', 
             fontsize=16, fontweight='bold')

# Plot 1: Fidelity vs Shots (NQ=2 / 2x2)
ax = axes[0, 0]
if tridiag_nq2 and hs_2x2:
    shots_tri = [d['shots'] for d in tridiag_nq2]
    fid_tri = [d['fidelity'] for d in tridiag_nq2]
    shots_hs = [d['shots'] for d in hs_2x2]
    fid_hs = [d['fidelity'] for d in hs_2x2]
    
    ax.semilogx(shots_tri, fid_tri, 'o-', linewidth=2, markersize=8, 
                label=f'Tridiagonal (κ={tridiag_nq2[0]["kappa"]:.1f})', color='#06A77D')
    ax.semilogx(shots_hs, fid_hs, 's-', linewidth=2, markersize=8, 
                label=f'Hele-Shaw 2×2 (κ={hs_2x2[0]["kappa"]:.1f})', color='#C73E1D')
    
    ax.set_xlabel('Number of Shots', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fidelity', fontweight='bold', fontsize=11)
    ax.set_title('4×4 Matrix (NQ=2)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])

# Plot 2: Fidelity vs Shots (NQ=5 / 3x3, 4x4)
ax = axes[0, 1]
if tridiag_nq5 and hs_3x3 and hs_4x4:
    shots_tri = [d['shots'] for d in tridiag_nq5]
    fid_tri = [d['fidelity'] for d in tridiag_nq5]
    shots_3x3 = [d['shots'] for d in hs_3x3]
    fid_3x3 = [d['fidelity'] for d in hs_3x3]
    shots_4x4 = [d['shots'] for d in hs_4x4]
    fid_4x4 = [d['fidelity'] for d in hs_4x4]
    
    ax.semilogx(shots_tri, fid_tri, 'o-', linewidth=2, markersize=8, 
                label=f'Tridiagonal (κ={tridiag_nq5[0]["kappa"]:.1f})', color='#06A77D')
    ax.semilogx(shots_3x3, fid_3x3, '^-', linewidth=2, markersize=8, 
                label=f'Hele-Shaw 3×3 (κ={hs_3x3[0]["kappa"]:.0f})', color='#F18F01')
    ax.semilogx(shots_4x4, fid_4x4, 's-', linewidth=2, markersize=8, 
                label=f'Hele-Shaw 4×4 (κ={hs_4x4[0]["kappa"]:.0f})', color='#C73E1D')
    
    ax.set_xlabel('Number of Shots', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fidelity', fontweight='bold', fontsize=11)
    ax.set_title('32×32 Matrix (NQ=5)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.05])

# Plot 3: Circuit Complexity Comparison
ax = axes[1, 0]
cases = ['Tridiag\nNQ=2', 'HS 2×2', 'Tridiag\nNQ=5', 'HS 3×3', 'HS 4×4']
qubits = [
    tridiag_nq2[0]['qubits'] if tridiag_nq2 else 0,
    7,  # From previous data
    tridiag_nq5[0]['qubits'] if tridiag_nq5 else 0,
    13,  # From previous data
    15   # From previous data
]
depths = [
    tridiag_nq2[0]['depth'] if tridiag_nq2 else 0,
    243,  # From previous data
    tridiag_nq5[0]['depth'] if tridiag_nq5 else 0,
    2156878,  # From previous data
    8592323   # From previous data
]

x = np.arange(len(cases))
width = 0.35

ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, qubits, width, label='Qubits', color='#F18F01', alpha=0.8)
bars2 = ax2.bar(x + width/2, depths, width, label='Depth', color='#06A77D', alpha=0.8)

ax.set_xlabel('Case', fontweight='bold', fontsize=11)
ax.set_ylabel('Number of Qubits', fontweight='bold', fontsize=11, color='#F18F01')
ax2.set_ylabel('Circuit Depth (log scale)', fontweight='bold', fontsize=11, color='#06A77D')
ax.set_title('Circuit Complexity', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=9)
ax.tick_params(axis='y', labelcolor='#F18F01')
ax2.tick_params(axis='y', labelcolor='#06A77D')
ax2.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Plot 4: Summary Table
ax = axes[1, 1]
ax.axis('off')

table_data = [
    ['Case', 'Matrix', 'κ', 'Qubits', 'Depth', 'Fidelity*'],
    ['Tridiag NQ=2', '4×4', f'{tridiag_nq2[0]["kappa"]:.1f}' if tridiag_nq2 else 'N/A',
     str(tridiag_nq2[0]['qubits']) if tridiag_nq2 else 'N/A',
     f'{tridiag_nq2[0]["depth"]/1e3:.1f}K' if tridiag_nq2 else 'N/A',
     f'{tridiag_nq2[-1]["fidelity"]:.3f}' if tridiag_nq2 else 'N/A'],
    ['HS 2×2', '4×4', '1', '7', '243', f'{hs_2x2[-1]["fidelity"]:.3f}' if hs_2x2 else 'N/A'],
    ['Tridiag NQ=5', '32×32', f'{tridiag_nq5[0]["kappa"]:.1f}' if tridiag_nq5 else 'N/A',
     str(tridiag_nq5[0]['qubits']) if tridiag_nq5 else 'N/A',
     f'{tridiag_nq5[0]["depth"]/1e6:.2f}M' if tridiag_nq5 else 'N/A',
     f'{tridiag_nq5[-1]["fidelity"]:.3f}' if tridiag_nq5 else 'N/A'],
    ['HS 3×3', '32×32', '43', '13', '2.2M', f'{hs_3x3[-1]["fidelity"]:.3f}' if hs_3x3 else 'N/A'],
    ['HS 4×4', '32×32', '160', '15', '8.6M', f'{hs_4x4[-1]["fidelity"]:.3f}' if hs_4x4 else 'N/A']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.12, 0.12, 0.12, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#E8E8E8')
    cell.set_text_props(weight='bold')

ax.set_title('Summary (Untranspiled)\n*Fidelity at max shots', 
             fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
output_path = tridiag_dir / "comparison_tridiag_vs_heleshaw.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved comparison plot to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print("COMPARISON SUMMARY: Tridiagonal vs Hele-Shaw")
print("="*60)

print("\n4×4 Matrix (NQ=2):")
if tridiag_nq2 and hs_2x2:
    print(f"  Tridiag: κ={tridiag_nq2[0]['kappa']:.1f}, qubits={tridiag_nq2[0]['qubits']}, "
          f"depth={tridiag_nq2[0]['depth']}, fidelity={tridiag_nq2[-1]['fidelity']:.4f}")
    print(f"  HS 2×2:  κ=1.0, qubits=7, depth=243, fidelity={hs_2x2[-1]['fidelity']:.4f}")

print("\n32×32 Matrix (NQ=5):")
if tridiag_nq5:
    print(f"  Tridiag: κ={tridiag_nq5[0]['kappa']:.1f}, qubits={tridiag_nq5[0]['qubits']}, "
          f"depth={tridiag_nq5[0]['depth']}, fidelity={tridiag_nq5[-1]['fidelity']:.4f}")
if hs_3x3:
    print(f"  HS 3×3:  κ=43, qubits=13, depth=2156878, fidelity={hs_3x3[-1]['fidelity']:.4f}")
if hs_4x4:
    print(f"  HS 4×4:  κ=160, qubits=15, depth=8592323, fidelity={hs_4x4[-1]['fidelity']:.4f}")

print("\nKey Findings:")
print("  - Tridiagonal has much lower κ (better conditioned)")
print("  - Tridiagonal achieves higher fidelity for 32×32 matrix")
print("  - Circuit complexity grows with problem structure, not just matrix size")
print("  - Hele-Shaw 3×3 and 4×4 have same matrix size but different fidelity")
print("="*60)
