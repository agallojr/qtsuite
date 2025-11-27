#!/usr/bin/env python3
"""
Gate density heat map for HHL circuits.
Shows how gate density varies with circuit parameters.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    
    # Collect circuit data from various workflows
    circuit_data = []
    
    # Scaling study (0bbb27f2) - different grid sizes
    for ckpt in sorted((base_dir / '0bbb27f2').glob('checkpoint_*.json')):
        with open(ckpt) as f:
            data = json.load(f)
        params = data['params']
        nx = params.get('nx', 0)
        ny = params.get('ny', 0)
        depth = params.get('_circuit_depth_transpiled', params.get('_circuit_depth', 0))
        gates = params.get('_circuit_gates_transpiled', params.get('_circuit_gates', 0))
        qubits = params.get('_circuit_qubits', 0)
        kappa = params.get('_condition_number', 0)
        
        # Estimate qubits if not stored: log2(nx*ny) for state + ancillas
        if not qubits:
            n = nx * ny
            qubits = int(np.ceil(np.log2(n))) + 5  # rough estimate
        
        if depth > 0 and gates > 0:
            density = gates / (qubits * depth) if qubits > 0 else 0
            circuit_data.append({
                'nx': nx, 'ny': ny, 'grid': f'{nx}x{ny}',
                'qubits': qubits, 'depth': depth, 'gates': gates,
                'density': density, 'kappa': kappa,
                'source': 'Scaling Study'
            })
    
    # Real hardware runs (different shot counts, same circuit)
    for wf_id in ['df2add62', '058cf139', 'ea21306c', '8097a5fe']:
        wf_dir = base_dir / wf_id
        for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
            with open(ckpt) as f:
                data = json.load(f)
            params = data['params']
            nx = params.get('nx', 2)
            ny = params.get('ny', 2)
            depth = params.get('_circuit_depth_transpiled', params.get('_circuit_depth', 0))
            gates = params.get('_circuit_gates_transpiled', params.get('_circuit_gates', 0))
            qubits = params.get('_circuit_qubits', 7)
            shots = params.get('qc_shots', 0)
            
            fidelity = 0
            if data.get('classical_solution') and data.get('quantum_solution'):
                c = np.array(data['classical_solution'])
                q = np.array(data['quantum_solution'])
                cn, qn = np.linalg.norm(c), np.linalg.norm(q)
                if cn > 0 and qn > 0:
                    fidelity = np.abs(np.dot(c, q)) / (cn * qn)
            
            if depth > 0 and gates > 0:
                density = gates / (qubits * depth) if qubits > 0 else 0
                circuit_data.append({
                    'nx': nx, 'ny': ny, 'grid': f'{nx}x{ny}',
                    'qubits': qubits, 'depth': depth, 'gates': gates,
                    'density': density, 'shots': shots, 'fidelity': fidelity,
                    'source': 'Real Torino'
                })
                break  # Only need one per workflow (same circuit)
    
    # Print summary
    print("Circuit Data Summary:")
    print("=" * 80)
    for d in circuit_data:
        print(f"{d['grid']:>5} | qubits={d['qubits']:>3} | depth={d['depth']:>10,} | "
              f"gates={d['gates']:>12,} | density={d['density']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Gate density vs grid size (bar chart)
    ax1 = axes[0, 0]
    scaling_data = [d for d in circuit_data if d['source'] == 'Scaling Study']
    scaling_data.sort(key=lambda x: x['nx'])
    
    grids = [d['grid'] for d in scaling_data]
    densities = [d['density'] for d in scaling_data]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grids)))
    
    bars = ax1.bar(grids, densities, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gate Density (gates / qubit / depth)', fontsize=12, fontweight='bold')
    ax1.set_title('Gate Density vs Grid Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{density:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 2: Qubits vs Depth (bubble size = gates)
    ax2 = axes[0, 1]
    qubits = [d['qubits'] for d in scaling_data]
    depths = [d['depth'] for d in scaling_data]
    gates = [d['gates'] for d in scaling_data]
    
    # Normalize bubble sizes
    max_gates = max(gates)
    sizes = [500 * (g / max_gates) + 50 for g in gates]
    
    scatter = ax2.scatter(qubits, depths, s=sizes, c=densities, cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    
    for i, (q, dep, grid) in enumerate(zip(qubits, depths, grids)):
        ax2.annotate(grid, (q, dep), xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Qubits', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax2.set_title('Circuit Complexity (bubble size = gate count)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Gate Density', fontsize=10)
    
    # Panel 3: Heat map - Grid size vs metric
    ax3 = axes[1, 0]
    
    # Create matrix for heat map
    metrics = ['Qubits', 'log₁₀(Depth)', 'log₁₀(Gates)', 'Density']
    grid_labels = [d['grid'] for d in scaling_data]
    
    heat_data = np.array([
        [d['qubits'] for d in scaling_data],
        [np.log10(d['depth']) for d in scaling_data],
        [np.log10(d['gates']) for d in scaling_data],
        [d['density'] * 10 for d in scaling_data]  # Scale for visibility
    ])
    
    # Normalize each row
    heat_normalized = (heat_data - heat_data.min(axis=1, keepdims=True)) / \
                      (heat_data.max(axis=1, keepdims=True) - heat_data.min(axis=1, keepdims=True) + 1e-10)
    
    im = ax3.imshow(heat_normalized, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(grid_labels)))
    ax3.set_xticklabels(grid_labels, fontsize=11)
    ax3.set_yticks(range(len(metrics)))
    ax3.set_yticklabels(metrics, fontsize=11)
    ax3.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax3.set_title('Normalized Circuit Metrics Heat Map', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(grid_labels)):
            val = heat_data[i, j]
            if i == 0:  # Qubits
                text = f'{int(val)}'
            elif i == 3:  # Density (scaled)
                text = f'{val/10:.3f}'
            else:  # Log values
                text = f'{val:.1f}'
            ax3.text(j, i, text, ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if heat_normalized[i, j] > 0.5 else 'black')
    
    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for d in scaling_data:
        table_data.append([
            d['grid'],
            f"{d['qubits']}",
            f"{d['depth']:,}",
            f"{d['gates']:,}",
            f"{d['density']:.4f}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Grid', 'Qubits', 'Depth', 'Gates', 'Density'],
        loc='center',
        cellLoc='center',
        colColours=['#e6e6e6'] * 5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    
    ax4.text(0.5, 0.9, 'HHL Circuit Metrics by Grid Size', fontsize=14, fontweight='bold',
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.1, 'Gate Density = Gates / (Qubits × Depth)\n'
             'Lower density → more parallelism potential',
             fontsize=10, ha='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('HHL Gate Density Analysis: Hele-Shaw Flow', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = base_dir / 'gate_density_heatmap.png'
    plt.savefig(output_path, dpi=150)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
