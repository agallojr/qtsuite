#!/usr/bin/env python3
"""
Error mitigation focus map for HHL circuits.
Single visualization showing where to apply ZNE/PEC techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.converters import circuit_to_dag
from pathlib import Path


def create_focus_map(circ_path, output_path=None, n_bins=100):
    """
    Create error mitigation focus map.
    
    Parameters
    ----------
    circ_path : Path
        Path to QPY circuit file
    output_path : Path, optional
        Path to save the figure
    n_bins : int
        Number of depth bins for visualization
    """
    # Load circuit
    with open(circ_path, 'rb') as f:
        circ = qpy.load(f)[0]

    print(f'Circuit: {circ.num_qubits} qubits, {circ.depth()} depth, {circ.size()} gates')

    # Get active qubits
    active_qubits = sorted(set(circ.find_bit(q).index for inst in circ.data for q in inst.qubits))
    n_active = len(active_qubits)
    qubit_map = {q: i for i, q in enumerate(active_qubits)}

    # Analyze by layer
    dag = circuit_to_dag(circ)
    depth = circ.depth()
    bin_size = max(1, depth // n_bins)

    two_q_activity = np.zeros((n_active, n_bins))

    for layer_idx, layer in enumerate(dag.layers()):
        bin_idx = min(layer_idx // bin_size, n_bins - 1)
        for node in layer['graph'].op_nodes():
            qubits = [circ.find_bit(q).index for q in node.qargs]
            if len(qubits) >= 2:
                for q in qubits:
                    if q in qubit_map:
                        two_q_activity[qubit_map[q], bin_idx] += 1

    # Create focus map
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute priority: 2Q density weighted by depth (later = more important)
    # Also weight by cumulative error exposure
    depth_weight = np.linspace(0.3, 1.0, n_bins)
    cumulative_2q = np.cumsum(two_q_activity, axis=1)
    max_cumulative = cumulative_2q[:, -1:] + 1  # avoid div by 0

    # Priority = current 2Q activity × depth weight × (1 + cumulative exposure)
    # This highlights late-circuit regions on high-error qubits
    priority = two_q_activity * depth_weight * (1 + cumulative_2q / max_cumulative)

    # Normalize per qubit for fair comparison
    priority_norm = priority / (priority.max(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(priority_norm, aspect='auto', cmap='magma', interpolation='bilinear',
                   vmin=0, vmax=1)

    ax.set_xlabel('Circuit Depth →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Qubit', fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_active))
    ax.set_yticklabels(active_qubits, fontsize=11)

    # Add phase annotations (approximate for HHL)
    phases = [
        (0, 15, 'State\nPrep'),
        (15, 45, 'QPE'),
        (45, 55, 'Rotation'),
        (55, 85, 'Inverse QPE'),
        (85, 100, 'Measure'),
    ]
    for start, end, label in phases:
        mid = (start + end) / 2
        ax.axvline(x=start, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(mid, -0.8, label, ha='center', va='bottom', fontsize=9, 
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Add qubit role annotations
    qubit_roles = ['Ancilla', 'Clock 0', 'Clock 1', 'State 0', 'State 1', 'Clock 2', 'Flag']
    for i, role in enumerate(qubit_roles[:n_active]):
        ax.text(n_bins + 1, i, role, ha='left', va='center', fontsize=9, style='italic')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label('Error Mitigation Priority (normalized)', fontsize=10)

    ax.set_title('HHL Error Mitigation Focus Map\n'
                 'Where to Apply ZNE/PEC',
                 fontsize=14, fontweight='bold')

    # Add summary stats
    total_2q = cumulative_2q[:, -1]
    top_qubit_idx = np.argmax(total_2q)
    top_qubit = active_qubits[top_qubit_idx]
    ax.text(0.02, 0.98, f'Highest error exposure: Qubit {top_qubit} ({int(max(total_2q))} 2Q gates)\n'
            f'Focus: Late-circuit (bins 60-100) on qubits with bright spots',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f'Saved to {output_path}')
    
    return fig, {
        'active_qubits': active_qubits,
        'total_2q_per_qubit': total_2q,
        'priority': priority,
        'top_qubit': top_qubit,
    }


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    
    # Default: analyze 2x2 Hele-Shaw transpiled for Torino
    circ_path = base_dir / 'df2add62/hs_torino_2x2_0/hele-shaw_circ_nqmatrix2_transpiled.qpy'
    
    fig, analysis = create_focus_map(circ_path, output_path=base_dir / 'error_mitigation_focus.png')
    
    # Print summary
    print('\n--- Error Mitigation Summary ---')
    for i, q in enumerate(analysis['active_qubits']):
        print(f'Qubit {q}: {int(analysis["total_2q_per_qubit"][i])} total 2Q gates')
    print(f'\nHighest priority: Qubit {analysis["top_qubit"]}')


if __name__ == '__main__':
    main()
