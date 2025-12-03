#!/usr/bin/env python3
"""
Circuit parallelism analysis for HHL circuits.
Shows gates per layer (degeneracy) and efficiency metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.converters import circuit_to_dag
from pathlib import Path


def analyze_parallelism(circ_path, output_path=None):
    """
    Analyze circuit parallelism and create visualization.
    
    Parameters
    ----------
    circ_path : Path
        Path to QPY circuit file
    output_path : Path, optional
        Path to save the figure
    """
    # Load circuit
    with open(circ_path, 'rb') as f:
        circ = qpy.load(f)[0]

    print(f'Circuit: {circ.num_qubits} qubits, {circ.depth()} depth, {circ.size()} gates')
    print(f'Average gates per layer: {circ.size() / circ.depth():.2f}')

    # Count gates per layer
    dag = circuit_to_dag(circ)
    gates_per_layer = []
    two_q_per_layer = []

    for layer in dag.layers():
        n_gates = 0
        n_2q = 0
        for node in layer['graph'].op_nodes():
            n_gates += 1
            if len(node.qargs) >= 2:
                n_2q += 1
        gates_per_layer.append(n_gates)
        two_q_per_layer.append(n_2q)

    gates_per_layer = np.array(gates_per_layer)
    two_q_per_layer = np.array(two_q_per_layer)

    # Get active qubits count
    active_qubits = set()
    for inst in circ.data:
        for q in inst.qubits:
            active_qubits.add(circ.find_bit(q).index)
    n_active = len(active_qubits)

    print(f'Max parallelism: {max(gates_per_layer)} gates in one layer')
    print(f'Layers with 1 gate: {np.sum(gates_per_layer == 1)} ({100*np.sum(gates_per_layer == 1)/len(gates_per_layer):.1f}%)')
    print(f'Layers with 2+ gates: {np.sum(gates_per_layer >= 2)} ({100*np.sum(gates_per_layer >= 2)/len(gates_per_layer):.1f}%)')

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Gates per layer over circuit depth
    ax1 = axes[0, 0]
    ax1.fill_between(range(len(gates_per_layer)), gates_per_layer, alpha=0.7, color='steelblue')
    ax1.plot(range(len(gates_per_layer)), gates_per_layer, color='darkblue', linewidth=0.5)
    ax1.set_xlabel('Circuit Layer', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Gates in Layer', fontsize=11, fontweight='bold')
    ax1.set_title('Parallelism: Gates per Layer', fontsize=12, fontweight='bold')
    ax1.axhline(y=np.mean(gates_per_layer), color='red', linestyle='--', label=f'Mean: {np.mean(gates_per_layer):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: 2Q gates per layer
    ax2 = axes[0, 1]
    ax2.fill_between(range(len(two_q_per_layer)), two_q_per_layer, alpha=0.7, color='coral')
    ax2.plot(range(len(two_q_per_layer)), two_q_per_layer, color='darkred', linewidth=0.5)
    ax2.set_xlabel('Circuit Layer', fontsize=11, fontweight='bold')
    ax2.set_ylabel('2Q Gates in Layer', fontsize=11, fontweight='bold')
    ax2.set_title('Error Hotspots: 2Q Gates per Layer', fontsize=12, fontweight='bold')
    ax2.axhline(y=np.mean(two_q_per_layer), color='red', linestyle='--', label=f'Mean: {np.mean(two_q_per_layer):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Histogram of gates per layer
    ax3 = axes[1, 0]
    max_gates = int(max(gates_per_layer))
    bins = range(0, max_gates + 2)
    ax3.hist(gates_per_layer, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Gates per Layer', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Layers', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Layer Parallelism', fontsize=12, fontweight='bold')
    ax3.axvline(x=np.mean(gates_per_layer), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gates_per_layer):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Cumulative gates vs depth (shows efficiency)
    ax4 = axes[1, 1]
    cumulative_gates = np.cumsum(gates_per_layer)
    cumulative_2q = np.cumsum(two_q_per_layer)
    layers = np.arange(len(gates_per_layer))

    ax4.plot(layers, cumulative_gates, color='steelblue', linewidth=2, label='All gates')
    ax4.plot(layers, cumulative_2q, color='coral', linewidth=2, label='2Q gates')
    ax4.plot([0, len(layers)], [0, circ.size()], 'k--', alpha=0.5, label='Perfect parallelism')
    ax4.fill_between(layers, cumulative_gates, alpha=0.3, color='steelblue')
    ax4.set_xlabel('Circuit Layer', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Gates', fontsize=11, fontweight='bold')
    ax4.set_title('Gate Accumulation vs Depth', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add efficiency metric
    efficiency = circ.size() / (circ.depth() * n_active)
    ax4.text(0.98, 0.02, f'Efficiency: {efficiency:.1%}\n(gates / depth / qubits)',
             transform=ax4.transAxes, ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    degeneracy = circ.size() / circ.depth()
    plt.suptitle(f'Circuit Parallelism Analysis\n'
                 f'{circ.size()} gates, {circ.depth()} depth, degeneracy = {degeneracy:.2f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f'\nSaved to {output_path}')

    return fig, {
        'gates_per_layer': gates_per_layer,
        'two_q_per_layer': two_q_per_layer,
        'degeneracy': degeneracy,
        'efficiency': efficiency,
    }


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    
    # Default: analyze 2x2 Hele-Shaw transpiled for Torino
    circ_path = base_dir / 'df2add62/hs_torino_2x2_0/hele-shaw_circ_nqmatrix2_transpiled.qpy'
    
    fig, analysis = analyze_parallelism(circ_path, output_path=base_dir / 'circuit_parallelism.png')
    
    print(f'\nDegeneracy: {analysis["degeneracy"]:.2f}')
    print(f'Efficiency: {analysis["efficiency"]:.1%}')


if __name__ == '__main__':
    main()
