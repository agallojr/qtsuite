#!/usr/bin/env python3
"""
Light cone analysis for HHL circuits.
Shows gate activity over circuit depth to identify error mitigation priorities.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.converters import circuit_to_dag
from pathlib import Path


def analyze_light_cone(circ_path, n_bins=100):
    """
    Analyze gate activity across circuit depth.
    
    Parameters
    ----------
    circ_path : Path
        Path to QPY circuit file
    n_bins : int
        Number of depth bins for visualization
        
    Returns
    -------
    dict
        Analysis results including activity matrices
    """
    print(f'Loading circuit from {circ_path}...')
    with open(circ_path, 'rb') as f:
        circ = qpy.load(f)[0]

    print(f'Circuit: {circ.num_qubits} qubits, {circ.depth()} depth, {circ.size()} gates')

    # Get active qubits
    active_qubits = set()
    for inst in circ.data:
        for q in inst.qubits:
            active_qubits.add(circ.find_bit(q).index)
    active_qubits = sorted(active_qubits)
    print(f'Active qubits: {len(active_qubits)} - {active_qubits}')

    # Build qubit × layer activity matrix
    depth = circ.depth()
    bin_size = max(1, depth // n_bins)

    n_active = len(active_qubits)
    qubit_map = {q: i for i, q in enumerate(active_qubits)}

    # Track gate activity and 2Q gate activity per bin
    activity = np.zeros((n_active, n_bins))
    two_q_activity = np.zeros((n_active, n_bins))

    # Analyze gate layers
    dag = circuit_to_dag(circ)

    print('Analyzing gate layers...')
    for layer_idx, layer in enumerate(dag.layers()):
        bin_idx = min(layer_idx // bin_size, n_bins - 1)
        
        for node in layer['graph'].op_nodes():
            qubits = [circ.find_bit(q).index for q in node.qargs]
            is_2q = len(qubits) >= 2
            
            for q in qubits:
                if q in qubit_map:
                    activity[qubit_map[q], bin_idx] += 1
                    if is_2q:
                        two_q_activity[qubit_map[q], bin_idx] += 1

    # Compute cumulative 2Q gates (error accumulation proxy)
    cumulative_2q = np.cumsum(two_q_activity, axis=1)

    # Compute error mitigation priority
    # Later gates matter more (less time for error to decohere)
    depth_weight = np.linspace(0.5, 1.5, n_bins)
    priority = two_q_activity * depth_weight

    return {
        'circuit': circ,
        'active_qubits': active_qubits,
        'n_active': n_active,
        'depth': depth,
        'n_bins': n_bins,
        'activity': activity,
        'two_q_activity': two_q_activity,
        'cumulative_2q': cumulative_2q,
        'priority': priority,
    }


def plot_light_cone(analysis, output_path=None):
    """
    Create light cone visualization.
    
    Parameters
    ----------
    analysis : dict
        Output from analyze_light_cone()
    output_path : Path, optional
        Path to save the figure
    """
    circ = analysis['circuit']
    active_qubits = analysis['active_qubits']
    n_active = analysis['n_active']
    activity = analysis['activity']
    two_q_activity = analysis['two_q_activity']
    cumulative_2q = analysis['cumulative_2q']
    priority = analysis['priority']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: All gate activity
    ax1 = axes[0, 0]
    im1 = ax1.imshow(activity, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Circuit Depth (binned)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Qubit', fontsize=11, fontweight='bold')
    ax1.set_yticks(range(n_active))
    ax1.set_yticklabels(active_qubits)
    ax1.set_title('All Gate Activity', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Gates per bin')

    # Panel 2: 2Q gate activity (error-prone)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(two_q_activity, aspect='auto', cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Circuit Depth (binned)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Qubit', fontsize=11, fontweight='bold')
    ax2.set_yticks(range(n_active))
    ax2.set_yticklabels(active_qubits)
    ax2.set_title('2-Qubit Gate Activity (Error Hotspots)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='2Q gates per bin')

    # Panel 3: Cumulative 2Q gates (error accumulation)
    ax3 = axes[1, 0]
    im3 = ax3.imshow(cumulative_2q, aspect='auto', cmap='inferno', interpolation='nearest')
    ax3.set_xlabel('Circuit Depth (binned)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Qubit', fontsize=11, fontweight='bold')
    ax3.set_yticks(range(n_active))
    ax3.set_yticklabels(active_qubits)
    ax3.set_title('Cumulative 2Q Gates (Error Accumulation)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Cumulative 2Q gates')

    # Panel 4: Error mitigation priority
    ax4 = axes[1, 1]
    im4 = ax4.imshow(priority, aspect='auto', cmap='plasma', interpolation='nearest')
    ax4.set_xlabel('Circuit Depth (binned)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Qubit', fontsize=11, fontweight='bold')
    ax4.set_yticks(range(n_active))
    ax4.set_yticklabels(active_qubits)
    ax4.set_title('Error Mitigation Priority\n(2Q density × depth weight)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Priority score')

    plt.suptitle(f'HHL Circuit Light Cone Analysis\n'
                 f'{circ.size()} gates, {analysis["depth"]} depth, {n_active} active qubits',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f'Saved to {output_path}')
    
    return fig


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    
    # Default: analyze 2x2 Hele-Shaw transpiled for Torino
    circ_path = base_dir / 'df2add62/hs_torino_2x2_0/hele-shaw_circ_nqmatrix2_transpiled.qpy'
    
    analysis = analyze_light_cone(circ_path)
    plot_light_cone(analysis, output_path=base_dir / 'light_cone_analysis.png')
    
    # Print summary
    print('\n--- Error Mitigation Summary ---')
    total_2q = analysis['cumulative_2q'][:, -1]
    for i, q in enumerate(analysis['active_qubits']):
        print(f'Qubit {q}: {int(total_2q[i])} total 2Q gates')
    
    # Identify hotspots
    priority = analysis['priority']
    hotspot_threshold = np.percentile(priority, 90)
    hotspots = np.argwhere(priority > hotspot_threshold)
    print(f'\nTop 10% priority hotspots (qubit, depth_bin):')
    for q_idx, d_idx in hotspots[:10]:
        print(f'  Qubit {analysis["active_qubits"][q_idx]}, bin {d_idx}')


if __name__ == '__main__':
    main()
