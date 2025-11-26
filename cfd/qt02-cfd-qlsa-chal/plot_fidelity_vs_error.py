#!/usr/bin/env python3
"""
Plot fidelity vs solution error for HHL results.
Shows the relationship between fidelity metric and actual L2 error in the solution.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_fidelity(classical_sol, quantum_sol):
    """Calculate fidelity (cosine similarity) between solutions."""
    classical_norm = np.linalg.norm(classical_sol)
    quantum_norm = np.linalg.norm(quantum_sol)
    if classical_norm > 0 and quantum_norm > 0:
        return np.abs(np.dot(classical_sol, quantum_sol)) / (classical_norm * quantum_norm)
    return 0.0


def calculate_errors(classical_sol, quantum_sol):
    """Calculate relative L2 error between solutions."""
    c = np.array(classical_sol)
    q = np.array(quantum_sol)
    # Scale quantum to match classical magnitude
    q_scaled = q * (np.linalg.norm(c) / np.linalg.norm(q)) if np.linalg.norm(q) > 0 else q
    rel_l2 = np.linalg.norm(q_scaled - c) / np.linalg.norm(c) if np.linalg.norm(c) > 0 else 0
    return rel_l2


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    data_points = []

    # Real Torino (4 runs)
    for wf_id in ['df2add62', '058cf139', 'ea21306c', '8097a5fe']:
        wf_dir = base_dir / wf_id
        for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
            with open(ckpt) as f:
                data = json.load(f)
            c_sol = np.array(data['classical_solution'])
            q_sol = np.array(data['quantum_solution'])
            fidelity = calculate_fidelity(c_sol, q_sol)
            rel_l2 = calculate_errors(c_sol, q_sol)
            data_points.append({'fidelity': fidelity, 'rel_l2': rel_l2, 'source': 'Real Torino'})

    # Torino Aer (0e63f2d8)
    for ckpt in sorted((base_dir / '0e63f2d8').glob('checkpoint_*.json')):
        with open(ckpt) as f:
            data = json.load(f)
        c_sol = np.array(data['classical_solution'])
        q_sol = np.array(data['quantum_solution'])
        fidelity = calculate_fidelity(c_sol, q_sol)
        rel_l2 = calculate_errors(c_sol, q_sol)
        data_points.append({'fidelity': fidelity, 'rel_l2': rel_l2, 'source': 'Torino Aer'})

    # Heron sim (4f765788)
    for ckpt in sorted((base_dir / '4f765788').glob('checkpoint_*.json')):
        with open(ckpt) as f:
            data = json.load(f)
        c_sol = np.array(data['classical_solution'])
        q_sol = np.array(data['quantum_solution'])
        fidelity = calculate_fidelity(c_sol, q_sol)
        rel_l2 = calculate_errors(c_sol, q_sol)
        data_points.append({'fidelity': fidelity, 'rel_l2': rel_l2, 'source': 'Heron Sim'})

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'Real Torino': '#C73E1D', 'Torino Aer': '#5B8FF9', 'Heron Sim': '#2E8B57'}
    markers = {'Real Torino': '^', 'Torino Aer': 's', 'Heron Sim': 'o'}

    for source in ['Heron Sim', 'Torino Aer', 'Real Torino']:
        pts = [p for p in data_points if p['source'] == source]
        fids = [p['fidelity'] for p in pts]
        errs = [p['rel_l2'] for p in pts]
        ax.scatter(fids, errs, c=colors[source], marker=markers[source], s=80, 
                   alpha=0.7, label=source, edgecolors='white', linewidths=0.5)

    # Theoretical curve: rel_l2 = sqrt(2*(1-fidelity))
    fid_theory = np.linspace(0.7, 1.0, 100)
    err_theory = np.sqrt(2 * (1 - fid_theory))
    ax.plot(fid_theory, err_theory, 'k--', linewidth=2, alpha=0.5, 
            label=r'Theory: $\sqrt{2(1-F)}$')

    ax.set_xlabel('Fidelity (F)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative L2 Error (vs classical solution)', fontsize=12, fontweight='bold')
    ax.set_title('Fidelity vs Solution Error\nHele-Shaw 2×2, Ax=b', fontsize=14, fontweight='bold')
    ax.set_xlim([0.70, 1.02])
    ax.set_ylim([0, 0.8])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    output_path = base_dir / 'fidelity_vs_error.png'
    plt.savefig(output_path, dpi=150)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
