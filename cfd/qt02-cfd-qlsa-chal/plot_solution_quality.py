#!/usr/bin/env python3
"""
Solution quality analysis for HHL results.
Shows fidelity, residual error, and their relationship to shots.
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


def calculate_residual(A, b, quantum_sol, classical_sol):
    """Calculate ||Ax_q - b|| / ||b|| using scaled quantum solution."""
    q = np.array(quantum_sol)
    c = np.array(classical_sol)
    # Scale quantum to match classical magnitude
    q_scaled = q * (np.linalg.norm(c) / np.linalg.norm(q)) if np.linalg.norm(q) > 0 else q
    residual = np.linalg.norm(A @ q_scaled - b) / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 0
    return residual


def load_checkpoint_data(ckpt_path, source_name):
    """Load data from a checkpoint file."""
    with open(ckpt_path) as f:
        data = json.load(f)
    
    c_sol = np.array(data['classical_solution'])
    q_sol = np.array(data['quantum_solution'])
    shots = data['params']['qc_shots']
    depth = data['params'].get('_circuit_depth_transpiled', 0)
    
    fidelity = calculate_fidelity(c_sol, q_sol)
    
    # Try to get A matrix for residual calculation
    A = data.get('A_matrix') or data.get('matrix')
    b = data.get('b_vector') or data.get('vector')
    
    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b)
        residual = calculate_residual(A, b, q_sol, c_sol)
        kappa = np.linalg.cond(A)
    else:
        residual = None
        kappa = None
    
    return {
        'fidelity': fidelity,
        'residual': residual,
        'kappa': kappa,
        'depth': depth,
        'shots': shots,
        'source': source_name
    }


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    data_points = []

    # Real Torino (4 runs)
    for wf_id in ['df2add62', '058cf139', 'ea21306c', '8097a5fe']:
        wf_dir = base_dir / wf_id
        for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
            try:
                pt = load_checkpoint_data(ckpt, 'Real Torino')
                data_points.append(pt)
            except Exception as e:
                print(f"Warning: {ckpt}: {e}")

    # Torino Aer (0e63f2d8)
    for ckpt in sorted((base_dir / '0e63f2d8').glob('checkpoint_*.json')):
        try:
            pt = load_checkpoint_data(ckpt, 'Torino Aer')
            data_points.append(pt)
        except Exception as e:
            print(f"Warning: {ckpt}: {e}")

    # Heron sim (4f765788)
    for ckpt in sorted((base_dir / '4f765788').glob('checkpoint_*.json')):
        try:
            pt = load_checkpoint_data(ckpt, 'Heron Sim')
            data_points.append(pt)
        except Exception as e:
            print(f"Warning: {ckpt}: {e}")

    # Check if we have residual data
    has_residual = any(p['residual'] is not None for p in data_points)
    
    # Create figure
    if has_residual:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = np.array([[axes[0], axes[1]], [None, None]])

    colors = {'Real Torino': '#C73E1D', 'Torino Aer': '#5B8FF9', 'Heron Sim': '#2E8B57'}
    markers = {'Real Torino': '^', 'Torino Aer': 's', 'Heron Sim': 'o'}

    # Panel 1: Fidelity vs Shots
    ax1 = axes[0, 0]
    for source in ['Heron Sim', 'Torino Aer', 'Real Torino']:
        pts = [p for p in data_points if p['source'] == source]
        shots = [p['shots'] for p in pts]
        fids = [p['fidelity'] for p in pts]
        ax1.scatter(shots, fids, c=colors[source], marker=markers[source], s=80, 
                   alpha=0.7, label=source, edgecolors='white', linewidths=0.5)

    ax1.set_xscale('log')
    ax1.set_xlabel('Shots', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fidelity', fontsize=11, fontweight='bold')
    ax1.set_title('Fidelity vs Shots', fontsize=12, fontweight='bold')
    ax1.set_ylim([0.70, 1.02])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='lower right')

    # Panel 2: Summary stats
    ax2 = axes[0, 1]
    ax2.axis('off')

    summary_data = []
    for source in ['Heron Sim', 'Torino Aer', 'Real Torino']:
        pts = [p for p in data_points if p['source'] == source]
        fids = [p['fidelity'] for p in pts]
        resids = [p['residual'] for p in pts if p['residual'] is not None]
        row = [
            source,
            f'{np.mean(fids):.1%}',
            f'{np.std(fids):.4f}',
            f'{len(pts)}'
        ]
        if resids:
            row.insert(3, f'{np.mean(resids):.4f}')
        summary_data.append(row)

    col_labels = ['Source', 'Mean F', 'σ(F)', 'N']
    if has_residual:
        col_labels.insert(3, 'Mean Resid')

    table = ax2.table(
        cellText=summary_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(col_labels)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax2.text(0.5, 0.85, 'Summary Statistics', fontsize=14, fontweight='bold',
             ha='center', transform=ax2.transAxes)

    if has_residual:
        # Panel 3: Fidelity vs Residual Error
        ax3 = axes[1, 0]
        for source in ['Heron Sim', 'Torino Aer', 'Real Torino']:
            pts = [p for p in data_points if p['source'] == source and p['residual'] is not None]
            fids = [p['fidelity'] for p in pts]
            resids = [p['residual'] for p in pts]
            ax3.scatter(fids, resids, c=colors[source], marker=markers[source], s=80, 
                       alpha=0.7, label=source, edgecolors='white', linewidths=0.5)

        ax3.set_xlabel('Fidelity', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Residual Error ||Ax-b||/||b||', fontsize=11, fontweight='bold')
        ax3.set_title('Fidelity vs Residual Error', fontsize=12, fontweight='bold')
        ax3.set_xlim([0.70, 1.02])
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=9, loc='upper left')

        # Panel 4: Residual vs Shots
        ax4 = axes[1, 1]
        for source in ['Heron Sim', 'Torino Aer', 'Real Torino']:
            pts = [p for p in data_points if p['source'] == source and p['residual'] is not None]
            shots = [p['shots'] for p in pts]
            resids = [p['residual'] for p in pts]
            ax4.scatter(shots, resids, c=colors[source], marker=markers[source], s=80, 
                       alpha=0.7, label=source, edgecolors='white', linewidths=0.5)

        ax4.set_xscale('log')
        ax4.set_xlabel('Shots', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Residual Error ||Ax-b||/||b||', fontsize=11, fontweight='bold')
        ax4.set_title('Residual Error vs Shots', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(fontsize=9, loc='upper right')

    plt.suptitle('HHL Solution Quality Analysis: Hele-Shaw 2×2', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = base_dir / 'solution_quality_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
