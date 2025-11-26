#!/usr/bin/env python3
"""
UQ comparison plot: Heron noise model vs ibm_torino_aer vs Real Torino hardware.
Hele-Shaw 2x2, n=4 runs each, showing mean with shaded 95% CI.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def calculate_fidelity(classical_sol, quantum_sol):
    """Calculate fidelity between classical and quantum solutions."""
    classical_norm = np.linalg.norm(classical_sol)
    quantum_norm = np.linalg.norm(quantum_sol)
    if classical_norm > 0 and quantum_norm > 0:
        return np.abs(np.dot(classical_sol, quantum_sol)) / (classical_norm * quantum_norm)
    return 0.0


def get_fidelities_by_shots(wf_dir):
    """Load fidelities from a workflow directory, grouped by shot count."""
    fidelities = {100: [], 1000: [], 10000: [], 100000: []}
    for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
        with open(ckpt) as f:
            data = json.load(f)
        shots = data['params']['qc_shots']
        fidelity = calculate_fidelity(
            np.array(data['classical_solution']),
            np.array(data['quantum_solution'])
        )
        fidelities[shots].append(fidelity)
    return fidelities


def get_stats(fids, shot_counts):
    """Calculate mean and 95% CI for each shot count."""
    means = []
    ci_lower = []
    ci_upper = []
    stds = []
    
    for s in shot_counts:
        data = fids[s]
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1) if n > 1 else 0.0
        
        if n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=stats.sem(data))
        else:
            ci = (mean, mean)
        
        means.append(mean)
        stds.append(std)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    
    return means, stds, ci_lower, ci_upper


def main():
    base_dir = Path('/Users/agallojr/.lwfm/out/qt02-cfd')
    shot_counts = [100, 1000, 10000, 100000]
    
    # Load data
    # Generic Heron noise (4f765788)
    heron_fids = get_fidelities_by_shots(base_dir / '4f765788')
    
    # ibm_torino_aer (0e63f2d8)
    torino_aer_fids = get_fidelities_by_shots(base_dir / '0e63f2d8')
    
    # Real Torino (4 runs)
    real_fids = {100: [], 1000: [], 10000: [], 100000: []}
    for wf_id in ['df2add62', '058cf139', 'ea21306c', '8097a5fe']:
        wf_dir = base_dir / wf_id
        for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
            with open(ckpt) as f:
                data = json.load(f)
            shots = data['params']['qc_shots']
            fidelity = calculate_fidelity(
                np.array(data['classical_solution']),
                np.array(data['quantum_solution'])
            )
            real_fids[shots].append(fidelity)
    
    # Calculate stats
    heron_means, heron_stds, heron_ci_lo, heron_ci_hi = get_stats(heron_fids, shot_counts)
    taer_means, taer_stds, taer_ci_lo, taer_ci_hi = get_stats(torino_aer_fids, shot_counts)
    real_means, real_stds, real_ci_lo, real_ci_hi = get_stats(real_fids, shot_counts)
    
    # Colors
    color_heron = '#2E8B57'  # green
    color_taer = '#5B8FF9'   # blue
    color_real = '#C73E1D'   # red
    
    # Create plot
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Plot individual Torino runs (faint lines)
    for wf_id in ['df2add62', '058cf139', 'ea21306c', '8097a5fe']:
        wf_dir = base_dir / wf_id
        run_fids = {s: None for s in shot_counts}
        for ckpt in sorted(wf_dir.glob('checkpoint_*.json')):
            with open(ckpt) as f:
                data = json.load(f)
            shots = data['params']['qc_shots']
            fidelity = calculate_fidelity(
                np.array(data['classical_solution']),
                np.array(data['quantum_solution'])
            )
            run_fids[shots] = fidelity
        run_vals = [run_fids[s] for s in shot_counts]
        ax.semilogx(shot_counts, run_vals, '^-', linewidth=1, markersize=5,
                    color=color_real, alpha=0.3)
    
    # Plot mean lines with markers (sims first, then real on top)
    ax.semilogx(shot_counts, heron_means, 'o-', linewidth=2.5, markersize=10,
                color=color_heron, label='Heron Noise Model (generic)')
    ax.semilogx(shot_counts, taer_means, 's-', linewidth=2.5, markersize=10,
                color=color_taer, label='ibm_torino_aer (calibration)')
    ax.semilogx(shot_counts, real_means, '^-', linewidth=2.5, markersize=10,
                color=color_real, label='IBM Torino (real, n=4 runs)')
    
    # Plot 95% CI as shaded regions
    ax.fill_between(shot_counts, heron_ci_lo, heron_ci_hi, alpha=0.2, color=color_heron)
    ax.fill_between(shot_counts, taer_ci_lo, taer_ci_hi, alpha=0.2, color=color_taer)
    ax.fill_between(shot_counts, real_ci_lo, real_ci_hi, alpha=0.2, color=color_real)
    
    # Formatting
    ax.set_xlabel('Number of Shots', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('HHL Fidelity UQ: Noise Models vs Real Hardware\n'
                 'Hele-Shaw 2×2 (n=4 runs, mean with shaded 95% CI)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0.70, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    # Stats box
    stats_text = (f'Mean Fidelity (all shots):\n'
                  f'  Heron:      {np.mean(heron_means):.1%}  σ={np.mean(heron_stds):.4f}\n'
                  f'  Torino Aer: {np.mean(taer_means):.1%}  σ={np.mean(taer_stds):.4f}\n'
                  f'  Real:       {np.mean(real_means):.1%}  σ={np.mean(real_stds):.4f}')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = base_dir / 'uq_three_way_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
