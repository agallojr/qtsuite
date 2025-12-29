#!/usr/bin/env python3
"""
Postprocessing script for noise study results.

Plots fidelity of quantum ground state to classically computed ground state
as a function of T1 and T2 noise parameters.

Usage:
    python postproc_noise.py /path/to/_postproc_groupname.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_case_data(case_dir: Path) -> dict:
    """Load params and results from a case directory."""
    params_file = case_dir / "params.json"
    stdout_file = case_dir / "stdout.json"
    
    data = {"case_dir": str(case_dir)}
    
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            data["params"] = json.load(f)
    
    if stdout_file.exists():
        with open(stdout_file, "r", encoding="utf-8") as f:
            try:
                data["output"] = json.load(f)
            except json.JSONDecodeError:
                data["output"] = None
    
    return data


def compute_metrics(case_data: dict) -> dict:
    """
    Compute metrics from case output.
    
    Returns dict with:
        - error_hartree: absolute error vs FCI energy
        - phase_concentration: fraction of shots at best phase (higher = less noise)
        - energy_std: standard deviation of energy estimates from phase distribution
        - energy_uncertainty: 95% confidence interval half-width for energy
    """
    output = case_data.get("output", {})
    if not output:
        return None
    
    metrics = {}
    
    # Error in Hartree
    analysis = output.get("analysis", {})
    error = analysis.get("error_hartree")
    if error is not None:
        metrics["error_hartree"] = abs(error)
    
    # Phase concentration and uncertainty from phase distribution
    qpe = output.get("qpe", {})
    phase_counts = qpe.get("phase_counts", {})
    evolution_time = qpe.get("evolution_time", 1.0)
    energy_shift = qpe.get("energy_shift", 0.0)
    
    if phase_counts:
        total_shots = sum(phase_counts.values())
        best_count = max(phase_counts.values())
        metrics["phase_concentration"] = best_count / total_shots
        
        # Compute energy distribution from phase counts
        energies = []
        weights = []
        for phase_str, count in phase_counts.items():
            phase = float(phase_str)
            # Convert phase to energy: E = phase * 2π / t + shift
            energy = phase * 2 * np.pi / evolution_time + energy_shift
            energies.append(energy)
            weights.append(count)
        
        # Weighted mean and std
        energies = np.array(energies)
        weights = np.array(weights) / total_shots
        mean_energy = np.sum(energies * weights)
        variance = np.sum(weights * (energies - mean_energy) ** 2)
        std_energy = np.sqrt(variance)
        
        metrics["energy_std"] = std_energy
        # 95% CI assuming roughly normal: ~1.96 * std
        metrics["energy_uncertainty"] = 1.96 * std_energy
    
    return metrics if metrics else None


def main():
    if len(sys.argv) < 2:
        print("Usage: python postproc_noise.py <postproc_json>")
        sys.exit(1)
    
    # Load postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    output_dir = Path(context["run_dir"])
    case_dirs = context.get("case_dirs", [])
    groups = context.get("groups", [])  # For _post_postproc
    group_name = context.get("group_id", "noise_study")  # For per-group postproc
    
    # Load data from all cases
    cases = []
    for case_dir_str in case_dirs:
        case_dir = Path(case_dir_str)
        if case_dir.exists():
            data = load_case_data(case_dir)
            # Add case directory name to identify group
            data["case_name"] = case_dir.name
            cases.append(data)
    
    if not cases:
        print("No valid case data found")
        sys.exit(1)
    
    # Separate noiseless cases (no t1/t2 params) from noisy cases
    noiseless_cases = []
    noisy_cases = []
    
    for case in cases:
        params = case.get("params", {})
        t1 = params.get("t1")
        t2 = params.get("t2")
        metrics = compute_metrics(case)
        
        if metrics is None:
            continue
        
        if t1 is None or t2 is None:
            # Noiseless case (no t1/t2 specified)
            noiseless_cases.append({
                'case_name': case.get("case_name", "unknown"),
                'error': metrics.get("error_hartree", 0),
                'concentration': metrics.get("phase_concentration", 0)
            })
        else:
            # Noisy case
            noisy_cases.append({
                'case_name': case.get("case_name", "unknown"),
                't1': float(t1),
                't2': float(t2),
                'error': metrics.get("error_hartree", 0),
                'concentration': metrics.get("phase_concentration", 0)
            })
    
    # Get noiseless baseline
    if noiseless_cases:
        noiseless_error = np.mean([c['error'] for c in noiseless_cases])
    else:
        print("Warning: No noiseless case found")
        noiseless_error = min([c['error'] for c in noisy_cases]) if noisy_cases else 0
    
    # Create bar chart comparing noisy vs noiseless
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort noisy cases by error (worst first)
    noisy_cases.sort(key=lambda x: x['error'], reverse=True)
    
    labels = [f"T1={c['t1']}, T2={c['t2']}" for c in noisy_cases]
    noisy_errors = [c['error'] for c in noisy_cases]
    
    # Add noiseless as reference
    labels.insert(0, "Noiseless")
    all_errors = [noiseless_error] + noisy_errors
    
    # Color bars: green for noiseless, gradient for noisy
    colors = ['#2ca02c'] + ['#d62728'] * len(noisy_errors)
    
    x = range(len(labels))
    bars = ax.bar(x, all_errors, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, err in zip(bars, all_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{err:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Add horizontal line for noiseless reference
    ax.axhline(y=noiseless_error, color='#2ca02c', linestyle='--', linewidth=2, 
               label=f'Noiseless baseline: {noiseless_error:.4f} Ha')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Error (Hartree)", fontsize=12)
    ax.set_title(f"Noise Study: {group_name}\nError vs Noiseless Baseline", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)
    
    # Compute and show degradation stats
    if noisy_errors:
        avg_noisy = np.mean(noisy_errors)
        worst_noisy = max(noisy_errors)
        degradation = (avg_noisy - noiseless_error) / noiseless_error * 100 if noiseless_error > 0 else 0
        
        summary = (f"Noiseless error: {noiseless_error:.4f} Ha\n"
                   f"Avg noisy error: {avg_noisy:.4f} Ha\n"
                   f"Worst noisy error: {worst_noisy:.4f} Ha\n"
                   f"Avg degradation: {degradation:+.1f}%")
        
        ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot with group name
    plot_file = output_dir / f"{group_name}_noise_study.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")
    
    # Also display the plot
    plt.show()
    
    # Print summary
    print("\n=== Noise Study Summary ===")
    print(f"Noiseless cases: {len(noiseless_cases)}")
    print(f"Noisy cases: {len(noisy_cases)}")
    if noisy_cases:
        t1_vals = [c['t1'] for c in noisy_cases]
        t2_vals = [c['t2'] for c in noisy_cases]
        print(f"T1 range: {min(t1_vals):.0f} - {max(t1_vals):.0f} μs")
        print(f"T2 range: {min(t2_vals):.0f} - {max(t2_vals):.0f} μs")


if __name__ == "__main__":
    main()
