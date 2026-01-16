#!/usr/bin/env python3
"""
Visualization script for SKQD (Sample-based Krylov Quantum Diagonalization) results.

Reads JSON output from gs_siam_skqd.py and generates visualization plots.

Usage:
    python src/postproc/skqd_viz.py <stdout_json_file>
    or
    python -m postproc.skqd_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of SKQD results.
    
    Args:
        data: Dictionary containing SKQD results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data from standardized JSON structure
    problem = data.get("problem", {})
    siam_info = problem.get("siam_info", {})
    model_name = siam_info.get("model", "SIAM")
    num_orbs = siam_info.get("num_orbs", 0)
    exact_energy = problem.get("exact_energy", 0)
    
    results = data.get("results", {})
    skqd_energy = results.get("skqd_energy", 0)
    energy_history = results.get("energy_history", [])
    
    metrics = data.get("metrics", {})
    error_abs = metrics.get("error_abs", 0)
    error_pct = metrics.get("error_pct", 0)
    
    # Create figure with subplots
    if energy_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
        ax2 = None
    
    # Left: Energy comparison bar chart
    energies = [exact_energy, skqd_energy]
    labels = ['Exact\n(ED)', 'SKQD\n(quantum)']
    colors = ['green', 'coral']
    
    bars = ax1.bar(labels, energies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Energy', fontsize=9)
    ax1.set_title(f'{model_name} ({num_orbs} orbitals) Ground State', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=8)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.4f}',
                ha='center', va='bottom', fontsize=7)
    
    # Right: Energy convergence (if available)
    if ax2 and energy_history:
        ax2.plot(range(1, len(energy_history) + 1), energy_history, 'b-o', 
                linewidth=2, markersize=4, label='SKQD Energy')
        ax2.axhline(y=exact_energy, color='g', linestyle='--', 
                   linewidth=2, label=f'Exact: {exact_energy:.4f}')
        ax2.set_xlabel('SQD Iteration', fontsize=9)
        ax2.set_ylabel('Energy', fontsize=9)
        ax2.set_title('SKQD Convergence', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=8)
    
    # Add error info to figure
    fig.text(0.5, 0.02, f'Error: {error_abs:.2e} ({error_pct:.2e}%)', 
             ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/skqd_viz.py <postproc_json>")
        sys.exit(1)
    
    # Load postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    # Process each case directory
    for case_dir_str in context.get("case_dirs", []):
        case_dir = Path(case_dir_str)
        stdout_file = case_dir / "stdout.json"
        
        if not stdout_file.exists():
            print(f"Warning: stdout.json not found in {case_dir}")
            continue
        
        with open(stdout_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Generate output filename
        output_file = case_dir / "skqd_results.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
