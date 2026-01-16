#!/usr/bin/env python3
"""
Visualization script for SQD ground state energy results.

Reads JSON output from gs_sqd.py and generates visualization plots.

Usage:
    python src/postproc/sqd_viz.py <stdout_json_file>
    or
    python -m postproc.sqd_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of SQD results.
    
    Args:
        data: Dictionary containing SQD results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data from standardized JSON structure
    problem = data.get("problem", {})
    molecule = problem.get("molecule", {})
    mol_name = molecule.get("name", "Unknown")
    fci_energy = problem.get("exact_energy_hartree", 0)
    
    results = data.get("results", {})
    sqd_energy = results.get("final_energy", 0)
    # energy_history is 2D (iterations x batches) - flatten to 1D by taking mean per iteration
    energy_history_raw = results.get("energy_history", [])
    spin_sq_history_raw = results.get("spin_sq_history", [])
    
    # Flatten 2D arrays to 1D (mean across batches)
    if energy_history_raw and isinstance(energy_history_raw[0], list):
        energy_history = [sum(row)/len(row) if row else 0 for row in energy_history_raw]
        spin_sq_history = [sum(row)/len(row) if row else 0 for row in spin_sq_history_raw]
    else:
        energy_history = energy_history_raw
        spin_sq_history = spin_sq_history_raw
    
    metrics = data.get("metrics", {})
    error = metrics.get("error_hartree", 0)
    within_accuracy = metrics.get("within_chemical_accuracy", False)
    
    # Create figure with subplots
    if energy_history and spin_sq_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
        ax2 = None
    
    # Left: Energy comparison bar chart
    energies = [fci_energy, sqd_energy]
    labels = ['FCI\n(exact)', 'SQD\n(quantum)']
    colors = ['green', 'coral']
    
    bars = ax1.bar(labels, energies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Energy (Hartree)', fontsize=9)
    ax1.set_title(f'{mol_name} Ground State Energy', fontsize=10, fontweight='bold')
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
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(energy_history, 'b-', linewidth=2, label='SQD Energy')
        ax2.axhline(y=fci_energy, color='g', linestyle='--', 
                   linewidth=2, label=f'FCI: {fci_energy:.4f}')
        ax2.set_xlabel('Iteration', fontsize=9)
        ax2.set_ylabel('Energy (Hartree)', fontsize=9, color='b')
        ax2.tick_params(axis='y', labelcolor='b', labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        if spin_sq_history:
            line2 = ax2_twin.plot(spin_sq_history, 'r--', linewidth=2, label='⟨S²⟩')
            ax2_twin.set_ylabel('⟨S²⟩', fontsize=9, color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r', labelsize=8)
            
            lines = line1 + line2
            labels_legend = [l.get_label() for l in lines]
            ax2.legend(lines, labels_legend, fontsize=7, loc='best')
        else:
            ax2.legend(fontsize=7)
        
        ax2.set_title('SQD Convergence', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=8)
    
    # Add error info to figure
    accuracy_text = "✓" if within_accuracy else "✗"
    fig.text(0.5, 0.02, f'Error: {error:.6f} Ha {accuracy_text}', 
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
        print("Usage: python src/postproc/sqd_viz.py <postproc_json>")
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
        output_file = case_dir / "sqd_results.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
