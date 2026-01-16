#!/usr/bin/env python3
"""
Visualization script for VQE ground state energy results.

Reads JSON output from gs_vqe.py and generates visualization plots.

Usage:
    python src/postproc/vqe_viz.py <stdout_json_file>
    or
    python -m postproc.vqe_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of VQE results.
    
    Args:
        data: Dictionary containing VQE results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data from standardized JSON structure
    problem = data.get("problem", {})
    molecule = problem.get("molecule", {})
    mol_name = molecule.get("name", "Unknown")
    ref_energies = problem.get("reference_energies", {})
    fci_energy = ref_energies.get("fci_hartree", 0)
    scf_energy = ref_energies.get("scf_hartree", 0)
    
    results = data.get("results", {})
    vqe_energy = results.get("energy", 0)
    energy_history = results.get("energy_history", [])
    
    metrics = data.get("metrics", {})
    error = metrics.get("error_hartree", 0)
    within_accuracy = metrics.get("within_chemical_accuracy", False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Left: Energy comparison bar chart
    energies = [fci_energy, scf_energy, vqe_energy]
    labels = ['FCI\n(exact)', 'SCF\n(mean-field)', 'VQE\n(quantum)']
    colors = ['green', 'orange', 'coral']
    
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
    if energy_history:
        ax2.plot(energy_history, 'b-', linewidth=2, label='VQE optimization')
        ax2.axhline(y=fci_energy, color='g', linestyle='--', 
                   linewidth=2, label=f'FCI: {fci_energy:.4f}')
        ax2.axhline(y=vqe_energy, color='r', linestyle=':',
                   label=f'VQE final: {vqe_energy:.4f}')
        ax2.set_xlabel('Iteration', fontsize=9)
        ax2.set_ylabel('Energy (Hartree)', fontsize=9)
        ax2.set_title('VQE Convergence', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=8)
    else:
        # If no energy history, show attempt results if available
        attempts = results.get("attempts", [])
        if attempts:
            attempt_nums = [a["attempt"] for a in attempts]
            attempt_energies = [a["energy"] for a in attempts]
            ax2.scatter(attempt_nums, attempt_energies, color='steelblue', s=50, alpha=0.7)
            ax2.axhline(y=fci_energy, color='g', linestyle='--', 
                       linewidth=2, label=f'FCI: {fci_energy:.4f}')
            ax2.set_xlabel('Attempt', fontsize=9)
            ax2.set_ylabel('Energy (Hartree)', fontsize=9)
            ax2.set_title('VQE Attempts', fontsize=10, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=7)
            ax2.tick_params(labelsize=8)
        else:
            ax2.text(0.5, 0.5, 'No convergence data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('VQE Convergence', fontsize=10, fontweight='bold')
    
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
        print("Usage: python src/postproc/vqe_viz.py <postproc_json>")
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
        output_file = case_dir / "vqe_results.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
