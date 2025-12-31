#!/usr/bin/env python3
"""
Visualization script for QPE ground state energy results.

Reads JSON output from gs_qpe.py and generates visualization plots.

Usage:
    python src/postproc/qpe_viz.py <stdout_json_file>
    or
    python -m postproc.qpe_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of QPE results.
    
    Args:
        data: Dictionary containing QPE results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data
    molecule = data.get("molecule", {})
    mol_name = molecule.get("name", "Unknown")
    ref_energies = data.get("reference_energies", {})
    fci_energy = ref_energies.get("fci_hartree", 0)
    scf_energy = ref_energies.get("scf_hartree", 0)
    
    qpe_data = data.get("qpe", {})
    qpe_energy = qpe_data.get("estimated_energy", 0)
    phase_counts = qpe_data.get("phase_counts", {})
    
    analysis = data.get("analysis", {})
    error = analysis.get("error_hartree", 0)
    within_accuracy = analysis.get("within_chemical_accuracy", False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Left: Energy comparison bar chart
    energies = [fci_energy, scf_energy, qpe_energy]
    labels = ['FCI\n(exact)', 'SCF\n(mean-field)', 'QPE\n(quantum)']
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
    
    # Right: Phase measurement histogram
    if phase_counts:
        phases = sorted(phase_counts.keys())
        counts = [phase_counts[p] for p in phases]
        
        ax2.bar(range(len(phases)), counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Measured Phase (binary)', fontsize=9)
        ax2.set_ylabel('Counts', fontsize=9)
        ax2.set_title('QPE Phase Measurements', fontsize=10, fontweight='bold')
        ax2.set_xticks(range(len(phases)))
        ax2.set_xticklabels(phases, rotation=45, ha='right', fontsize=7)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
    
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
        print("Usage: python src/postproc/qpe_viz.py <postproc_json>")
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
        output_file = case_dir / "qpe_results.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
