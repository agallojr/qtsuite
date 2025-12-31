#!/usr/bin/env python3
"""
Visualization script for phase kickback results.

Reads JSON output from phase_kickback.py and generates visualization plots.

Usage:
    python src/postproc/phase_kickback_viz.py <stdout_json_file>
    or
    python -m postproc.phase_kickback_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram


def make_circuit_phase_kickback(x_gate: bool) -> QuantumCircuit:
    """
    Recreate the phase kickback circuit for visualization.
    
    Args:
        x_gate: Whether to apply X gate to target qubit
    
    Returns:
        QuantumCircuit without measurements
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    if x_gate:
        qc.x(1)
    qc.cz(0, 1)
    qc.h(0)
    return qc


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of phase kickback results.
    
    Args:
        data: Dictionary containing phase kickback results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    with_kickback = data.get("with_kickback", {})
    without_kickback = data.get("without_kickback", {})
    
    counts_kick = with_kickback.get("counts", {})
    counts_no_kick = without_kickback.get("counts", {})
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))
    
    # Top row: With kickback
    qc_kick = make_circuit_phase_kickback(x_gate=True)
    qc_kick.draw("mpl", ax=axes[0, 0])
    axes[0, 0].set_title("With Kickback (target=|1⟩)", fontsize=12, fontweight='bold')
    
    plot_histogram(counts_kick, ax=axes[0, 1])
    axes[0, 1].set_title("Results: control → |1⟩", fontsize=12, fontweight='bold')
    
    # Bottom row: Without kickback
    qc_no_kick = make_circuit_phase_kickback(x_gate=False)
    qc_no_kick.draw("mpl", ax=axes[1, 0])
    axes[1, 0].set_title("No Kickback (target=|0⟩)", fontsize=12, fontweight='bold')
    
    plot_histogram(counts_no_kick, ax=axes[1, 1])
    axes[1, 1].set_title("Results: control → |0⟩", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/phase_kickback_viz.py <postproc_json>")
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
        output_file = case_dir / "phase_kickback.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
