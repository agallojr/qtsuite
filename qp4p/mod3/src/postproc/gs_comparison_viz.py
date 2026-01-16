#!/usr/bin/env python3
"""
Ground state energy algorithm comparison visualization.

Creates a summary plot comparing energy accuracy, circuit depth, and qubit count
across different quantum ground state algorithms (VQE, QPE, SQD).

Usage:
    python mod3/src/postproc/gs_comparison_viz.py <final_postproc_json>
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_comparison_plot(data, output_file=None, display=True):
    """
    Create a multi-panel comparison plot for ground state algorithm results.
    
    Args:
        data: List of (algorithm_name, results_dict) tuples
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    if not data:
        print("No data to plot")
        return
    
    algorithms = [d[0] for d in data]
    results = [d[1] for d in data]
    
    # Extract metrics - handle different JSON structures
    energies = []
    errors = []
    depths = []
    qubits = []
    exact_energy = None
    mol_name = "Unknown"
    
    for r in results:
        algorithm = r.get("algorithm", "")
        metrics = r.get("metrics", {})
        problem = r.get("problem", {})
        results_data = r.get("results", {})
        circuit_info = r.get("circuit_info", {})
        
        # Get molecule name
        molecule = problem.get("molecule", {})
        if molecule.get("name"):
            mol_name = molecule.get("name")
        
        # Get exact energy reference
        if exact_energy is None:
            if "reference_energies" in problem:
                exact_energy = problem["reference_energies"].get("fci_hartree", 0)
            elif "exact_energy_hartree" in problem:
                exact_energy = problem.get("exact_energy_hartree", 0)
        
        # Get computed energy based on algorithm type
        if algorithm == "vqe":
            energies.append(metrics.get("vqe_energy_hartree", results_data.get("energy", 0)))
            errors.append(metrics.get("error_hartree", 0))
        elif algorithm == "qpe":
            energies.append(metrics.get("qpe_energy_hartree", results_data.get("estimated_energy", 0)))
            errors.append(metrics.get("error_hartree", 0))
        elif algorithm == "sqd":
            energies.append(metrics.get("sqd_energy_hartree", results_data.get("final_energy", 0)))
            errors.append(metrics.get("error_hartree", 0))
        else:
            # Fallback
            energies.append(metrics.get("energy", 0))
            errors.append(metrics.get("error_hartree", 0))
        
        # Get circuit info
        if "transpiled_stats" in results_data:
            depths.append(results_data["transpiled_stats"].get("depth", 0))
            qubits.append(results_data["transpiled_stats"].get("num_qubits", 0))
        elif "circuit_stats" in results_data:
            depths.append(results_data["circuit_stats"].get("depth", 0))
            qubits.append(results_data["circuit_stats"].get("num_qubits", 0))
        elif circuit_info:
            depths.append(circuit_info.get("depth", 0) or 0)
            qubits.append(circuit_info.get("num_qubits", 0) or 0)
        else:
            depths.append(0)
            qubits.append(0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.arange(len(algorithms))
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:len(algorithms)]
    
    # Panel 1: Energy comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x, energies, color=colors)
    if exact_energy:
        ax1.axhline(y=exact_energy, color='green', linestyle='--', linewidth=2, 
                   label=f'Exact: {exact_energy:.4f}')
        ax1.legend(fontsize=8)
    ax1.set_ylabel('Energy (Hartree)', fontsize=10)
    ax1.set_title('Ground State Energy', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    # Add value labels
    for bar, val in zip(bars1, energies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, 
                f'{val:.4f}', ha='center', va='top', fontsize=8, color='white', fontweight='bold')
    
    # Panel 2: Error (log scale if needed)
    ax2 = axes[1]
    bars2 = ax2.bar(x, errors, color=colors)
    ax2.axhline(y=0.0016, color='red', linestyle='--', linewidth=1.5, 
               label='Chemical accuracy')
    ax2.legend(fontsize=8)
    ax2.set_ylabel('Error (Hartree)', fontsize=10)
    ax2.set_title('Energy Error vs Exact', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    # Add value labels
    for bar, val in zip(bars2, errors):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Panel 3: Circuit Depth
    ax3 = axes[2]
    bars3 = ax3.bar(x, depths, color=colors)
    ax3.set_ylabel('Depth', fontsize=10)
    ax3.set_title('Circuit Depth', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    # Add value labels
    for bar, val in zip(bars3, depths):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(f'{mol_name} Ground State Algorithm Comparison', fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python mod3/src/postproc/gs_comparison_viz.py <final_postproc_json>")
        sys.exit(1)
    
    # Load final postproc context from JSON file
    postproc_json = Path(sys.argv[1])
    with open(postproc_json, "r", encoding="utf-8") as f:
        context = json.load(f)
    
    # Collect results from all case directories
    data = []
    for case_dir_str in context.get("case_dirs", []):
        case_dir = Path(case_dir_str)
        stdout_file = case_dir / "stdout.json"
        
        if not stdout_file.exists():
            print(f"Warning: stdout.json not found in {case_dir}")
            continue
        
        try:
            with open(stdout_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # Use algorithm field or directory name as label
            algorithm_name = results.get("algorithm", case_dir.name).upper()
            data.append((algorithm_name, results))
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {stdout_file}: {e}")
            continue
    
    if not data:
        print("No valid results found")
        sys.exit(1)
    
    # Generate output filename in the run directory
    run_dir = postproc_json.parent
    output_file = run_dir / "gs_algorithm_comparison.png"
    
    create_comparison_plot(data, output_file=output_file, display=False)


if __name__ == "__main__":
    main()
