#!/usr/bin/env python3
"""
Algorithm comparison visualization for Ax=b solvers.

Creates a summary plot comparing fidelity, circuit depth, and qubit count
across different quantum linear system algorithms.

Usage:
    python mod2/src/postproc/algorithm_comparison_viz.py <final_postproc_json>
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_comparison_plot(data, output_file=None, display=True):
    """
    Create a multi-panel comparison plot for algorithm results.
    
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
    
    # Extract metrics
    fidelities = []
    l2_errors = []
    depths = []
    qubits = []
    
    for r in results:
        metrics = r.get("metrics", {})
        circuit_info = r.get("circuit_info", {})
        
        fidelities.append(metrics.get("fidelity", 0))
        l2_errors.append(metrics.get("l2_error", 0))
        
        # Handle different circuit_info structures
        if "transpiled_stats" in circuit_info:
            depths.append(circuit_info["transpiled_stats"].get("depth", circuit_info.get("depth", 0)))
            qubits.append(circuit_info["transpiled_stats"].get("num_qubits", circuit_info.get("num_qubits", 0)))
        else:
            depths.append(circuit_info.get("depth", 0) or 0)
            qubits.append(circuit_info.get("num_qubits", 0) or 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    x = np.arange(len(algorithms))
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:len(algorithms)]
    
    # Panel 1: Fidelity
    ax1 = axes[0]
    bars1 = ax1.bar(x, fidelities, color=colors)
    ax1.set_ylabel('Fidelity', fontsize=10)
    ax1.set_title('Solution Fidelity', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    # Add value labels
    for bar, val in zip(bars1, fidelities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Panel 2: Circuit Depth
    ax2 = axes[1]
    bars2 = ax2.bar(x, depths, color=colors)
    ax2.set_ylabel('Depth', fontsize=10)
    ax2.set_title('Circuit Depth', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    # Add value labels
    for bar, val in zip(bars2, depths):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    # Panel 3: Qubit Count
    ax3 = axes[2]
    bars3 = ax3.bar(x, qubits, color=colors)
    ax3.set_ylabel('Qubits', fontsize=10)
    ax3.set_title('Qubit Count', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    # Add value labels
    for bar, val in zip(bars3, qubits):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    # Get problem info from first result for title
    problem = results[0].get("problem", {})
    dim = problem.get("dimension", problem.get("original_size", "?"))
    cond = problem.get("condition_number", 0)
    
    fig.suptitle(f'Ax=b Algorithm Comparison (n={dim}, κ={cond:.2f})', fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python mod2/src/postproc/algorithm_comparison_viz.py <final_postproc_json>")
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
            
            # Use directory name as algorithm label
            algorithm_name = case_dir.name.upper()
            data.append((algorithm_name, results))
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {stdout_file}: {e}")
            continue
    
    if not data:
        print("No valid results found")
        sys.exit(1)
    
    # Generate output filename in the run directory
    run_dir = postproc_json.parent
    output_file = run_dir / "algorithm_comparison.png"
    
    create_comparison_plot(data, output_file=output_file, display=False)


if __name__ == "__main__":
    main()
