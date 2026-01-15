#!/usr/bin/env python3
"""
Visualization script for linear system (Ax=b) VQLS results.

Reads JSON output from ax_equals_b.py and generates comparison plots.

Usage:
    python src/postproc/linear_system_viz.py <stdout_json_file>
    or
    python -m postproc.linear_system_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization comparing classical and quantum solutions.
    
    Args:
        data: Dictionary containing VQLS results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data from standardized JSON format
    results = data.get("results", {})
    metrics = data.get("metrics", {})
    
    classical_x = np.array(results.get("classical_solution", []))
    quantum_x = np.array(results.get("quantum_solution", []))
    fidelity = metrics.get("fidelity", 0)
    problem = data.get("problem", {})
    n_orig = problem.get("original_size", len(classical_x))
    condition_num = problem.get("condition_number", 0)
    
    # Create bar chart comparing solutions
    fig, ax = plt.subplots(figsize=(max(4, n_orig * 0.5), 3))
    
    x_indices = np.arange(n_orig)
    width = 0.35
    
    ax.bar(x_indices - width/2, classical_x, width, label='Classical', color='steelblue')
    ax.bar(x_indices + width/2, quantum_x, width, label='VQLS', color='coral')
    
    ax.set_xlabel('Solution component index', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.set_title(f'Ax=b Solution Comparison ({n_orig}x{n_orig} system)\n'
                f'Fidelity: {fidelity:.4f} | Condition #: {condition_num:.1f}',
                fontsize=10, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/linear_system_viz.py <postproc_json>")
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
        output_file = case_dir / "linear_system.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
