#!/usr/bin/env python3
"""
Visualization script for Grover's algorithm results.

Reads JSON output from grovers.py and generates visualization plots.

Usage:
    python src/postproc/grovers_viz.py <stdout_json_file>
    or
    python -m postproc.grovers_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import AmplificationProblem, Grover


def create_visualization(data, output_file=None, display=True):
    """
    Create visualization of Grover's algorithm results.
    
    Args:
        data: Dictionary containing Grover's algorithm results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    # Extract data from standardized JSON structure
    problem = data.get("problem", {})
    target_states = problem.get("target_states", [])
    circuit_info = data.get("circuit_info", {})
    iterations = circuit_info.get("iterations", 0)
    viz_data = data.get("visualization_data", {})
    probabilities_by_iter = viz_data.get("probabilities_by_iteration", [])
    target_indices = viz_data.get("target_indices", [])
    use_noise = viz_data.get("use_noise", False)
    
    # Reconstruct the Grover circuit for visualization
    marked_states = target_states
    oracle_vector = sum((Statevector.from_label(s) for s in marked_states), 
                        Statevector.from_label('0' * len(marked_states[0])) * 0)
    import numpy as np
    oracle = oracle_vector / np.linalg.norm(oracle_vector.data)
    problem = AmplificationProblem(oracle, is_good_state=marked_states)
    sampler = StatevectorSampler()
    grover = Grover(sampler=sampler, iterations=iterations)
    grover_circuit = grover.construct_circuit(problem)
    
    # Create combined figure with circuit and iteration plots
    fig = plt.figure(figsize=(max(8, 2 * (iterations + 1)), 5))
    gs = GridSpec(2, iterations + 1, figure=fig, height_ratios=[1.2, 1])
    
    # Top row: circuit diagram spanning all columns
    ax_circuit = fig.add_subplot(gs[0, :])
    decomposed = grover_circuit.decompose().decompose()
    decomposed.draw('mpl', ax=ax_circuit, fold=-1)
    ax_circuit.set_title("Grover Circuit (decomposed)", fontsize=10, fontweight='bold')
    
    # Bottom row: probability distributions for each iteration
    for i, probs in enumerate(probabilities_by_iter):
        ax = fig.add_subplot(gs[1, i])
        x = range(len(probs))
        colors = ['red' if idx in target_indices else 'steelblue' for idx in x]
        ax.bar(x, probs, color=colors)
        ax.set_title(f"After {i} iter", fontsize=9)
        ax.set_xlabel("State", fontsize=8)
        ax.set_ylabel("Prob" if i == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        total_target_prob = sum(probs[idx] for idx in target_indices)
        ax.text(0.95, 0.95, f"P={total_target_prob:.2f}", 
                transform=ax.transAxes, ha='right', va='top', color='red', fontsize=8)
    
    target_labels = ", ".join(f"|{s}⟩" for s in marked_states)
    config = data.get("config", {})
    t1 = config.get("t1_us")
    t2 = config.get("t2_us")
    noise_label = f" (T1={t1}µs, T2={t2}µs)" if use_noise and t1 and t2 else " (ideal)"
    fig.suptitle(f"Grover's Algorithm: {target_labels}{noise_label}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/grovers_viz.py <postproc_json>")
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
        output_file = case_dir / "grovers.png"
        
        create_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
