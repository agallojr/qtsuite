#!/usr/bin/env python3
"""
Postprocessing script for noise study results.

Plots fidelity of quantum ground state to classically computed ground state
as a function of T1 and T2 noise parameters.

Usage:
    python postproc_noise.py --output-dir /path/to/run /path/to/case1 /path/to/case2 ...
"""

import argparse
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


def compute_fidelity(case_data: dict) -> float:
    """
    Compute fidelity metric from case output.
    
    Uses the error vs chemical accuracy as a proxy for fidelity.
    Lower error = higher fidelity.
    """
    output = case_data.get("output", {})
    if not output:
        return None
    
    analysis = output.get("analysis", {})
    
    # Use error in Hartree - smaller is better
    error = analysis.get("error_hartree")
    if error is None:
        return None
    
    # Convert error to a fidelity-like metric (0-1 scale)
    # Using exponential decay: fidelity = exp(-|error| / scale)
    # Scale chosen so chemical accuracy (0.0016 Ha) gives ~0.99 fidelity
    scale = 0.05  # Hartree
    fidelity = np.exp(-abs(error) / scale)
    
    return fidelity


def main():
    parser = argparse.ArgumentParser(
        description="Plot noise study results: fidelity vs T1/T2"
    )
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save the plot")
    parser.add_argument("case_dirs", nargs="+",
                        help="Case directories to process")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Load data from all cases
    cases = []
    for case_dir_str in args.case_dirs:
        case_dir = Path(case_dir_str)
        if case_dir.exists():
            data = load_case_data(case_dir)
            cases.append(data)
    
    if not cases:
        print("No valid case data found")
        sys.exit(1)
    
    # Extract T1, T2, and fidelity values
    t1_values = []
    t2_values = []
    fidelities = []
    
    for case in cases:
        params = case.get("params", {})
        t1 = params.get("t1")
        t2 = params.get("t2")
        fidelity = compute_fidelity(case)
        
        if t1 is not None and t2 is not None and fidelity is not None:
            t1_values.append(float(t1))
            t2_values.append(float(t2))
            fidelities.append(fidelity)
    
    if not fidelities:
        print("No valid fidelity data found")
        sys.exit(1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with color representing fidelity
    scatter = ax.scatter(
        t1_values, t2_values,
        c=fidelities,
        cmap="RdYlGn",
        s=200,
        vmin=0, vmax=1,
        edgecolors="black",
        linewidths=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Fidelity", fontsize=12)
    
    # Add value labels on each point
    for t1, t2, fid in zip(t1_values, t2_values, fidelities):
        ax.annotate(
            f"{fid:.3f}",
            (t1, t2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9
        )
    
    ax.set_xlabel("T1 (μs)", fontsize=12)
    ax.set_ylabel("T2 (μs)", fontsize=12)
    ax.set_title("Ground State Fidelity vs Noise Parameters (T1, T2)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plot_file = output_dir / "noise_fidelity.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")
    
    # Also display the plot
    plt.show()
    
    # Print summary
    print("\n=== Noise Study Summary ===")
    print(f"Number of cases: {len(fidelities)}")
    print(f"T1 range: {min(t1_values):.0f} - {max(t1_values):.0f} μs")
    print(f"T2 range: {min(t2_values):.0f} - {max(t2_values):.0f} μs")
    print(f"Fidelity range: {min(fidelities):.4f} - {max(fidelities):.4f}")
    print(f"Mean fidelity: {np.mean(fidelities):.4f}")


if __name__ == "__main__":
    main()
