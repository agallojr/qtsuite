#!/usr/bin/env python3
"""
Plot condition number sensitivity analysis.

Shows how CFD parameters affect matrix conditioning and trigger
automatic matrix expansion.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(workflow_dir):
    """Load results from workflow directory."""
    results_file = Path(workflow_dir) / "results.pkl"
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)
    
    with open(results_file, 'rb') as f:
        case_data = pickle.load(f)
    
    return case_data


def extract_condition_data(case_data):
    """Extract condition number and matrix size data."""
    data = []
    
    # Debug: print first case structure
    if case_data:
        print("\nDebug: First case structure:")
        print(f"  Keys: {case_data[0].keys()}")
        print(f"  Sample params: {list(case_data[0]['params'].keys())[:10]}")
    
    for case in case_data:
        params = case['params']
        
        # Determine which parameter varies - create composite key if needed
        # Check for mesh size (nx*ny)
        mesh_size = params.get('nx', 2) * params.get('ny', 2)
        mu = params.get('mu', 1.0)
        
        # Use mesh_size as primary x-axis, mu as secondary
        x_param = 'mesh_size'
        x_val_base = mesh_size
        x_label = 'Grid Points (nx × ny)'
        
        # If mu varies, use it as distinguisher
        case_label = f"{params.get('nx', 2)}×{params.get('ny', 2)}, μ={mu}"
        
        x_val = mesh_size
        
        # Get original (pre-conditioning) metrics
        kappa_orig = params.get('_matrix_condition_number_original', None)
        matrix_size_orig = params.get('_matrix_size_original', None)
        
        # Get conditioned metrics
        kappa = params.get('_matrix_condition_number', None)
        matrix_size = params.get('_matrix_size', None)
        circuit_qubits = params.get('_circuit_qubits', None)
        
        print(f"  Case {x_param}={x_val}:")
        print(f"    Original: κ={kappa_orig}, matrix_size={matrix_size_orig}")
        print(f"    Conditioned: κ={kappa}, matrix_size={matrix_size}, qubits={circuit_qubits}")
        
        if kappa_orig is not None and matrix_size_orig is not None:
            data.append({
                'x': x_val,
                'kappa_original': kappa_orig,
                'matrix_size_original': matrix_size_orig,
                'kappa': kappa,
                'matrix_size': matrix_size,
                'circuit_qubits': circuit_qubits,
                'x_param': x_param,
                'x_label': x_label,
                'case_label': case_label,
                'mu': mu
            })
        else:
            print(f"    Skipping: missing original data")
    
    # Sort by x value
    data.sort(key=lambda d: d['x'])
    
    return data


def plot_condition_analysis(data, output_path, show_plot=False):
    """
    Create condition number analysis plot.
    
    Shows:
    1. Condition number vs parameter (with threshold)
    2. Matrix size growth
    3. Circuit qubit growth
    """
    if not data:
        print("No data to plot")
        return
    
    x_vals = [d['x'] for d in data]
    kappas_orig = [d['kappa_original'] for d in data]
    kappas = [d['kappa'] for d in data]
    matrix_sizes_orig = [d['matrix_size_original'] for d in data]
    matrix_sizes = [d['matrix_size'] for d in data]
    circuit_qubits = [d['circuit_qubits'] for d in data]
    x_label = data[0]['x_label']
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Color code by original condition number severity
    colors = []
    for kappa in kappas_orig:
        if kappa < 100:
            colors.append('green')
        elif kappa < 1e4:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Plot 1: Condition number (before and after)
    ax1.scatter(x_vals, kappas_orig, c=colors, s=100, alpha=0.7, edgecolors='black', 
                linewidth=1.5, label='Original (pre-conditioning)', marker='o')
    ax1.plot(x_vals, kappas_orig, 'k--', alpha=0.3, linewidth=1)
    
    ax1.scatter(x_vals, kappas, c='blue', s=80, alpha=0.7, edgecolors='black',
                linewidth=1.5, label='After conditioning', marker='s')
    ax1.plot(x_vals, kappas, 'b:', alpha=0.3, linewidth=1)
    
    # Add threshold line
    ax1.axhline(y=1e4, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                label='Max threshold (κ=10⁴)')
    ax1.axhline(y=100, color='orange', linestyle='--', linewidth=1, alpha=0.3,
                label='Well-conditioned (κ=100)')
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Condition Number (κ)', fontsize=12, fontweight='bold')
    ax1.set_title('Matrix Conditioning Sensitivity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10, loc='best')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Excellent (κ<100)'),
        Patch(facecolor='orange', edgecolor='black', label='Fair (100<κ<10⁴)'),
        Patch(facecolor='red', edgecolor='black', label='Poor (κ>10⁴)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Plot 2: Matrix size (before and after)
    ax2.scatter(x_vals, matrix_sizes_orig, c=colors, s=100, alpha=0.7, edgecolors='black',
                linewidth=1.5, label='Original', marker='o')
    ax2.plot(x_vals, matrix_sizes_orig, 'k--', alpha=0.3, linewidth=1)
    
    ax2.scatter(x_vals, matrix_sizes, c='blue', s=80, alpha=0.7, edgecolors='black',
                linewidth=1.5, label='After conditioning', marker='s')
    ax2.plot(x_vals, matrix_sizes, 'b:', alpha=0.3, linewidth=1)
    
    ax2.set_ylabel('Matrix Size', fontsize=12, fontweight='bold')
    ax2.set_title('Matrix Expansion Due to Conditioning', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    # Annotate where expansion occurred
    for i in range(len(matrix_sizes)):
        if matrix_sizes[i] != matrix_sizes_orig[i]:
            expansion = matrix_sizes[i] / matrix_sizes_orig[i]
            ax2.annotate(f'{expansion:.1f}× expansion',
                        xy=(x_vals[i], matrix_sizes[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Plot 3: Circuit qubits
    ax3.scatter(x_vals, circuit_qubits, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.plot(x_vals, circuit_qubits, 'k--', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax3.set_ylabel('Circuit Qubits', fontsize=12, fontweight='bold')
    ax3.set_title('Quantum Resource Requirements', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Annotate qubit changes
    for i in range(1, len(circuit_qubits)):
        if circuit_qubits[i] != circuit_qubits[i-1]:
            ax3.annotate(f'{circuit_qubits[i]} qubits',
                        xy=(x_vals[i], circuit_qubits[i]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {output_path}")
    
    if show_plot:
        import subprocess
        subprocess.Popen(['eog', output_path])
    
    return output_path


def main():
    """Main analysis workflow."""
    if len(sys.argv) < 2:
        print("Usage: python plot_condition_number.py <workflow_dir>")
        print("\nExample:")
        print("  python plot_condition_number.py /tmp/lwfm/qt02-cfd/abc123")
        sys.exit(1)
    
    workflow_dir = sys.argv[1]
    
    print(f"Loading results from {workflow_dir}...")
    case_data = load_results(workflow_dir)
    print(f"Loaded {len(case_data)} cases")
    
    print("Extracting condition number data...")
    data = extract_condition_data(case_data)
    print(f"Found {len(data)} cases with condition data")
    
    if not data:
        print("Error: No condition number data found")
        sys.exit(1)
    
    # Print summary
    print("\nCondition Number Summary:")
    print("-" * 80)
    for d in data:
        status = "✓" if d['kappa_original'] < 1e4 else "✗"
        expansion = d['matrix_size'] / d['matrix_size_original']
        print(f"{status} {d['case_label']}:")
        print(f"    Original:    κ={d['kappa_original']:8.1f}, matrix={d['matrix_size_original']:3d}")
        print(f"    Conditioned: κ={d['kappa']:8.1f}, matrix={d['matrix_size']:3d} ({expansion:.1f}×), qubits={d['circuit_qubits']:2d}")
    
    # Create plot
    output_path = Path(workflow_dir) / "condition_analysis.png"
    plot_condition_analysis(data, str(output_path), show_plot=True)


if __name__ == '__main__':
    main()
