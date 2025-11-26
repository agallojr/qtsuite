"""
Plotting functions for QLSA results visualization.
"""

import subprocess
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def plot_qlsa_results(
    classical_solution: np.ndarray,
    quantum_results: list,
    case_labels: list[str],
    shot_counts: list[int],
    output_path: str,
    show_plot: bool = False) -> str:
    """
    Plot fidelity vs shots for QLSA convergence study (single matrix size).
    
    Parameters
    ----------
    classical_solution : np.ndarray
        Classical solution vector (same for all cases)
    quantum_results : list
        List of quantum solution vectors (one per case)
    case_labels : list[str]
        Labels for each quantum case (e.g., shot counts)
    shot_counts : list[int]
        Number of shots for each case
    output_path : str
        Path to save the plot
    show_plot : bool
        Whether to open the plot after saving
        
    Returns
    -------
    str
        Path to the saved plot file
    """

    # Calculate fidelity for each quantum result
    fidelities = []
    for i, qresult in enumerate(quantum_results):
        print(f"\n=== Fidelity calculation for case {i+1} ===")
        print(f"Classical solution: {classical_solution}")
        print(f"Quantum result: {qresult}")

        # For quantum linear systems, fidelity is often measured as the squared overlap
        # between normalized classical and quantum solution vectors
        classical_norm = np.linalg.norm(classical_solution)
        quantum_norm = np.linalg.norm(qresult)

        print(f"Classical norm: {classical_norm:.6f}")
        print(f"Quantum norm: {quantum_norm:.6f}")

        if classical_norm > 0 and quantum_norm > 0:
            # Normalize both vectors
            classical_normalized = classical_solution / classical_norm
            quantum_normalized = qresult / quantum_norm

            print(f"Classical normalized: {classical_normalized}")
            print(f"Quantum normalized: {quantum_normalized}")

            # Fidelity = |<classical_normalized|quantum_normalized>|^2
            inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
            fidelity = inner_product ** 2
            print(f"Inner product: {inner_product:.6f}")
            print(f"Fidelity: {fidelity:.6f}")
        else:
            fidelity = 0.0
            print("Zero norm detected, fidelity = 0.0")

        fidelities.append(fidelity)

    # Create figure with single plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot fidelity vs shots
    ax.semilogx(shot_counts, fidelities, 'o-', linewidth=0, markersize=8,
                color='#2E8B57', markerfacecolor='#FF6B6B', markeredgecolor='#2E8B57')

    # Add horizontal line at fidelity = 1 (perfect match)
    ax.axhline(y=1.0, color='#BF5700', linestyle='--', alpha=0.7,
               label='Perfect Fidelity')

    # Add fidelity values and case labels as text annotations
    for i, (shots, fidelity, label) in enumerate(zip(shot_counts, fidelities, case_labels)):
        ax.annotate(label,
                   (shots, fidelity),
                   textcoords="offset points",
                   xytext=(-12, 10),
                   ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('QLSA Fidelity Convergence', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fancybox=True,
        shadow=True)

    # Set y-axis limits to focus on high fidelity range (0.2-1.0)
    min_fidelity = min(fidelities)
    y_min = min_fidelity - 0.02    # slightly below min fidelity
    ax.set_ylim(y_min, 1.02)       # slightly above max fidelity

    # Format x-axis to show shot counts clearly
    ax.set_xlim(min(shot_counts) * 0.8, max(shot_counts) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        subprocess.Popen(['open', output_path])  # macOS

    return output_path


def plot_qlsa_multi_matrix(
    classical_solutions: list,
    quantum_results: list,
    case_labels: list[str],
    x_values: list[int],
    output_path: str,
    x_label: str = "Matrix Size",
    show_plot: bool = False) -> str:
    """
    Plot fidelity for QLSA with different matrix sizes or parameters.
    
    Parameters
    ----------
    classical_solutions : list
        List of classical solution vectors (one per case, potentially different dimensions)
    quantum_results : list
        List of quantum solution vectors (one per case)
    case_labels : list[str]
        Labels for each case
    x_values : list[int]
        X-axis values (e.g., matrix sizes or shot counts)
    output_path : str
        Path to save the plot
    x_label : str
        Label for x-axis
    show_plot : bool
        Whether to open the plot after saving
        
    Returns
    -------
    str
        Path to the saved plot file
    """

    # Calculate fidelity for each quantum result
    fidelities = []
    for i, (classical_solution, qresult) in enumerate(zip(classical_solutions, quantum_results)):
        print(f"\n=== Fidelity calculation for case {i+1} ===")
        print(f"Classical solution: {classical_solution}")
        print(f"Quantum result: {qresult}")

        # For quantum linear systems, fidelity is often measured as the squared overlap
        # between normalized classical and quantum solution vectors
        classical_norm = np.linalg.norm(classical_solution)
        quantum_norm = np.linalg.norm(qresult)

        print(f"Classical norm: {classical_norm:.6f}")
        print(f"Quantum norm: {quantum_norm:.6f}")

        if classical_norm > 0 and quantum_norm > 0:
            # Normalize both vectors
            classical_normalized = classical_solution / classical_norm
            quantum_normalized = qresult / quantum_norm

            print(f"Classical normalized: {classical_normalized}")
            print(f"Quantum normalized: {quantum_normalized}")

            # Fidelity = |<classical_normalized|quantum_normalized>|^2
            inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
            fidelity = inner_product ** 2
            print(f"Inner product: {inner_product:.6f}")
            print(f"Fidelity: {fidelity:.6f}")
        else:
            fidelity = 0.0
            print("Zero norm detected, fidelity = 0.0")

        fidelities.append(fidelity)

    # Create figure with single plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot fidelity vs x_values (could be matrix size, shots, etc.)
    # Use log scale if x_values span multiple orders of magnitude
    if max(x_values) / min(x_values) > 10:
        ax.semilogx(x_values, fidelities, 'o-', linewidth=2, markersize=8,
                    color='#2E8B57', markerfacecolor='#FF6B6B', markeredgecolor='#2E8B57')
    else:
        ax.plot(x_values, fidelities, 'o-', linewidth=2, markersize=8,
                color='#2E8B57', markerfacecolor='#FF6B6B', markeredgecolor='#2E8B57')

    # Add horizontal line at fidelity = 1 (perfect match)
    ax.axhline(y=1.0, color='#BF5700', linestyle='--', alpha=0.7,
               label='Perfect Fidelity')

    # Add fidelity values and case labels as text annotations
    for i, (x_val, fidelity, label) in enumerate(zip(x_values, fidelities, case_labels)):
        ax.annotate(label,
                   (x_val, fidelity),
                   textcoords="offset points",
                   xytext=(-12, 10),
                   ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('QLSA Fidelity Analysis', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fancybox=True,
        shadow=True)

    # Set y-axis limits to focus on high fidelity range
    if fidelities:
        min_fidelity = min(fidelities)
        y_min = max(0, min_fidelity - 0.05)  # slightly below min fidelity, but not below 0
        ax.set_ylim(y_min, 1.02)  # slightly above max fidelity

    # Format x-axis
    if len(x_values) > 1:
        x_range = max(x_values) - min(x_values)
        ax.set_xlim(min(x_values) - 0.1 * x_range, max(x_values) + 0.1 * x_range)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        subprocess.Popen(['eog', output_path])  # macOS

    return output_path


def plot_qlsa_generic(
    case_data: list,
    output_path: str,
    show_plot: bool = False) -> str:
    """
    Generic plotting function that automatically detects series based on metadata.
    
    Parameters
    ----------
    case_data : list
        List of dicts, each containing:
        - 'case_id': str
        - 'params': dict (with '_metadata' key)
        - 'classical_solution': np.ndarray
        - 'quantum_solution': np.ndarray
    output_path : str
        Path to save the plot
    show_plot : bool
        Whether to open the plot after saving
        
    Returns
    -------
    str
        Path to the saved plot file
    """

    # Extract metadata to determine series structure
    if not case_data:
        raise ValueError("No case data provided")

    # Get list parameters from first case's metadata
    first_metadata = case_data[0]['metadata']
    list_params = first_metadata.get('_list_params', [])

    if not list_params:
        # No list expansion, just plot all points
        return _plot_simple(case_data, output_path, show_plot)

    # Determine x-axis and series grouping
    # Strategy: Use first list param as x-axis, others for series grouping
    x_param = list_params[0]
    series_params = list_params[1:] if len(list_params) > 1 else []

    logger_info = f"Plotting with x-axis='{x_param}'"
    if series_params:
        logger_info += f", series grouped by {series_params}"
    print(logger_info)

    # Group cases into series by original case ID
    series_dict = defaultdict(list)

    for case in case_data:
        metadata = case['metadata']
        original_case_id = metadata.get('_original_case_id', case['case_id'])
        series_dict[original_case_id].append(case)

    # Calculate fidelities and prepare plot data
    series_plot_data = {}

    for series_key, series_cases in series_dict.items():
        # Sort by x-axis parameter
        series_cases.sort(key=lambda c: c['params'][x_param])

        x_values = []
        fidelities = []

        for case in series_cases:
            classical_sol = case['classical_solution']
            quantum_sol = case['quantum_solution']

            # Skip cases with missing solutions
            if classical_sol is None or quantum_sol is None:
                continue

            # Calculate fidelity
            classical_norm = np.linalg.norm(classical_sol)
            quantum_norm = np.linalg.norm(quantum_sol)

            if classical_norm > 0 and quantum_norm > 0:
                classical_normalized = classical_sol / classical_norm
                quantum_normalized = quantum_sol / quantum_norm
                inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
                fidelity = inner_product ** 2
            else:
                fidelity = 0.0

            x_values.append(case['params'][x_param])
            fidelities.append(fidelity)

        # Use case ID as series label
        series_label = series_key

        series_plot_data[series_key] = {
            'x_values': x_values,
            'fidelities': fidelities,
            'label': series_label
        }

    # Create plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot each series
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(series_plot_data))]

    for (series_key, data), color in zip(series_plot_data.items(), colors):
        x_vals = data['x_values']
        y_vals = data['fidelities']
        label = data['label']

        # Determine if log scale is appropriate
        if len(x_vals) > 1 and max(x_vals) / min(x_vals) > 10:
            ax.semilogx(x_vals, y_vals, 'o-', linewidth=2, markersize=8,
                       label=label, color=color)
        else:
            ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8,
                   label=label, color=color)

    # Add horizontal line at fidelity = 1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Fidelity')

    # Labels and formatting
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('QLSA Fidelity Analysis', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set y-axis limits
    all_fidelities = [f for data in series_plot_data.values() for f in data['fidelities']]
    if all_fidelities:
        min_fid = min(all_fidelities)
        y_min = max(0, min_fid - 0.05)
        ax.set_ylim(y_min, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        subprocess.Popen(['eog', output_path])

    return output_path


def _plot_simple(case_data, output_path, show_plot):
    """
    Simple plot for cases without list expansion.
    """
    fidelities = []
    labels = []

    for case in case_data:
        classical_sol = case['classical_solution']
        quantum_sol = case['quantum_solution']

        classical_norm = np.linalg.norm(classical_sol)
        quantum_norm = np.linalg.norm(quantum_sol)

        if classical_norm > 0 and quantum_norm > 0:
            classical_normalized = classical_sol / classical_norm
            quantum_normalized = quantum_sol / quantum_norm
            inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
            fidelity = inner_product ** 2
        else:
            fidelity = 0.0

        fidelities.append(fidelity)
        labels.append(case['case_id'])

    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_pos = range(len(fidelities))
    ax.bar(x_pos, fidelities, color='#2E8B57', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('QLSA Fidelity Results', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        subprocess.Popen(['eog', output_path])

    return output_path
