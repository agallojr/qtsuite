#!/usr/bin/env python3
"""
Uncertainty Quantification analysis for QLSA results.

This script analyzes multiple workflow runs to calculate and visualize
uncertainty metrics across parameter sweeps.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats


def load_workflow_results(workflow_dirs):
    """
    Load results from multiple workflow runs.

    Parameters
    ----------
    workflow_dirs : list of str
        Paths to workflow directories containing saved results

    Returns
    -------
    list
        Combined case_data from all workflows
    """
    all_case_data = []

    for wf_dir in workflow_dirs:
        results_file = Path(wf_dir) / "results.pkl"
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping")
            continue

        with open(results_file, 'rb') as f:
            case_data = pickle.load(f)
            all_case_data.extend(case_data)

    return all_case_data


def calculate_fidelity(classical_sol, quantum_sol):
    """Calculate fidelity between classical and quantum solutions."""
    classical_norm = np.linalg.norm(classical_sol)
    quantum_norm = np.linalg.norm(quantum_sol)

    if classical_norm > 0 and quantum_norm > 0:
        classical_normalized = classical_sol / classical_norm
        quantum_normalized = quantum_sol / quantum_norm
        inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
        fidelity = inner_product ** 2
    else:
        fidelity = 0.0

    return fidelity


def group_cases_for_uq(case_data):
    """
    Group cases by their parameter configuration for UQ analysis.

    Parameters
    ----------
    case_data : list
        List of case dictionaries

    Returns
    -------
    dict
        Grouped cases by parameter configuration
    """
    # Parameters to exclude from grouping (vary per run but not meaningful for UQ)
    exclude_params = {'savedir', 'circuit_hhl_path'}
    
    grouped = defaultdict(list)

    for case in case_data:
        params = case['params']
        metadata = case['metadata']

        # Create key from input parameters only (exclude outputs starting with _)
        param_key = tuple(
            (k, v) for k, v in sorted(params.items())
            if k != '_metadata' and k not in exclude_params and not k.startswith('_')
        )

        grouped[param_key].append(case)

    return grouped


def calculate_uq_metrics(grouped_cases):
    """
    Calculate UQ metrics for each parameter group.

    Parameters
    ----------
    grouped_cases : dict
        Cases grouped by parameter configuration

    Returns
    -------
    dict
        UQ metrics for each group
    """
    uq_results = {}

    for param_key, cases in grouped_cases.items():
        fidelities = []

        for case in cases:
            fidelity = calculate_fidelity(
                case['classical_solution'],
                case['quantum_solution']
            )
            fidelities.append(fidelity)

        fidelities = np.array(fidelities)

        # Calculate statistics
        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities, ddof=1) if len(fidelities) > 1 else 0.0
        n = len(fidelities)

        # 95% confidence interval
        if n > 1:
            ci = stats.t.interval(
                0.95,
                n - 1,
                loc=mean_fid,
                scale=stats.sem(fidelities)
            )
        else:
            ci = (mean_fid, mean_fid)

        uq_results[param_key] = {
            'mean': mean_fid,
            'std': std_fid,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_samples': n,
            'cv': std_fid / mean_fid if mean_fid > 0 else 0.0,
            'params': dict(param_key)
        }

    return uq_results


def plot_uq_results(uq_results, x_param, output_path, show_plot=False):
    """
    Plot UQ results with error bars and confidence intervals.

    Parameters
    ----------
    uq_results : dict
        UQ metrics from calculate_uq_metrics
    x_param : str
        Parameter to use for x-axis
    output_path : str
        Path to save plot
    show_plot : bool
        Whether to display plot
    """
    # Group by series (cases with same non-x parameters)
    series_dict = defaultdict(list)

    for param_key, metrics in uq_results.items():
        params = metrics['params']

        if x_param not in params:
            continue

        # Create series key from non-x parameters
        series_key = tuple(
            (k, v) for k, v in sorted(params.items())
            if k != x_param
        )

        series_dict[series_key].append({
            'x': params[x_param],
            'mean': metrics['mean'],
            'std': metrics['std'],
            'ci_lower': metrics['ci_lower'],
            'ci_upper': metrics['ci_upper'],
            'n': metrics['n_samples']
        })

    # Sort each series by x value
    for series_key in series_dict:
        series_dict[series_key].sort(key=lambda d: d['x'])

    # Check if we have meaningful variance
    all_stds = [d['std'] for data in series_dict.values() for d in data]
    has_variance = any(std > 1e-6 for std in all_stds)

    # Create plot - single or dual panel based on variance
    if has_variance:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None

    # Get colormap
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(series_dict))]

    # Determine which parameters vary across series for smart labeling
    # Exclude common infrastructure parameters
    exclude_params = {
        'savedir', 'circuit_hhl_path', 'preprocess_site', 'exec_site',
        'warmup_sites', 'show_plot', 'sim_custom_noise', 'qc_backend',
        'P_in', 'P_out', 'U_top', 'U_bottom', 'mu', 'rho', 'case', 'var'
    }
    
    all_params = [dict(series_key) for series_key in series_dict.keys()]
    if len(all_params) > 1:
        varying_params = {
            k for k in all_params[0].keys()
            if k not in exclude_params and
            not all(p.get(k) == all_params[0][k] for p in all_params)
        }
    else:
        # For single series, show key physics params
        varying_params = {
            k for k in all_params[0].keys()
            if k in {'nx', 'ny', 'NQ_MATRIX', 'L', 'D'}
        } if all_params else set()

    # Plot 1: Mean with confidence intervals
    for (series_key, data), color in zip(series_dict.items(), colors):
        x_vals = [d['x'] for d in data]
        means = [d['mean'] for d in data]
        ci_lower = [d['ci_lower'] for d in data]
        ci_upper = [d['ci_upper'] for d in data]

        # Create series label - only show varying parameters
        if series_key and varying_params:
            label = ', '.join(f"{k}={v}" for k, v in series_key if k in varying_params)
        elif series_key:
            label = ', '.join(f"{k}={v}" for k, v in list(series_key)[:2])  # Max 2 params
        else:
            label = "Mean ± 95% CI"

        # Plot mean line with markers
        ax1.plot(x_vals, means, '-', linewidth=1.5, color=color, alpha=0.7,
                label=label)
        ax1.plot(x_vals, means, 'o', markersize=6, color=color, alpha=0.9)

        # Plot confidence interval as shaded region
        ax1.fill_between(x_vals, ci_lower, ci_upper, alpha=0.15, color=color)

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(x_param, fontsize=12)
    ax1.set_ylabel('Mean Fidelity', fontsize=12)
    
    # Create informative title
    if len(series_dict) > 5:
        title = f'QLSA Fidelity with 95% CI ({len(series_dict)} parameter combinations)'
    else:
        title = 'QLSA Fidelity with 95% Confidence Intervals'
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Only show legend if reasonable number of series
    if len(series_dict) <= 5:
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)

    # Determine if log scale is appropriate
    all_x = [d['x'] for data in series_dict.values() for d in data]
    if len(all_x) > 1 and max(all_x) / min(all_x) > 10:
        ax1.set_xscale('log')
        if ax2:
            ax2.set_xscale('log')

    # Plot 2: Standard deviation (only if variance exists)
    if ax2:
        for (series_key, data), color in zip(series_dict.items(), colors):
            x_vals = [d['x'] for d in data]
            stds = [d['std'] for d in data]

            # Use same label logic as plot 1
            if series_key and varying_params:
                label = ', '.join(f"{k}={v}" for k, v in series_key if k in varying_params)
            elif series_key:
                label = ', '.join(f"{k}={v}" for k, v in list(series_key)[:2])
            else:
                label = "Std Dev"

            ax2.plot(x_vals, stds, 'o-', linewidth=2, markersize=8,
                    label=label, color=color)

        ax2.set_xlabel(x_param, fontsize=12)
        ax2.set_ylabel('Standard Deviation', fontsize=12)
        ax2.set_title('Fidelity Uncertainty', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Only show legend if reasonable number of series
        if len(series_dict) <= 5:
            ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    else:
        # Add note about deterministic results
        ax1.text(0.02, 0.02, 'Note: Deterministic simulation (zero variance)',
                transform=ax1.transAxes, fontsize=9, alpha=0.6,
                verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        import subprocess
        subprocess.Popen(['eog', output_path])

    return output_path


def main():
    """Main UQ analysis workflow."""
    if len(sys.argv) < 2:
        print("Usage: python uq_analysis.py [--case CASE_ID] <workflow_dir1> [workflow_dir2] ...")
        print("\nExample:")
        print("  python uq_analysis.py /tmp/lwfm/qt02-cfd/wf123 /tmp/lwfm/qt02-cfd/wf124")
        print("  python uq_analysis.py --case LD3 /tmp/lwfm/qt02-cfd/wf123 /tmp/lwfm/qt02-cfd/wf124")
        sys.exit(1)

    # Parse arguments
    case_filter = None
    workflow_dirs = []
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--case':
            case_filter = sys.argv[i + 1]
            i += 2
        else:
            workflow_dirs.append(sys.argv[i])
            i += 1
    
    if not workflow_dirs:
        print("Error: No workflow directories specified")
        sys.exit(1)

    print(f"Loading results from {len(workflow_dirs)} workflow(s)...")
    case_data = load_workflow_results(workflow_dirs)

    if not case_data:
        print("Error: No case data loaded")
        sys.exit(1)

    print(f"Loaded {len(case_data)} cases total")

    # Filter by case if specified
    if case_filter:
        original_count = len(case_data)
        case_data = [
            c for c in case_data
            if c['metadata'].get('_original_case_id', '').startswith(case_filter)
        ]
        print(f"Filtered to {len(case_data)} cases matching '{case_filter}' (from {original_count})")
        
        if not case_data:
            print(f"Error: No cases found matching '{case_filter}'")
            sys.exit(1)

    # Group cases by parameter configuration
    print("Grouping cases for UQ analysis...")
    grouped_cases = group_cases_for_uq(case_data)
    print(f"Found {len(grouped_cases)} unique parameter configurations")

    # Calculate UQ metrics
    print("Calculating UQ metrics...")
    uq_results = calculate_uq_metrics(grouped_cases)

    # Print summary
    print("\nUQ Summary:")
    print("-" * 80)
    for param_key, metrics in sorted(uq_results.items()):
        params_str = ', '.join(f"{k}={v}" for k, v in metrics['params'].items())
        print(f"{params_str}")
        print(f"  Mean: {metrics['mean']:.6f}")
        print(f"  Std:  {metrics['std']:.6f}")
        print(f"  95% CI: [{metrics['ci_lower']:.6f}, {metrics['ci_upper']:.6f}]")
        print(f"  CV:   {metrics['cv']:.4f}")
        print(f"  N:    {metrics['n_samples']}")
        print()

    # Determine x-axis parameter (first list param from metadata)
    first_case = case_data[0]
    metadata = first_case['metadata']
    list_params = metadata.get('_list_params', [])

    if list_params:
        x_param = list_params[0]
        print(f"Creating UQ plot with x-axis: {x_param}")

        output_path = Path(workflow_dirs[0]).parent / "uq_analysis.png"
        plot_uq_results(uq_results, x_param, str(output_path), show_plot=True)
        print(f"Saved UQ plot to: {output_path}")
    else:
        print("Warning: No list parameters found, skipping plot")


if __name__ == '__main__':
    main()
