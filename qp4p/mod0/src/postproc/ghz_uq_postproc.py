"""
Postprocessing script for GHZ uncertainty quantification.

Extracts count information from multiple runs and plots with confidence ranges.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_case_results(case_dirs):
    """
    Load results from all case directories.
    
    Returns:
        List of dicts with counts and metadata
    """
    results = []
    for case_dir in case_dirs:
        case_path = Path(case_dir)
        stdout_file = case_path / "stdout.json"
        
        if not stdout_file.exists():
            continue
        
        try:
            with open(stdout_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'case_dir': case_dir,
                    'counts': data['results']['counts'],
                    'config': data.get('config', {}),
                    't1': data['config'].get('t1'),
                    't2': data['config'].get('t2')
                })
        except Exception as e:
            print(f"Warning: Could not load {stdout_file}: {e}", file=sys.stderr)
    
    return results


def compute_statistics(results):
    """
    Compute GHZ fidelity statistics focusing on valid outcomes (all 0s or all 1s).
    
    Returns:
        Dict with statistics for valid states, invalid states, and fidelity
    """
    from scipy import stats as sp_stats
    
    # Determine number of qubits from first result
    first_counts = results[0]['counts']
    n_qubits = len(list(first_counts.keys())[0])
    
    # Valid GHZ states are all 0s or all 1s
    valid_state_0 = '0' * n_qubits
    valid_state_1 = '1' * n_qubits
    
    # Collect counts for valid and invalid states across all runs
    valid_0_counts = []
    valid_1_counts = []
    invalid_counts = []
    total_shots = []
    fidelities = []
    
    for r in results:
        counts = r['counts']
        total = sum(counts.values())
        
        count_0 = counts.get(valid_state_0, 0)
        count_1 = counts.get(valid_state_1, 0)
        count_valid = count_0 + count_1
        count_invalid = total - count_valid
        
        valid_0_counts.append(count_0)
        valid_1_counts.append(count_1)
        invalid_counts.append(count_invalid)
        total_shots.append(total)
        fidelities.append(count_valid / total if total > 0 else 0)
    
    def compute_ci(data):
        """Compute mean, std, and 95% CI for a dataset."""
        mean = np.mean(data)
        std = np.std(data, ddof=1) if len(data) > 1 else 0
        n = len(data)
        if n > 1:
            ci = sp_stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            ci_lower, ci_upper = ci
        else:
            ci_lower = ci_upper = mean
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': data
        }
    
    stats = {
        'n_qubits': n_qubits,
        'valid_state_0': valid_state_0,
        'valid_state_1': valid_state_1,
        'counts_all_0': compute_ci(valid_0_counts),
        'counts_all_1': compute_ci(valid_1_counts),
        'counts_invalid': compute_ci(invalid_counts),
        'fidelity': compute_ci(fidelities),
        'total_shots': total_shots[0] if len(set(total_shots)) == 1 else total_shots
    }
    
    return stats


def plot_results(stats, output_file):
    """
    Plot GHZ fidelity and valid state counts with error bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Valid state counts
    categories = ['All 0s', 'All 1s', 'Invalid']
    data_keys = ['counts_all_0', 'counts_all_1', 'counts_invalid']
    means = [stats[k]['mean'] for k in data_keys]
    ci_lower = [stats[k]['ci_lower'] for k in data_keys]
    ci_upper = [stats[k]['ci_upper'] for k in data_keys]
    
    yerr_lower = [means[i] - ci_lower[i] for i in range(len(means))]
    yerr_upper = [ci_upper[i] - means[i] for i in range(len(means))]
    
    x = np.arange(len(categories))
    colors = ['green', 'blue', 'red']
    
    ax1.bar(x, means, alpha=0.7, color=colors)
    ax1.errorbar(x, means, yerr=[yerr_lower, yerr_upper], 
                fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    ax1.set_xlabel('Measurement Outcome', fontsize=12)
    ax1.set_ylabel('Counts', fontsize=12)
    ax1.set_title(f'GHZ State ({stats["n_qubits"]} qubits) Measurement Distribution', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, m in enumerate(means):
        ax1.text(i, m + yerr_upper[i] + 5, f'{m:.1f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Right plot: Fidelity
    fid_mean = stats['fidelity']['mean']
    fid_ci_lower = stats['fidelity']['ci_lower']
    fid_ci_upper = stats['fidelity']['ci_upper']
    fid_yerr_lower = fid_mean - fid_ci_lower
    fid_yerr_upper = fid_ci_upper - fid_mean
    
    ax2.bar([0], [fid_mean], alpha=0.7, color='purple', width=0.5)
    ax2.errorbar([0], [fid_mean], yerr=[[fid_yerr_lower], [fid_yerr_upper]], 
                fmt='none', ecolor='black', capsize=10, linewidth=2)
    
    ax2.set_ylabel('Fidelity', fontsize=12)
    ax2.set_title('GHZ State Fidelity with 95% CI', fontsize=13)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([])
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value label
    ax2.text(0, fid_mean + fid_yerr_upper + 0.02, 
            f'{fid_mean:.4f}\n±{stats["fidelity"]["std"]:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")


def main():
    """Main postprocessing function."""
    if len(sys.argv) < 2:
        print("Usage: python ghz_uq_postproc.py <postproc_json>", file=sys.stderr)
        sys.exit(1)
    
    postproc_json = Path(sys.argv[1])
    
    # Load postproc context
    with open(postproc_json, 'r') as f:
        context = json.load(f)
    
    group_id = context['group_id']
    run_dir = Path(context['run_dir'])
    case_dirs = context['case_dirs']
    
    print(f"Processing group: {group_id}")
    print(f"Number of cases: {len(case_dirs)}")
    
    # Load results
    results = load_case_results(case_dirs)
    print(f"Successfully loaded {len(results)} results")
    
    if len(results) == 0:
        print("No results to process", file=sys.stderr)
        sys.exit(1)
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Save statistics to JSON
    stats_file = run_dir / f"{group_id}_statistics.json"
    
    def serialize_stat(data):
        """Convert stat dict to serializable format."""
        if isinstance(data, dict) and 'mean' in data:
            return {
                'mean': float(data['mean']),
                'std': float(data['std']),
                'ci_lower': float(data['ci_lower']),
                'ci_upper': float(data['ci_upper']),
                'samples': [float(x) for x in data['samples']]
            }
        return data
    
    stats_serializable = {
        'n_qubits': stats['n_qubits'],
        'valid_state_0': stats['valid_state_0'],
        'valid_state_1': stats['valid_state_1'],
        'total_shots': stats['total_shots'],
        'counts_all_0': serialize_stat(stats['counts_all_0']),
        'counts_all_1': serialize_stat(stats['counts_all_1']),
        'counts_invalid': serialize_stat(stats['counts_invalid']),
        'fidelity': serialize_stat(stats['fidelity'])
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"Statistics saved to: {stats_file}")
    
    # Plot results
    plot_file = run_dir / f"{group_id}_plot.png"
    plot_results(stats, plot_file)


if __name__ == "__main__":
    main()
