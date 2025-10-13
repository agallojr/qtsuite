"""
Plotting functions for scaling analysis.

Shows how matrix size and circuit qubits scale with input parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_scaling_analysis(case_data, output_path, show_plot=False):
    """
    Plot scaling relationships: NQ_MATRIX, nx, ny vs matrix_size and circuit_qubits.
    
    Parameters
    ----------
    case_data : list
        List of case dictionaries with params and metadata
    output_path : str
        Path to save the plot
    show_plot : bool
        Whether to display the plot
        
    Returns
    -------
    str
        Path to saved plot
    """
    # Extract data
    cases = []
    for case in case_data:
        params = case['params']
        cases.append({
            'case_id': case['case_id'],
            'NQ_MATRIX': params.get('NQ_MATRIX', 0),
            'nx': params.get('nx', 0),
            'ny': params.get('ny', 0),
            'grid_size': params.get('nx', 0) * params.get('ny', 0),
            'matrix_size': params.get('_matrix_size', 0),
            'circuit_qubits': params.get('_circuit_qubits', 0),
            'circuit_depth': params.get('_circuit_depth', 0),
            'circuit_depth_transpiled': params.get('_circuit_depth_transpiled', 0)
        })
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Matrix size vs grid size (nx*ny)
    grid_sizes = [c['grid_size'] for c in cases]
    matrix_sizes = [c['matrix_size'] for c in cases]
    nq_values = [c['NQ_MATRIX'] for c in cases]
    
    # Color by NQ_MATRIX
    unique_nq = sorted(set(nq_values))
    colors = plt.cm.tab10(range(len(unique_nq)))
    nq_color_map = {nq: colors[i] for i, nq in enumerate(unique_nq)}
    
    for nq in unique_nq:
        mask = [c['NQ_MATRIX'] == nq for c in cases]
        gs = [cases[i]['grid_size'] for i, m in enumerate(mask) if m]
        ms = [cases[i]['matrix_size'] for i, m in enumerate(mask) if m]
        ax1.scatter(gs, ms, s=100, alpha=0.7, color=nq_color_map[nq], 
                   label=f'NQ_MATRIX={nq}')
    
    ax1.set_xlabel('Grid Size (nx × ny)', fontsize=11)
    ax1.set_ylabel('Matrix Size', fontsize=11)
    ax1.set_title('A Matrix Size vs Grid Size', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Circuit qubits vs grid size
    for nq in unique_nq:
        mask = [c['NQ_MATRIX'] == nq for c in cases]
        gs = [cases[i]['grid_size'] for i, m in enumerate(mask) if m]
        cq = [cases[i]['circuit_qubits'] for i, m in enumerate(mask) if m]
        ax2.scatter(gs, cq, s=100, alpha=0.7, color=nq_color_map[nq],
                   label=f'NQ_MATRIX={nq}')
    
    ax2.set_xlabel('Grid Size (nx × ny)', fontsize=11)
    ax2.set_ylabel('Circuit Qubits', fontsize=11)
    ax2.set_title('Total Qubits vs Grid Size', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Circuit qubits vs NQ_MATRIX (for fixed grid sizes)
    unique_grids = sorted(set(grid_sizes))
    grid_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_grids)))
    grid_color_map = {gs: grid_colors[i] for i, gs in enumerate(unique_grids)}
    
    for gs in unique_grids:
        mask = [c['grid_size'] == gs for c in cases]
        nq = [cases[i]['NQ_MATRIX'] for i, m in enumerate(mask) if m]
        cq = [cases[i]['circuit_qubits'] for i, m in enumerate(mask) if m]
        if nq and cq:
            ax3.plot(nq, cq, 'o-', markersize=8, linewidth=2, alpha=0.7,
                    color=grid_color_map[gs], label=f'Grid {int(np.sqrt(gs))}×{int(np.sqrt(gs))}')
    
    ax3.set_xlabel('NQ_MATRIX', fontsize=11)
    ax3.set_ylabel('Circuit Qubits', fontsize=11)
    ax3.set_title('Qubits vs NQ_MATRIX (by grid size)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Transpiled circuit depth vs total qubits
    for nq in unique_nq:
        mask = [c['NQ_MATRIX'] == nq for c in cases]
        cq = [cases[i]['circuit_qubits'] for i, m in enumerate(mask) if m]
        cd_trans = [cases[i]['circuit_depth_transpiled'] for i, m in enumerate(mask) if m]
        ax4.scatter(cq, cd_trans, s=100, alpha=0.7, color=nq_color_map[nq],
                   label=f'NQ_MATRIX={nq}')
    
    ax4.set_xlabel('Circuit Qubits', fontsize=11)
    ax4.set_ylabel('Transpiled Circuit Depth', fontsize=11)
    ax4.set_title('Transpiled Depth vs Qubits', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Quantum Circuit Scaling Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if show_plot:
        import subprocess
        subprocess.Popen(['eog', output_path])
    
    return output_path


def plot_scaling_table(case_data, output_path):
    """
    Create a table showing the scaling relationships.
    
    Parameters
    ----------
    case_data : list
        List of case dictionaries
    output_path : str
        Path to save the table image
        
    Returns
    -------
    str
        Path to saved table
    """
    # Extract and sort data
    table_data = []
    for case in case_data:
        params = case['params']
        table_data.append([
            case['case_id'],
            params.get('NQ_MATRIX', 0),
            params.get('nx', 0),
            params.get('ny', 0),
            params.get('_matrix_size', 0),
            params.get('_circuit_qubits', 0),
            params.get('_circuit_depth', 0),
            params.get('_circuit_depth_transpiled', 0)
        ])
    
    # Sort by NQ_MATRIX, then grid size
    table_data.sort(key=lambda x: (x[1], x[2] * x[3]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Case', 'NQ_MATRIX', 'nx', 'ny', 'Matrix Size', 'Circuit Qubits', 'Depth (Logical)', 'Depth (Transpiled)']
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.18, 0.10, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Scaling Analysis: Input Parameters vs Circuit Properties', 
             fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path
