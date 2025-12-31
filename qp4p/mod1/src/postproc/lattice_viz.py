#!/usr/bin/env python3
"""
Visualization script for Ising lattice VQE results.

Reads JSON output from lattice.py and generates visualization plots.

Usage:
    python src/postproc/lattice_viz.py <stdout_json_file>
    or
    python -m postproc.lattice_viz <stdout_json_file>
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def create_lattice_visualization(data, output_file=None, display=True):
    """
    Create comprehensive visualization of lattice VQE results.
    
    Args:
        data: Dictionary containing lattice VQE results
        output_file: Optional path to save the plot
        display: Whether to display the plot
    """
    lattice_config = data.get("lattice", {})
    viz_data = data.get("visualization_data", {})
    vqe_results = data.get("vqe_results", {})
    
    rows = lattice_config.get("rows", 0)
    cols = lattice_config.get("cols", 0)
    coupling_matrix = np.array(viz_data.get("coupling_matrix", []))
    interaction_matrix = np.array(viz_data.get("interaction_matrix", []))
    energy_history = viz_data.get("energy_history", [])
    ground_state_spins = viz_data.get("ground_state_spins", [])
    final_energy = vqe_results.get("ground_state_energy", 0.0)
    
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:])
    
    # Create lattice graph
    G = nx.Graph()
    pos = {}
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            pos[node_id] = (j, rows - 1 - i)
            G.add_node(node_id)
    
    graph_real = np.real(interaction_matrix)
    for i in range(graph_real.shape[0]):
        for j in range(i+1, graph_real.shape[1]):
            if abs(graph_real[i, j]) > 0.01:
                G.add_edge(i, j)
    
    # Plot 1: Lattice structure
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_color='lightblue', node_size=400)
    nx.draw_networkx_labels(G, pos, ax=ax0, font_size=7, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax0, edge_color='black', width=2)
    ax0.set_title(f'Square Lattice Structure ({rows}×{cols})', 
                  fontsize=9, fontweight='bold')
    ax0.set_aspect('equal')
    ax0.margins(0.1)
    ax0.axis('off')
    
    # Plot 2: Coupling matrix
    coupling_real = np.real(coupling_matrix)
    im1 = ax1.imshow(coupling_real, cmap='RdBu', aspect='auto')
    ax1.set_title('Coupling Matrix (Real Part)', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Node Index')
    ax1.set_ylabel('Node Index')
    plt.colorbar(im1, ax=ax1, label='Coupling Strength')
    if coupling_real.shape[0] <= 6:
        for i in range(coupling_real.shape[0]):
            for j in range(coupling_real.shape[1]):
                ax1.text(j, i, f'{coupling_real[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=6)
    
    # Plot 3: Interaction graph (adjacency matrix)
    im2 = ax2.imshow(graph_real, cmap='Greys', aspect='auto')
    ax2.set_title('Interaction Graph (Adjacency Matrix)', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Node Index')
    ax2.set_ylabel('Node Index')
    plt.colorbar(im2, ax=ax2, label='Connected (1) / Not Connected (0)')
    if graph_real.shape[0] <= 6:
        for i in range(graph_real.shape[0]):
            for j in range(graph_real.shape[1]):
                ax2.text(j, i, f'{int(graph_real[i, j])}',
                        ha="center", va="center",
                        color="white" if graph_real[i, j] > 0.5 else "black",
                        fontsize=6)
    
    # Plot 4: Ground state spin configuration
    spin_colors = ['red' if s == 1 else 'blue' for s in ground_state_spins]
    nx.draw_networkx_nodes(G, pos, ax=ax3, node_color=spin_colors, node_size=400,
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax3, font_size=7, font_weight='bold',
                           font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax3, edge_color='gray', width=2)
    ax3.set_title('Ground State Spin Configuration', fontsize=9, fontweight='bold')
    ax3.text(0.5, -0.15, 'Red = Spin Up (↑), Blue = Spin Down (↓)',
             transform=ax3.transAxes, ha='center', fontsize=7)
    ax3.set_aspect('equal')
    ax3.margins(0.1)
    ax3.axis('off')
    
    # Plot 5: Energy convergence
    if energy_history:
        ax4.plot(energy_history, 'b-', linewidth=2)
        ax4.axhline(y=final_energy, color='r', linestyle='--',
                    label=f'Final energy: {final_energy:.4f}')
        ax4.set_xlabel('VQE Iteration', fontsize=8)
        ax4.set_ylabel('Energy', fontsize=8)
        ax4.set_title('VQE Energy Convergence', fontsize=9, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    if display:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postproc/lattice_viz.py <postproc_json>")
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
        
        # Generate output filename based on lattice config
        lattice_config = data.get("lattice", {})
        vqe_config = data.get("vqe_config", {})
        rows = lattice_config.get("rows", 0)
        cols = lattice_config.get("cols", 0)
        ansatz = vqe_config.get("ansatz_type", "unknown")
        optimizer = vqe_config.get("optimizer", "unknown")
        
        output_file = case_dir / f'ising_{rows}x{cols}_{ansatz}_{optimizer}.png'
        
        create_lattice_visualization(data, output_file=output_file, display=False)
        print(f"Created visualization: {output_file}")


if __name__ == "__main__":
    main()
