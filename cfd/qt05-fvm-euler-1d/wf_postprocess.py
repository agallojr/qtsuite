from pathlib import Path
from typing import List
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def postprocess(case_dirs: List[Path]) -> None:
    """Post-process results from multiple case directories.
    
    Args:
        case_dirs: List of Path objects pointing to case directories
    """
    for case_dir in case_dirs:
        rollup_file = case_dir / "rollup.json"
        if not rollup_file.exists():
            print(f"Warning: No rollup.json found in {case_dir}")
            continue
        
        # Load rollup data
        with open(rollup_file, 'r', encoding='utf-8') as f:
            rollup = json.load(f)
        
        # Generate HTML report
        html_file = case_dir / "report.html"
        generate_html_report(rollup, html_file, case_dir)
        
        # Print fully qualified path
        print(f"file://{html_file.resolve()}")

def generate_html_report(rollup: dict, output_file: Path, case_dir: Path) -> None:
    """Generate an HTML report from rollup data."""
    
    # Extract case arguments
    case_args = rollup.get('wf_context', {}).get('case_args', {})
    nelem = case_args.get('nelem', 'N/A')
    max_iters = case_args.get('max_iters', 'N/A')
    max_inner_iters = case_args.get('max_inner_iters', 'N/A')
    backend = case_args.get('backend', 'N/A')
    
    # Extract matrix sizes from first iteration
    iterations = rollup.get('solver_log', {}).get('iterations', [])
    matrix_size = 'N/A'
    padded_size = 'N/A'
    herm_size = 'N/A'
    if iterations:
        first_iter = iterations[0]
        matrix_size = first_iter.get('matrix_original_shape', 'N/A')
        padded_size = first_iter.get('matrix_scaled_shape', 'N/A')
        herm_size = first_iter.get('matrix_hermitian_shape', 'N/A')
    
    # Get QPY file info
    qpy_files = rollup.get('qpy_files', [])
    
    # Get CSV data
    csv_data = rollup.get('csv_data', {})
    csv_records = csv_data.get('records', [])
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HHL Solver Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 15px 0;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .info-item {{
            background-color: #34495e;
            padding: 10px;
            border-radius: 3px;
        }}
        .info-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        .info-value {{
            font-size: 18px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .chart-container {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-image {{
            width: 100%;
            max-width: 1000px;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HHL Quantum Solver Report</h1>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Number of Elements</div>
                <div class="info-value">{nelem}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Max Iterations</div>
                <div class="info-value">{max_iters}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Max Inner Iterations</div>
                <div class="info-value">{max_inner_iters}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Matrix Size</div>
                <div class="info-value">{matrix_size}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Padded Size</div>
                <div class="info-value">{padded_size}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Hermitian Size</div>
                <div class="info-value">{herm_size}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Backend</div>
                <div class="info-value">{backend}</div>
            </div>
        </div>
    </div>
"""
    
    # Generate charts as PNG files
    if iterations:
        generate_iteration_charts_png(iterations, qpy_files, case_dir)
        html += """
    <div class="chart-container">
        <h2>Iteration Metrics</h2>
        <img src="condition_chart.png" class="chart-image" alt="Condition Numbers and Fidelity">
    </div>
"""
        html += generate_circuit_table(iterations, qpy_files)
    
    # Add CSV data table and chart
    if csv_records:
        generate_csv_chart_png(csv_records, case_dir)
        html += generate_csv_table(csv_records)
    
    html += """
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def generate_iteration_charts_png(iterations: list, qpy_files: list, case_dir: Path) -> None:
    """Generate PNG charts for iteration metrics using matplotlib."""
    
    # Extract data for charts
    iter_nums = list(range(len(iterations)))
    matrix_conds = [it.get('matrix_original_condition_number') for it in iterations]
    padded_conds = [it.get('matrix_padded_condition_number') for it in iterations]
    herm_conds = [it.get('matrix_hermitian_condition_number') for it in iterations]
    qubits = [it.get('circuit_qubits') for it in iterations]
    gates = [it.get('circuit_gates') for it in iterations]
    depths = [it.get('circuit_depth') for it in iterations]
    gen_times = [it.get('hhl_circuit_time_sec') for it in iterations]
    trans_times = [it.get('transpile_time_sec') for it in iterations]
    fidelities = [it.get('fidelity') for it in iterations]
    
    # QPY file sizes (match by iteration)
    qpy_sizes = []
    for i in range(len(iterations)):
        matching_qpy = [qpy for qpy in qpy_files if qpy.get('iteration') == i]
        if matching_qpy:
            qpy_sizes.append(matching_qpy[0].get('size_bytes', 0) / 1024 / 1024)  # Convert to MB
        else:
            qpy_sizes.append(0)
    
    # Condition number and fidelity chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot original and hermitian condition numbers
    ax1.plot(iter_nums, matrix_conds, 'o-', label='Original Matrix', linewidth=2, markersize=8, color='#3498db')
    ax1.plot(iter_nums, herm_conds, 's-', label='Hermitian Matrix', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Condition Number', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 1 - Fidelity on right axis
    ax2 = ax1.twinx()
    ax2.plot(iter_nums, fidelities, 'gD-', label='1 - Fidelity', linewidth=2, markersize=8)
    ax2.set_ylabel('Fidelity', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    if fidelities and min(fidelities) is not None:
        ax2.set_ylim([min(fidelities) * 0.9999, 1.0001])
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title('Matrix Condition Numbers and Solution Fidelity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(case_dir / 'condition_chart.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_circuit_table(iterations: list, qpy_files: list) -> str:
    """Generate HTML table for circuit metrics."""
    
    if not iterations:
        return ""
    
    html = """
    <div class="chart-container">
        <h2>Circuit Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Iteration</th>
                    <th>QPY Size (MB)</th>
                    <th>Qubits</th>
                    <th>Gates</th>
                    <th>Depth</th>
                    <th>Build Time (s)</th>
                    <th>Transpile Time (s)</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, iteration in enumerate(iterations):
        # Find matching QPY file
        qpy_size = 0
        matching_qpy = [qpy for qpy in qpy_files if qpy.get('iteration') == i]
        if matching_qpy:
            qpy_size = matching_qpy[0].get('size_bytes', 0) / 1024 / 1024  # Convert to MB
        
        qubits = iteration.get('circuit_qubits', 'N/A')
        gates = iteration.get('circuit_gates', 'N/A')
        depth = iteration.get('circuit_depth', 'N/A')
        build_time = iteration.get('hhl_circuit_time_sec', 'N/A')
        transpile_time = iteration.get('transpile_time_sec', 'N/A')
        
        # Format values
        qpy_size_str = f"{qpy_size:.2f}" if qpy_size > 0 else "N/A"
        build_time_str = f"{build_time:.2f}" if isinstance(build_time, (int, float)) else build_time
        transpile_time_str = f"{transpile_time:.2f}" if isinstance(transpile_time, (int, float)) else transpile_time
        
        html += f"""                <tr>
                    <td>{i}</td>
                    <td>{qpy_size_str}</td>
                    <td>{qubits}</td>
                    <td>{gates}</td>
                    <td>{depth}</td>
                    <td>{build_time_str}</td>
                    <td>{transpile_time_str}</td>
                </tr>
"""
    
    html += """            </tbody>
        </table>
    </div>
"""
    
    return html

def generate_csv_chart_png(csv_records: list, case_dir: Path) -> None:
    """Generate PNG chart for CSV data using matplotlib."""
    
    if not csv_records:
        return
    
    # Get headers from first record
    headers = list(csv_records[0].keys())
    
    # Extract xp and other columns for plotting
    try:
        xp_values = [float(r.get('xp', 0)) for r in csv_records]
        
        plt.figure(figsize=(10, 6))
        
        # Plot each numeric column (skip xp as it's the x-axis)
        for header in headers:
            if header.strip() != 'xp':
                try:
                    y_values = [float(r.get(header, 0)) for r in csv_records]
                    plt.plot(xp_values, y_values, 'o-', label=header.strip(), linewidth=2, markersize=6)
                except (ValueError, TypeError):
                    pass
        
        plt.xlabel('Position (xp)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Results by Position', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(case_dir / 'csv_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

def generate_csv_table(csv_records: list) -> str:
    """Generate HTML table for CSV data."""
    
    if not csv_records:
        return ""
    
    # Get headers from first record
    headers = list(csv_records[0].keys())
    
    # Build table
    html = """
    <div class="chart-container">
        <h2>Simulation Results</h2>
        <table>
            <thead>
                <tr>
"""
    
    for header in headers:
        html += f"                    <th>{header.strip()}</th>\n"
    
    html += """                </tr>
            </thead>
            <tbody>
"""
    
    for record in csv_records:
        html += "                <tr>\n"
        for header in headers:
            value = record.get(header, '')
            # Try to format as float if possible
            try:
                value = f"{float(value):.6f}"
            except (ValueError, TypeError):
                pass
            html += f"                    <td>{value}</td>\n"
        html += "                </tr>\n"
    
    html += """            </tbody>
        </table>
    </div>
    
    <div class="chart-container">
        <h2>Results Visualization</h2>
        <img src="csv_chart.png" class="chart-image" alt="Results Visualization">
    </div>
"""
    
    return html
