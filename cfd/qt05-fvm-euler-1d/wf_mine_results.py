"""
Mining and consolidating results from solver output.
"""

#pylint: disable=broad-exception-caught

from pathlib import Path
import json
import re
import csv
from typing import Dict

def get_qpy_files_info(case_dir: Path) -> list[dict]:
    """Extract information about QPY files in the directory.
    
    Returns a list of dicts with filename, iteration, and size in bytes.
    """
    qpy_files = []
    for qpy_file in case_dir.glob('*.qpy'):
        # Extract iteration number from filename (assumes pattern like '..._iterN.qpy')
        match = re.search(r'iter(\d+)\.qpy$', qpy_file.name)
        iteration = int(match.group(1)) if match else None

        qpy_files.append({
            'filename': qpy_file.name,
            'iteration': iteration,
            'size_bytes': qpy_file.stat().st_size
        })

    # Sort by iteration number if available, otherwise by filename
    return sorted(
        qpy_files,
        key=lambda x: (x['iteration'] is None, x['iteration'] or 0, x['filename'])
    )

def parse_solver_log(log_file: Path) -> Dict:
    """Parse the solver log file and extract key metrics and information.
    
    Args:
        log_file: Path to the solver log file
        
    Returns:
        Dictionary containing parsed log information
    """
    if not log_file.exists():
        return {}

    log_data = {
        'iterations': []  # List to store data for each iteration
    }

    current_section = None
    current_iteration = None
    in_solution_table = False

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Parse Hermitian-specific information FIRST (before section detection)
            if 'Determinant of Hermitian matrix:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if numbers:
                    current_iteration['matrix_hermitian_determinant'] = float(numbers[0])
                continue

            if 'Condition # of Hermitian matrix:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if numbers:
                    current_iteration['matrix_hermitian_condition_number'] = float(numbers[0])
                continue

            if 'Shape of hermitian A:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]+', line)
                if len(numbers) >= 4:
                    current_iteration['matrix_hermitian_shape'] = f"{numbers[2]}x{numbers[3]}"
                continue

            # Parse padded matrix properties (different from scaled)
            if 'Determinant of padded matrix:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if numbers:
                    current_iteration['matrix_padded_determinant'] = float(numbers[0])
                continue

            if 'Condition # of padded matrix:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if numbers:
                    current_iteration['matrix_padded_condition_number'] = float(numbers[0])
                continue

            # Detect new iteration and create iteration data structure
            if 'Original system' in line and '-----' in line:
                current_iteration = {
                    'initial_residual': None,
                    'matrix_original_shape': None,
                    'matrix_original_condition_number': None,
                    'matrix_original_determinant': None,
                    'matrix_scaled_condition_number': None,
                    'matrix_scaled_determinant': None,
                    'matrix_scaled_shape': None,
                    'matrix_padded_condition_number': None,
                    'matrix_padded_determinant': None,
                    'matrix_hermitian_shape': None,
                    'matrix_hermitian_condition_number': None,
                    'matrix_hermitian_determinant': None,
                    'hhl_circuit_time_sec': None,
                    'transpile_time_sec': None,
                    'circuit_qubits': None,
                    'circuit_depth': None,
                    'circuit_gates': None,
                    'fidelity': None,
                    'linear_system_residual': None,
                    'solution_comparison': []
                }
                log_data['iterations'].append(current_iteration)
                current_section = 'matrix_original'  # Set section after creating iteration
                continue
            
            # Capture initial residual for current iteration
            if 'Initial unsteady residual:' in line and current_iteration is not None:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                if numbers:
                    current_iteration['initial_residual'] = float(numbers[0])
                continue

            # Detect solution comparison table start
            if 'Compare system solutions' in line or 'LU sol' in line and 'Herm LU sol' in line:
                in_solution_table = True
                continue
            
            # Detect solution comparison table end (empty line or separator line)
            if in_solution_table and (line == '' or '===' in line or 'Initial unsteady residual' in line):
                in_solution_table = False
                continue

            # Parse iteration-specific timing and metrics
            if current_iteration is not None:
                if 'Time elapsed for generating HHL circuit:' in line:
                    # Extract time in seconds (handle "X min Y sec" format)
                    match = re.search(r'(\d+)\s+min\s+([\d.]+)\s+sec', line)
                    if match:
                        minutes, seconds = float(match.group(1)), float(match.group(2))
                        current_iteration['hhl_circuit_time_sec'] = minutes * 60 + seconds
                    continue

                if 'Time elapsed for transpiling the circuit:' in line:
                    match = re.search(r'(\d+)\s+min\s+([\d.]+)\s+sec', line)
                    if match:
                        minutes, seconds = float(match.group(1)), float(match.group(2))
                        current_iteration['transpile_time_sec'] = minutes * 60 + seconds
                    continue

                if 'Transpiled circuit, num_qubits:' in line:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        current_iteration['circuit_qubits'] = int(numbers[0])
                    continue

                if 'Transpiled circuit depth:' in line:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        current_iteration['circuit_depth'] = int(numbers[0])
                    continue

                if 'Transpiled circuit gates:' in line:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        current_iteration['circuit_gates'] = int(numbers[0])
                    continue

                if 'Fidelity of true vs hhl solutions:' in line:
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if numbers:
                        current_iteration['fidelity'] = float(numbers[0])
                    continue

                if '[HHL info]: linear system residual:' in line:
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if numbers:
                        current_iteration['linear_system_residual'] = float(numbers[0])
                    continue

                # Parse solution comparison table (only when in_solution_table is True)
                if in_solution_table and re.search(r'[-+]?[0-9]*\.?[0-9]+[eE][-+][0-9]+', line):
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if len(numbers) >= 4:
                        current_iteration['solution_comparison'].append({
                            'lu_sol': float(numbers[0]),
                            'herm_lu_sol': float(numbers[1]),
                            'hhl_sol': float(numbers[2]),
                            'l1_diff_pct': float(numbers[3])
                        })
                    continue

            # Detect sections (but skip "Original system" since it's handled above)
            if line.startswith('A info: before HHL prepare:'):
                current_section = 'matrix_original'
                continue
            elif 'A_hhl info: after scaling by spectral radius:' in line:
                current_section = 'matrix_scaled'
                continue
            elif 'padded matrix system' in line.lower():
                current_section = 'matrix_padded'
                continue
            elif 'hermitian matrix' in line.lower() or 'Hermitian matrix' in line:
                current_section = 'matrix_hermitian'
                continue

            # Parse shape information (can appear in any section)
            if '[A]_' in line and 'x' in line and current_iteration is not None:
                # Extract shape from lines like "[A]_6x6:"
                shape_match = re.search(r'\[A\]_(\d+)x(\d+)', line)
                if shape_match:
                    rows, cols = shape_match.groups()
                    current_iteration['matrix_original_shape'] = f"{rows}x{cols}"

            if '[A_padded]_' in line and current_iteration is not None:
                # Extract padded shape from lines like "[A_padded]_8x8 matrix:"
                shape_match = re.search(r'\[A_padded\]_(\d+)x(\d+)', line)
                if shape_match:
                    rows, cols = shape_match.groups()
                    current_iteration['matrix_scaled_shape'] = f"{rows}x{cols}"

            # Parse matrix properties section
            if current_section and current_iteration is not None and \
                ('det(' in line or 'condition number' in line):
                # Extract all numerical values with scientific notation
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)

                # Parse specific values we're interested in
                if 'condition number' in line and numbers:
                    current_iteration[f'{current_section}_condition_number'] = \
                        float(numbers[0])

                if 'det(A):' in line and numbers:
                    current_iteration[f'{current_section}_determinant'] = \
                    float(numbers[0])


    return log_data

def get_csv_data(case_dir: Path) -> tuple[list[dict], str]:
    """Extract data from results*.csv files in the directory.
    
    Returns:
        tuple: (list of dicts where each dict represents a row of data, 
                source filename or empty string if no file found)
    """
    csv_files = list(case_dir.glob('results*.csv'))
    if not csv_files:
        return [], ""

    # For simplicity, we'll use the first matching CSV file
    csv_file = csv_files[0]
    data = []

    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Warning: Could not read CSV file {csv_file}: {e}")
        return [], str(csv_file.name)

    return data, str(csv_file.name)

def mine_results(case_dir: Path) -> None:
    """
    Extract and consolidate results from the solver output into a rollup.json file.
    
    Args:
        case_dir: Path to the case directory containing output files
    """
    rollup = {}

    # Add wf_context if it exists
    wf_context_file = case_dir / "wf_context.json"
    if wf_context_file.exists():
        with open(wf_context_file, 'r', encoding='utf-8') as f:
            rollup['wf_context'] = json.load(f)

    # Add QPY file information
    qpy_files = get_qpy_files_info(case_dir)
    if qpy_files:
        rollup['qpy_files'] = qpy_files

    # Parse solver log if it exists
    log_files = list(case_dir.glob('*solver.log'))
    if log_files:
        log_data = parse_solver_log(log_files[0])
        if log_data:
            rollup['solver_log'] = log_data

    # Add CSV data if available
    csv_data, csv_filename = get_csv_data(case_dir)
    if csv_data and csv_filename:
        rollup['csv_data'] = {
            'source_file': csv_filename,
            'records': csv_data
        }

    # Write the consolidated data to rollup.json
    if rollup:  # Only write if we have data
        rollup_file = case_dir / "rollup.json"
        with open(rollup_file, 'w', encoding='utf-8') as f:
            json.dump(rollup, f, indent=2)
    