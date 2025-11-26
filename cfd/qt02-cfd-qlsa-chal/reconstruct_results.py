#!/usr/bin/env python3
"""Reconstruct results from completed workflow case directories"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
import re
import ast

if len(sys.argv) < 2:
    print("Usage: python reconstruct_results.py <workflow_id>")
    sys.exit(1)

workflow_id = sys.argv[1]
workflow_dir = Path(f"/Users/agallojr/.lwfm/out/qt02-cfd/{workflow_id}")

if not workflow_dir.exists():
    print(f"Workflow directory not found: {workflow_dir}")
    sys.exit(1)

print(f"Reconstructing results for workflow: {workflow_id}")

# Scan for case directories
case_dirs = sorted([d for d in workflow_dir.iterdir() if d.is_dir() and d.name.startswith('hs_')])
print(f"Found {len(case_dirs)} case directories")

case_data = []

for case_dir in case_dirs:
    case_id = case_dir.name
    
    # Parse case parameters from directory name
    # Format: hs_{backend}_{mesh}__{shot_index}
    parts = case_id.split('_')
    if len(parts) >= 4:
        backend_part = '_'.join(parts[1:-2])  # sv or brisbane
        mesh_part = parts[-2]  # 2x2, 3x3, 4x4
        shot_idx = int(parts[-1])  # 0, 1, 2, 3
        
        # Map shot index to actual shot count
        shot_map = {0: 100, 1: 1000, 2: 10000, 3: 100000}
        qc_shots = shot_map.get(shot_idx)
        
        # Parse mesh size
        mesh_match = re.match(r'(\d+)x(\d+)', mesh_part)
        if mesh_match:
            nx = int(mesh_match.group(1))
            ny = int(mesh_match.group(2))
        else:
            print(f"  Warning: Could not parse mesh from {case_id}")
            continue
        
        # Map backend name
        if backend_part == 'sv':
            qc_backend = 'statevector_sim_aer'
        elif backend_part == 'brisbane':
            qc_backend = 'ibm_brisbane'
        else:
            qc_backend = backend_part
    else:
        print(f"  Warning: Could not parse case name: {case_id}")
        continue
    
    # Check for results.out file
    results_file = case_dir / "results.out"
    if not results_file.exists():
        print(f"  {case_id}: No results.out (likely failed)")
        continue
    
    # Parse results.out to extract quantum solution
    try:
        results_text = results_file.read_text()
        
        # Extract quantum solution (last array in the file)
        # Format: [0.69985719 0.0142828  0.71413999 0.        ]
        array_match = re.search(r'\[[\d\.\s]+\]', results_text)
        if array_match:
            array_str = array_match.group(0)
            quantum_solution = np.fromstring(array_str.strip('[]'), sep=' ')
        else:
            print(f"  {case_id}: Could not extract quantum solution")
            continue
    except Exception as e:
        print(f"  {case_id}: Error reading results: {e}")
        continue
    
    # Load matrix data to compute classical solution
    # Find the pkl file (matrix size varies: nqmatrix2, nqmatrix5, etc.)
    pkl_files = list(case_dir.glob("hele-shaw_circ_nqmatrix*.pkl"))
    pkl_file = pkl_files[0] if pkl_files else None
    metadata_file = case_dir / "hele-shaw_metadata.pkl"
    
    classical_solution = None
    matrix_size = None
    matrix_size_original = None
    condition_number = None
    condition_number_original = None
    circuit_construction_time = None
    
    if pkl_file and pkl_file.exists():
        try:
            with open(pkl_file, 'rb') as f:
                pkl_data = pickle.load(f)
                matrix = pkl_data.get('matrix')
                vector = pkl_data.get('vector')
                circuit_construction_time = pkl_data.get('t_circ')
                
                if matrix is not None and vector is not None:
                    # Compute classical solution
                    classical_solution = np.linalg.solve(matrix, vector)
                    classical_solution = classical_solution / np.linalg.norm(classical_solution)
                    matrix_size = matrix.shape[0]
                    condition_number = np.linalg.cond(matrix)
        except Exception as e:
            print(f"  {case_id}: Error loading pkl: {e}")
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                if 'A_original' in metadata:
                    matrix_size_original = metadata['A_original'].shape[0]
                if 'original_cond' in metadata:
                    condition_number_original = metadata['original_cond']
        except Exception as e:
            print(f"  {case_id}: Error loading metadata: {e}")
    
    # Load circuit properties from transpiled QPY
    circuit_qubits = None
    circuit_depth = None
    circuit_gates = None
    
    # Find transpiled qpy file (matrix size varies)
    transpiled_qpy_files = list(case_dir.glob("hele-shaw_circ_nqmatrix*_transpiled.qpy"))
    transpiled_qpy = transpiled_qpy_files[0] if transpiled_qpy_files else None
    
    # Skip QPY loading for now - it's slow and we have the key data
    # if transpiled_qpy and transpiled_qpy.exists():
    #     try:
    #         from qiskit.qpy import load as qpy_load
    #         with open(transpiled_qpy, 'rb') as f:
    #             circuits = qpy_load(f)
    #             if circuits:
    #                 circ = circuits[0]
    #                 circuit_qubits = circ.num_qubits
    #                 circuit_depth = circ.depth()
    #                 circuit_gates = circ.size()
    #     except Exception as e:
    #         print(f"  {case_id}: Error loading circuit: {e}")
    
    # Calculate fidelity
    fidelity = None
    if quantum_solution is not None and classical_solution is not None:
        fidelity = float(np.abs(np.dot(quantum_solution, classical_solution)))
    
    # Build case record
    record = {
        'case_id': case_id,
        'input_parameters': {
            'nx': nx,
            'ny': ny,
            'mu': 1.0,
            'qc_shots': qc_shots,
            'qc_backend': qc_backend,
            'NQ_MATRIX': 2,
            'case': 'hele-shaw',
            'max_condition_number': 10000.0
        },
        'matrix_properties': {
            'size_original': matrix_size_original,
            'size_hermitian': matrix_size,
            'condition_number_original': condition_number_original,
            'condition_number_hermitian': condition_number
        },
        'circuit_properties': {
            'num_qubits': circuit_qubits,
            'depth': circuit_depth,
            'num_gates': circuit_gates,
            'depth_transpiled': circuit_depth,
            'num_gates_transpiled': circuit_gates
        },
        'timing': {
            'circuit_construction_sec': circuit_construction_time,
            'circuit_generation_sec': None,
            'execution_sec': None
        },
        'results': {
            'fidelity': fidelity,
            'quantum_solution': quantum_solution.tolist() if quantum_solution is not None else None,
            'classical_solution': classical_solution.tolist() if classical_solution is not None else None
        },
        'metadata': {
            '_original_case_id': case_id,
            '_list_params': ['qc_shots'],
            '_scalar_params': ['qc_backend', 'nx', 'ny', 'mu', 'NQ_MATRIX']
        }
    }
    
    case_data.append(record)
    fid_str = f"{fidelity:.6f}" if fidelity is not None else "N/A"
    print(f"  {case_id}: Reconstructed (fidelity={fid_str})")

# Save to JSON
json_file = workflow_dir / "results_reconstructed.json"
with open(json_file, 'w') as f:
    json.dump(case_data, f, indent=2)

print(f"\nReconstructed {len(case_data)} cases")
print(f"Saved to: {json_file}")

# Also save as pickle for compatibility
pkl_file = workflow_dir / "results_reconstructed.pkl"
# Convert to format expected by plotting functions
case_data_pkl = []
for record in case_data:
    case_info = {
        'case_id': record['case_id'],
        'params': {**record['input_parameters'], **record['matrix_properties'], 
                   **{f'_circuit_{k}': v for k, v in record['circuit_properties'].items()},
                   **{f'_time_{k}': v for k, v in record['timing'].items()},
                   '_metadata': record['metadata']},
        'quantum_solution': np.array(record['results']['quantum_solution']) if record['results']['quantum_solution'] else None,
        'classical_solution': np.array(record['results']['classical_solution']) if record['results']['classical_solution'] else None,
        'metadata': record['metadata']
    }
    case_data_pkl.append(case_info)

with open(pkl_file, 'wb') as f:
    pickle.dump(case_data_pkl, f)
print(f"Saved pickle to: {pkl_file}")

print("\nSummary by configuration:")
configs = {}
for record in case_data:
    backend = record['input_parameters']['qc_backend']
    mesh = f"{record['input_parameters']['nx']}x{record['input_parameters']['ny']}"
    key = f"{backend} {mesh}"
    if key not in configs:
        configs[key] = []
    configs[key].append(record)

for config, cases in sorted(configs.items()):
    print(f"\n{config}: {len(cases)} cases")
    for case in sorted(cases, key=lambda c: c['input_parameters']['qc_shots']):
        shots = case['input_parameters']['qc_shots']
        fid = case['results']['fidelity']
        fid_display = f"{fid:.6f}" if fid is not None else "N/A"
        print(f"  {shots:6d} shots: fidelity={fid_display}")
