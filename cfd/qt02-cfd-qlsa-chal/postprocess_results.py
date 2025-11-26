#!/usr/bin/env python3
"""Post-process results from a completed workflow run"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from plotting import plot_qlsa_generic, plot_scaling_analysis, plot_scaling_table, export_scaling_data_csv

if len(sys.argv) < 2:
    print("Usage: python postprocess_results.py <workflow_id>")
    sys.exit(1)

workflow_id = sys.argv[1]
workflow_dir = Path(f"/Users/agallojr/.lwfm/out/qt02-cfd/{workflow_id}")

if not workflow_dir.exists():
    print(f"Workflow directory not found: {workflow_dir}")
    sys.exit(1)

print(f"Post-processing workflow: {workflow_id}")

# Scan for case directories
case_dirs = sorted([d for d in workflow_dir.iterdir() if d.is_dir() and d.name.startswith('hs_')])
print(f"Found {len(case_dirs)} case directories")

# Load results from each case
case_data = []
quantum_solutions = []
classical_solutions = []

for case_dir in case_dirs:
    case_id = case_dir.name
    print(f"Loading case: {case_id}")
    
    # Look for result files
    result_pkl = case_dir / f"{case_id}_result.pkl"
    
    # Try to find any .pkl file with results
    pkl_files = list(case_dir.glob("*.pkl"))
    
    # For now, we'll need to reconstruct from individual case outputs
    # Check if quantum solution exists
    qpy_files = list(case_dir.glob("*.qpy"))
    
    # This is tricky - we need the actual result data
    # Let's check if there's a results file at the case level
    print(f"  Found {len(pkl_files)} pkl files, {len(qpy_files)} qpy files")

print("\nNote: The workflow stores results in memory during execution.")
print("Since the run crashed during post-processing, the data is lost.")
print("\nTo preserve results in future runs, we need to add incremental saving.")
print("\nYou'll need to re-run the workflow with the fixed plotting code.")
