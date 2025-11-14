import sys

# Parse the result from the computation
# Your result variable should be a list of energies

# If you have the result from running 7_local_boston, it's stored in 'result'
# For now, check if result_boston.npz exists

import numpy as np
import os

if os.path.exists('result_boston.npz'):
    data = np.load('result_boston.npz')
    result = data['energy_history']
    print("\n" + "="*60)
    print("COMPUTATION RESULTS (from saved file):")
    print("="*60)
    print(f"Energy history: {result}")
    print(f"Final energy: {result[-1]:.6f}")
    print(f"Number of iterations: {len(result)}")
    print("="*60)
else:
    print("No saved result found. Run 7_local_boston_no_grade.py first")
