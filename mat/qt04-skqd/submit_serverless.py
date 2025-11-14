from qc_grader.challenges.qdc_2025 import qdc25_lab4
import numpy as np

# Load saved result
print("Loading result from result_serverless.npz...")
data = np.load('result_serverless.npz')

result = data['energy_history']
num_orbs = int(data['num_orbs'])
hopping = float(data['hopping'])
onsite = float(data['onsite'])
hybridization = float(data['hybridization'])
chemical_potential = float(data['chemical_potential'])

print("="*60)
print("LOADED RESULT:")
print("="*60)
print(f"Energy history: {result}")
print(f"Final energy: {result[-1]:.6f}")
print(f"Parameters: num_orbs={num_orbs}, hopping={hopping}, onsite={onsite}")
print(f"            hybridization={hybridization}, chemical_potential={chemical_potential}")
print("="*60)

# Submit for grading
print("\nSubmitting to grader...")
qdc25_lab4.grade_lab4_ex6(result, num_orbs, hopping, onsite, hybridization, chemical_potential)
