from qiskit_ibm_catalog import QiskitServerless

# Connect to serverless
client = QiskitServerless(name='qdc-2025')

# List available functions
print("Available functions:")
functions = client.list()
for func in functions:
    print(f"  - {func}")

# Try to load diagonalization_engine
print("\nAttempting to load 'diagonalization_engine'...")
try:
    engine = client.load("diagonalization_engine")
    if engine:
        print(f"  Found: {engine}")
    else:
        print("  Not found (returned None)")
except Exception as e:
    print(f"  Error: {e}")
