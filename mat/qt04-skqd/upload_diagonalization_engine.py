"""
Upload the diagonalization_engine to Qiskit Serverless.
"""

from qiskit_ibm_catalog import QiskitServerless, QiskitFunction

# Connect to serverless
print("Connecting to Qiskit Serverless...")
client = QiskitServerless(name='qdc-2025')

# Define the function
print("Defining diagonalization_engine function...")
diag_function = QiskitFunction(
    title="diagonalization_engine",
    entrypoint="diagonalization_engine.py",
    working_dir="./serverless_functions/",  # Use dedicated directory to avoid uploading .venv
    dependencies=[
        "qiskit-addon-sqd==0.12.0",  # Required by QDC 2025 serverless
        # qiskit-ibm-runtime and numpy are pre-installed in serverless environment
    ],
)

print(f"Function defined: {diag_function}")

# Upload to serverless
print("\nUploading to serverless...")
try:
    result = client.upload(diag_function)
    print(f"✓ Successfully uploaded: {result}")
    print("\nVerifying upload...")
    
    # List functions to confirm
    functions = client.list()
    print(f"\nAvailable functions ({len(functions)}):")
    for func in functions:
        print(f"  - {func}")
    
    # Try to load it
    print("\nTrying to load the uploaded function...")
    loaded = client.load("diagonalization_engine")
    if loaded:
        print(f"✓ Successfully loaded: {loaded}")
    else:
        print("⚠ Function not found after upload")
        
except Exception as e:
    print(f"✗ Error during upload: {e}")
    print("\nMake sure you have:")
    print("  1. Valid credentials for QDC 2025")
    print("  2. Permission to upload functions")
    print("  3. The diagonalization_engine.py file in the current directory")
