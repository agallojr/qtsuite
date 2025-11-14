from qiskit_ibm_catalog import QiskitServerless

# Re-save credentials for QDC 2025
token = "OgNs66DBdwBqjItRWCIzkcPK6l7TL4ll3qNUrdxAmOe3"
instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/e2e570bc5af249dc9d81711cc2febac7:6ec3c5e8-fcf6-4877-98a2-1fe9c1a8bda4::"

print("Saving QiskitServerless credentials...")
QiskitServerless.save_account(
    token=token,
    instance=instance,
    channel="ibm_quantum_platform",
    name="qdc-2025",
    overwrite=True
)
print("Credentials saved successfully!")

# Test connection
print("\nTesting connection...")
client = QiskitServerless(name='qdc-2025')
print(f"Connected successfully!")

# List functions
functions = client.list()
print(f"\nAvailable functions: {len(functions)}")
for func in functions:
    print(f"  - {func}")
