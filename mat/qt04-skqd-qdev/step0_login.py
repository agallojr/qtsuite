"""
Step 0: Save IBM Quantum account credentials.
"""

from qiskit_ibm_runtime import QiskitRuntimeService


def save_account():
    """Save IBM Quantum account credentials."""
    inst = \
        "crn:v1:bluemix:public:quantum-computing:us-east:a/e2e570bc5af249dc9d81711cc2febac7:b1404376-8dd2-4d80-a845-173b68c9fa37::"
    token = "421HJnwBHQfy4lVpNKqIawLn7gpklVHMLu3pF4x-DaMY"
    
    service = QiskitRuntimeService.save_account(
        token=token, instance=inst, name="qdc-2025", overwrite=True
    )
    print("Account saved successfully.")
    return service


if __name__ == "__main__":
    save_account()
