import skqd_helpers

inst = \
    "crn:v1:bluemix:public:quantum-computing:us-east:a/e2e570bc5af249dc9d81711cc2febac7:b1404376-8dd2-4d80-a845-173b68c9fa37::"
token = "SSH7N2bP7c4MUZ3hE06peO0QLYOrkJ4LAUDcBjTdDzzQ"
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService.save_account(token=token, instance=inst, name="qdc-2025",
    overwrite=True)


