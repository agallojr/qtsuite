import skqd_helpers

inst = \
    "crn:v1:bluemix:public:quantum-computing:us-east:a/e2e570bc5af249dc9d81711cc2febac7:6ec3c5e8-fcf6-4877-98a2-1fe9c1a8bda4::"
token = "OgNs66DBdwBqjItRWCIzkcPK6l7TL4ll3qNUrdxAmOe3"
from qiskit_ibm_runtime import QiskitRuntimeService
#service = QiskitRuntimeService(name='qdc-2025')

service = QiskitRuntimeService.save_account(token=token, instance=inst, name="qdc-2025",
    overwrite=True)

#from qc_grader.challenges.qdc_2025 import qdc25_lab4
#qdc25_lab4.submit_name("Team L1-8")
