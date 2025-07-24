"""
An example of using PennyLane to create a quantum circuit and running it on an IBM 
backend using lwfm.

We'll not show all the lwfm bells and whistles here - see the other examples.

1. Make a simple circuit, PennyLane style.
2. Execute it on PennyLane's default simulator device right here.
3. Prepare to toss to the IBM backend by converting it to industry "standard" QASM.
4. Make a workflow context, define the job, and submit it to the IBM backend.
5. Wait for the job to complete and compare the results to the above.

This shows that different backends might produce results in different formats, and the 
app (i.e. this script / workflow) would be responsible for that kinds of handling. lwfm 
normalizes what it can.

This also shows you can avoid using the lwfm "Workflow" stuff if you want, and just
get the benefit of the backend interop.
"""

from typing import cast

import pennylane as qml    # a perhaps regrettable name, but this is what they call it

from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager

NUM_QUBITS = 2

device = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(device)
def circuit():
    """
    Make a simple circuit, PennyLane style.
    """
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Execute it on PennyLane's default simulator device right here
result1 = circuit()
print(f"\nPennyLane result (as probabilities):\t {result1}")

# Now prepare to toss to IBM backend by converting it to industry "standard" QASM
qnode = qml.QNode(circuit, device)
(compiled_tape,), _ = qml.workflow.construct_batch(qnode)()
qasm_code = compiled_tape.to_openqasm()

# lwfm stuff, tersely - define the job
jobDefn = JobDefn(qasm_code, JobDefn.ENTRY_TYPE_STRING, {"format": "qasm"})
runArgs = { "shots": 1024, "optimization_level": 3 }
# fire it at the site - we'll use the ibm simulator and wait synchronously
site = lwfManager.getSite("ibm-quantum-venv")
status = site.getRunDriver().submit(jobDefn, None, "automatic_sim_aer", runArgs)
# there are a few ways in lwfm to get at the results, here's one
result2_raw = lwfManager.deserialize(cast(str,lwfManager.wait(status.getJobId()).getNativeInfo()))
result2 = result2_raw.get_counts()
print("Qiskit result2 (as counts):\t\t", result2)
