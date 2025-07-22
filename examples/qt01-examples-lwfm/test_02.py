"""
Test 02: show how to run the ORNL WCISCC2025 in piece parts, separating out the 
circuit construction from the circuit execution. This will demonstrate negotiating
a circuit built on an old qiskit (wciscc2025) and running it on a backend using the new qiskit.
"""

# wciscc2025 isn't packaged as we would like, so manually put it on the path
import sys
sys.path.append("wciscc2025/qlsa")

#pylint: disable=wrong-import-position, invalid-name

from typing import Any

import numpy as np
from scipy.sparse import diags
from qiskit import qpy

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.base.JobStatus import JobStatus
from lwfm.base.WorkflowEvent import NotificationEvent
from lwfm.midware.LwfManager import lwfManager

# some stuff that used to be built into qiskit but got abandoned...now its own project
from linear_solvers import HHL

if __name__ == '__main__':
    wf = Workflow()
    wf.setName("qt01_examples_lwfm.test_02")
    wf.setProps({"cuzReason": "for giggles"})   # arbitrary metadata about the workflow

    # Generate matrix and vector for linear solver for some number of qubits, then
    # make a circuit using the ORNL HHL code
    N_QUBITS_MATRIX = 2
    MATRIX_SIZE = 2 ** N_QUBITS_MATRIX
    A = 1
    B = -1/3
    matrix = diags([B, A, B],
                [-1, 0, 1], # type: ignore
                shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
    hhl = HHL(1e-3)
    circ = hhl.construct_circuit(matrix, vector)

    # take a look at the circuit - it has some non-unitary custom gates - this
    # complicates the portability of the wciscc code as they are built on an old qiskit
    print(circ.draw())

    # write the circuit in Qiskit-specific QPY portability format (not OpenQASM cuz
    # the above) & notate it as part of the workflow
    qpy_filename = f"/tmp/workflow_{wf.getWorkflowId()}_circuit.qpy"
    with open(qpy_filename, 'wb') as qpy_file:
        qpy.dump(circ, qpy_file)
    lwfManager.notatePut(qpy_filename, wf.getWorkflowId(),
        {   "description": "HHL circuit QPY file",
            "cuzReason": "for testing ORNL WCISCC2025 code",
            "num_qubits": N_QUBITS_MATRIX,
            "matrix_size": MATRIX_SIZE
        })

    # to run we'll use a pre-prepared virtual environment containing the IBM cloud code
    # and the latest qiskit libs
    site = lwfManager.getSite("ibm-quantum-venv")

    # define the job to run our Python script
    jobDefn = JobDefn(qpy_filename, JobDefn.ENTRY_TYPE_STRING)
    runArgs: dict[str,Any] = {
        "shots": 1024,                  # number of runs of the circuit
        "optimization_level": 3         # agressive transpiler optimization (values: 0-3)
    }
    # iterate over a set of target backends, run each, send an email when they finish
    # if not on a corporate network, add  "ibm_brisbane" to run on a real cloud backend
    # site.getSpinDriver().listComputeTypes() will show available compute types
    target_compute_types = [ "statevector_sim_aer", "matrix_product_state_sim_aer" ]
    for target in target_compute_types:
        runArgs["computeType"] = target
        status = site.getRunDriver().submit(jobDefn, wf, runArgs["computeType"], runArgs)
        lwfManager.setEvent(NotificationEvent(status.getJobId(), JobStatus.COMPLETE,
            lwfManager.getSiteProperties("lwfm").get("emailMe") or "",
            f"wf {wf.getWorkflowId()}: {target}", "done!", status.getJobContext()))
