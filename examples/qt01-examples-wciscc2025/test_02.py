"""
Test 02: show how to run the ORNL WCISCC2025 in piece parts, separating out the 
circuit construction from the circuit execution. This will let us exercise their code
more broadly.
"""

import sys
sys.path.append("wciscc2025/qlsa")

#pylint: disable=wrong-import-position

import numpy as np
from scipy.sparse import diags
from qiskit.qasm2 import dumps

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager

from linear_solvers import HHL

if __name__ == '__main__':
    # define a workflow, give it some metadata, though we only have one job to run
    wf = Workflow()
    wf.setName("qt01_examples_wciscc2025.test_02")
    wf.setDescription("running the ORNL WCISCC2025 in 2 parts - circuit build & run")
    wf.setProps({"cuzReason": "for giggles"})

    # Generate matrix and vector for linear solver for some number of qubits
    N_QUBITS_MATRIX = 2
    MATRIX_SIZE = 2 ** N_QUBITS_MATRIX
    A = 1
    B = -1/3
    matrix = diags([B, A, B],
                [-1, 0, 1],
                shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))

    # make a circuit using the ORNL HHL code & take a look at it
    hhl = HHL(1e-3)
    circ = hhl.construct_circuit(matrix, vector)
    print(circ.draw())

    # now run this on IBM hardware - a local simulator, and real computer in the cloud
    site = lwfManager.getSite("ibm-quantum-venv")
    jobDefn = JobDefn()
    site.getRunDriver().submit(jobDefn, wf)

    # define the job & some runtime args
    # qpy.dump(circ, "/tmp/hhl.qpy")
    jobDefn = JobDefn(dumps(circ), JobDefn.ENTRY_TYPE_STRING, {"format": "qasm"})
    runArgs = {
        "shots": 1024,                  # number of runs of the circuit
        "optimization_level": 3         # agressive transpiler optimization (values: 0-3)
    }
    runArgs["computeType"] = "ibm_brisbane_aer"  # run on an aer simulator
    statusA = site.getRunDriver().submit(jobDefn, wf, runArgs["computeType"], runArgs)
    # we will wait synchronously
    statusA = lwfManager.wait(statusA.getJobId())
    print(statusA)


    # now demonstrate the same in a second run on a real backend in the cloud
    # runArgs["computeType"] = runArgs["computeType"].replace("_aer", "")
    # statusB = site.getRunDriver().submit(jobDefn, wf, runArgs["computeType"], runArgs)
    # statusB = lwfManager.wait(statusB.getJobId())
