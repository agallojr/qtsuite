"""
Yet another example of running the WCISCC2025 code, separating out circuit construction
from backend execution. This example shows running the ORNL code by defining a CFD 
case file akin to something engineers are more familiar.

If you were running this on the command line as per ORNL instructions, you might do this:

$ python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata

$ python solver.py -case sample-tridiag -casefile input_vars.yaml -backtyp ideal -s 1000 --savedata

"""


# we're going to use wciscc2025 directly, but it isn't packaged as we would like,
# so manually put it on the path
import sys
sys.path.append("wciscc2025/qlsa")

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager


if __name__ == '__main__':
    wf = Workflow("HHL Circuit Example", "Example of running HHL circuit with a CFD case file.")

    # we'll run the first part of the ORNL instructions as is, here, in a sandbox
    # using their library dependencies
    site = lwfManager.getSite("local")

    # Create JobDefn to generate the circuit using circuit_HHL.py run here, synchronously
    # the input_vars.yaml file can be anywhere, represents your engineering "case"
    entry_point = "python wciscc2025/qlsa/circuit_HHL.py"
    args = ["-case", "sample-tridiag", "-casefile", "wciscc2025/qlsa/input_vars.yaml", "--savedata"]
    jobDefn = JobDefn(entry_point, JobDefn.ENTRY_TYPE_SHELL, args)
    status = site.getRunDriver().submit(jobDefn, wf)
    if status is None:
        raise RuntimeError("Job submission failed")
    status = lwfManager.wait(status.getJobId())

    # the ORNL code is going to make a ./models/ dir containing a circuit in QPY format,
    # and we've already seen how to deal with that in prior examples
    # their executor, wciscc2025/qlsa/solver.py, will accept some arguments like the
    # number of shots, noise info about the target backend simulator, etc.
    # it uses numpy to solve the equations classically for comparison, and we'll not
    # do that part, so here's the meat of the matter:
    #    matrix, vector, input_vars = matvec.get_matrix_vector(args)
    #    n_qubits_matrix = int(np.log2(matrix.shape[0]))
    #    "decide if you want a simulator or a real backend from IBM or IQM" - we'll ignore IQM
    #    qc_circ(n_qubits_matrix, classical_solution, args, input_vars)
    # this last function decides if we're using Aer simulator or a real backend,
    # loads the circuit from qpy format, transpiles the circuit accordingly
    # (see qlsa/func_qc.py), runs the circuit with some twists, and saves the results to a file
    # if you're good being pinning to their library dependencies, just use their code,
    # but otherwise we can use the lwfm ability to define a site in a virtual environment,
    # then look at their code (again, func_qc.py) and decide which of the "twists" are needed.
    site = lwfManager.getSite("ibm-quantum-venv")
    computeType = "statevector_sim_aer" # for simplicity of the example
    runArgs = {"shots": 1024}
    jobDefn = JobDefn("models/sample_HHL_circ_nqmatrix1.qpy",
                      JobDefn.ENTRY_TYPE_STRING, {"format": "qpy"})
    status = site.getRunDriver().submit(jobDefn, wf, computeType, runArgs)
    if status is None:
        raise RuntimeError("Job submission failed")
    status = lwfManager.wait(status.getJobId())
    print(lwfManager.deserialize(status.getNativeInfo()))  # type: ignore
