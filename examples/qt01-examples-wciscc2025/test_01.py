"""
Test 01: show how to run the ORNL WCISCC2025 code as-is, with their command line
interface, but within the lwfm framework. This is also of course a general example of
running command line tools as steps in an lwfm workflow - notice a lack of quantum
import statements.
"""

from lwfm.base.JobDefn import JobDefn
from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager


if __name__ == '__main__':

    site = lwfManager.getSite("local")

    # define a workflow, give it some metadata
    wf = Workflow()
    wf.setName("qt01_examples_wciscc2025.test_01")
    wf.setDescription("running the ORNL WCISCC2025 code as-is")
    wf.setProps({"cuzReason": "for giggles"})

    # define the job with the ORNL code & command line args & iterate
    ENTRY_POINT = "python wciscc2025/qlsa/test_linear_solver.py"
    for num_qubits in [2, 3, 4]:
        args = ["-nq", str(num_qubits)]
        jobDefn = JobDefn(ENTRY_POINT, JobDefn.ENTRY_TYPE_SHELL, args)
        # run it
        status = site.getRunDriver().submit(jobDefn, wf)

    # since its local, let's wait synchronously for the last job to finish
    status = lwfManager.wait(status.getJobId())
    print(f"Job {status.getJobId()} finished with status {status.getStatus()}")

    # post-process the results - the ORNL code dumps everything to stdout
    # TODO fix this
    print(lwfManager.getLoggingByWorkflowId(wf.getWorkflowId()))
