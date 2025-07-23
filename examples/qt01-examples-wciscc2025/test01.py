"""
Test 01: show how to run the ORNL WCISCC2025 code as-is, with their command line
interface, but within the lwfm framework. This is also of course a general example of
running any command line tool as a step in an lwfm workflow. Notice a lack of quantum
import statements.
"""

#pylint: disable=broad-exception-caught, invalid-name

from typing import Optional

from lwfm.base.JobDefn import JobDefn
from lwfm.base.Workflow import Workflow
from lwfm.base.JobStatus import JobStatus
from lwfm.midware.LwfManager import lwfManager, logger


if __name__ == '__main__':
    # we will run locally. ~/.lwfm/sites.toml will define this site
    site = lwfManager.getSite("local")

    # define a workflow, give it some metadata about the project
    wf = Workflow()
    wf.setName("qt01_examples_wciscc2025.test01")
    wf.setDescription("running the ORNL WCISCC2025 code as-is iteratively by qubit count")
    wf.setProps({"cuzReason": "for giggles"})
    logger.info(f"Workflow created with name: {wf.getName()}")

    # define the job with the command line args & run it; iterate over qubit counts
    ENTRY_POINT = "python wciscc2025/qlsa/test_linear_solver.py"
    status: Optional[JobStatus] = None
    for num_qubits in [2, 3, 4]:
        args = ["-nq", str(num_qubits)]
        jobDefn = JobDefn(ENTRY_POINT, JobDefn.ENTRY_TYPE_SHELL, args)
        status = site.getRunDriver().submit(jobDefn, wf)
        logger.info(f"Job submitted with ID: {status.getJobId()} for {num_qubits} qubits")

    if status is None:
        ex = RuntimeError("Job submission failed")
        logger.error(str(ex))  # persists to lwfm store
        raise ex

    # since its local, let's wait synchronously for the last job to finish
    status = lwfManager.wait(status.getJobId())
    statusList = lwfManager.getJobStatusesForWorkflow(wf.getWorkflowId())
    if not statusList:
        ex = RuntimeError("No job statuses found for the workflow(?)")
        logger.error(str(ex))
        raise ex

    # stand-in for some "post-processing"
    # 1. we'll make a file, for convenience we'll name it using the workflow id
    # 2. for each job in the workflow, get the stdout it produced and write it to the file
    output_filename = f"/tmp/workflow_{wf.getWorkflowId()}_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"Workflow Results for: {wf.getName()}\n")
        output_file.write(f"Workflow ID: {wf.getWorkflowId()}\n")
        output_file.write(f"Description: {wf.getDescription()}\n")
        output_file.write("="*50 + "\n\n")
        # iterate over the job statuses and write their stdout to the file
        # this is a stand-in for more complex post-processing
        for status in reversed(statusList):
            output_file.write("="*50 + "\n\n")
            output_file.write(f"Job {status.getJobId()} status {status.getStatus()}\n")
            out_filename = lwfManager.getStdoutFilename(status.getJobContext())
            try:
                with open(out_filename, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    output_file.write(f"Log content:\n{log_content}\n")
            except Exception as ex:
                error_msg = f"Error reading log file {out_filename}: {ex}"
                output_file.write(f"{error_msg}\n")
            output_file.write("\n")

    # 3. tag the file with some metadata for later finding
    lwfManager.notatePut(output_filename, wf.getWorkflowId(), 
                         {  "description": "Workflow results",
                            "cuzReason": "for more giggles" })
    logger.info(f"Results file {output_filename}")

    # # let's show we can find it again
    # metasheets = lwfManager.find({"workflowId": wf.getWorkflowId()})
    # if not metasheets:
    #     logger.warning("No metadata sheets found for the workflow")
    # else:
    #     for sheet in metasheets:
    #         logger.info(f"Found metadata sheet: {sheet}")
