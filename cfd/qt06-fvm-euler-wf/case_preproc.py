"""
case preprocessing
"""

#pylint: disable=invalid-name

import numpy as np

from qtlib import log_with_time
from lwfm.base.JobDefn import JobDefn
from lwfm.base.JobContext import JobContext
from lwfm.base.JobStatus import JobStatus
from lwfm.midware.LwfManager import lwfManager, logger

def case_preproc(wf, caseId, caseArgs, case_start_time, preprocess_site, caseOutDir):
    """
    run a single case's preprocessing step
    """
    preprocess_status = preprocess_site.getRunDriver().submit(
        JobDefn(
            f"python {caseArgs['circuit_gen_path']}",
            JobDefn.ENTRY_TYPE_SHELL,
            [
                "-case", caseArgs['case'], "-casefile", str(out_path),
                "--savedata", "--no-transpile"
            ]
        ),
        JobContext().initialize(
            "preproc", wf.getWorkflowId(), preprocess_site.getSiteName()
        )
    )
    if preprocess_status is None:
        logger.error(f"Preprocess job submission failed {caseId}")
        return

    log_with_time(f"[{caseId}] Waiting for circuit generation to complete", case_start_time)
    preprocess_status = lwfManager.wait(preprocess_status.getJobId())
    if (
        (preprocess_status is None)
        or (preprocess_status.getStatus() != JobStatus.COMPLETE)
    ):
        logger.error(f"Preprocess job failed {caseId}")
        return
    log_with_time(f"[{caseId}] Circuit generation complete", case_start_time)

    # locate the QPY file produced by the preprocess step
    qpy_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.qpy"))
    if not qpy_candidates:
        logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
        return
    circuit_qpy_path = max(qpy_candidates, key=lambda p: p.stat().st_mtime)
    lwfManager.notatePut(
        circuit_qpy_path.as_posix(),
        preprocess_status.getJobContext(),
        {"case": caseId})
    # pick the most recently modified PKL in case multiple exist
    pkl_candidates = list(
        caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.pkl")
    )
    if not pkl_candidates:
        logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
        return
    circuit_pkl_path = max(pkl_candidates, key=lambda p: p.stat().st_mtime)
    lwfManager.notatePut(
        circuit_pkl_path.as_posix(),
        preprocess_status.getJobContext(),
        {"case": caseId})
    log_with_time(f"[{caseId}] Located circuit files: {circuit_qpy_path.name}", case_start_time)

    # Load and analyze the circuit
    with open(circuit_qpy_path, "rb") as f:
        circuits = qpy_load(f)
        circuit = circuits[0] if isinstance(circuits, list) else circuits

    # Get circuit properties
    num_qubits_circuit = circuit.num_qubits
    circuit_depth = circuit.depth()
    circuit_size = circuit.size()  # total number of gates
    num_ops = len(circuit.count_ops())  # number of unique gate types

    # Log input parameters vs circuit properties
    logger.info(
        f"[{caseId}] Input parameters: NQ_MATRIX={caseArgs['NQ_MATRIX']}, "
        f"nx={caseArgs['nx']}, ny={caseArgs['ny']}")
    logger.info(
        f"[{caseId}] Circuit properties: qubits={num_qubits_circuit}, "
        f"depth={circuit_depth}, gates={circuit_size}, "
        f"gate_types={num_ops}")
    logger.info(f"[{caseId}] Gate breakdown: {circuit.count_ops()}")
    log_with_time(f"[{caseId}] Circuit loaded: {num_qubits_circuit} qubits, "
        f"depth={circuit_depth}, {circuit_size} gates",
        case_start_time)

    # get the matrix and vector from the PKL
    log_with_time(f"[{caseId}] Loading matrix data and computing classical solution",
        case_start_time)
    with open(circuit_pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
        matrix = pkl_data["matrix"]
        vector = pkl_data["vector"]

    # Load metadata to get original untransformed matrix
    # ORNL code saves as {case}_metadata.pkl, not {circuit_file}_metadata.pkl
    case_name = caseArgs.get('case', 'hele-shaw')
    metadata_pkl_path = circuit_pkl_path.parent / f"{case_name}_metadata.pkl"
    if metadata_pkl_path.exists():
        with open(metadata_pkl_path, "rb") as f:
            metadata = pickle.load(f)
            if 'A_original' in metadata and 'original_cond' in metadata:
                A_original = metadata['A_original']
                original_cond = metadata['original_cond']
                logger.info(
                    f"[{caseId}] Original CFD matrix (before transformation): "
                    f"size={A_original.shape}, κ={original_cond:.4e}"
                )
                # Save pre-conditioning metrics
                caseArgs['_matrix_condition_number_original'] = float(original_cond)
                caseArgs['_matrix_size_original'] = A_original.shape[0]
            else:
                logger.warning(f"[{caseId}] Original matrix not found in metadata")
    else:
        logger.warning(f"[{caseId}] Metadata file not found")

    # based on the size of the matrix, we can infer the number of qubits
    n_qubits_matrix = int(np.log2(matrix.shape[0]))
    logger.info(
        f"[{caseId}] Matrix properties: "
        f"qubits_from_matrix={n_qubits_matrix}, "
        f"rows={matrix.shape[0]}, cols={matrix.shape[1]}")
    logger.info(
        f"[{caseId}] Matrix sparsity: "
        f"nonzeros={np.count_nonzero(matrix)}, "
        f"density={np.count_nonzero(matrix) / matrix.size:.4f}")
    logger.info(f"[{caseId}] Matrix sample (first 4x4): \n{matrix[:4, :4]}")
    logger.info(f"[{caseId}] Matrix diagonal: {np.diag(matrix)}")

    # Calculate condition number
    condition_number = np.linalg.cond(matrix)
    logger.info(f"[{caseId}] Matrix condition number: κ={condition_number:.4e}")

    # Save condition number for analysis
    caseArgs['_matrix_condition_number'] = float(condition_number)

    # if K > threshold, matrix is too ill-conditioned for circuit
    max_condition_number = caseArgs.get('max_condition_number', 1e12)
    if condition_number > max_condition_number:
        logger.warning(
            f"[{caseId}] Matrix is too ill-conditioned for circuit "
            f"(κ={condition_number:.4e} > {max_condition_number:.4e})")
        return

    # Compare input parameters with actual circuit/matrix
    grid_size = caseArgs['nx'] * caseArgs['ny']
    logger.info(
        f"[{caseId}] Comparison: grid_size={grid_size} (nx*ny), "
        f"NQ_MATRIX={caseArgs['NQ_MATRIX']}, "
        f"matrix_size={matrix.shape[0]}, "
        f"circuit_qubits={num_qubits_circuit}")

    # Calculate classical solution to use as reference
    classical_solution_vector = np.linalg.solve(
        matrix, vector/np.linalg.norm(vector)
    )
    classical_euclidean_norm = float(np.linalg.norm(classical_solution_vector))
    logger.info(f"Classical solution vector: {classical_solution_vector}")
    logger.info(f"Classical euclidean norm: {classical_euclidean_norm}")
    log_with_time(logger,
        f"[{caseId}] Classical solution computed "
        f"(norm={classical_euclidean_norm:.6f})",
        case_start_time)
