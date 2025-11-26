"""
Main workflow execution logic for QLSA challenge.
"""

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens
#pylint: disable=multiple-statements, broad-exception-caught
#pylint: disable=redefined-outer-name, consider-using-enumerate
#pylint: disable=consider-iterating-dictionary

import sys
import os
from pathlib import Path
from typing import cast
import pickle
import time

from qiskit.result import Result as QiskitJobResult
from qiskit.qpy import load as qpy_load

import numpy as np

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager, logger
from lwfm.base.JobStatus import JobStatus
from lwfm.base.JobContext import JobContext

from qtlib import get_cases_args, log_with_time
from noise import add_custom_noise


def run_workflow(workflow_toml):
    """
    Execute the main workflow for all cases.
    
    Parameters
    ----------
    workflow_toml : str
        Path to the TOML configuration file
    
    Returns
    -------
    tuple
        (wf, caseResults, quantum_solutions, classical_solution_vector,
         casesArgs, globalArgs, exec_status)
    """
    workflow_start_time = time.time()

    # ******************************************************************************
    # get the arguments for the cases in this workflow from the TOML file

    log_with_time("Starting workflow - loading case arguments", [workflow_start_time])
    casesArgs = get_cases_args(workflow_toml)
    globalArgs = casesArgs["global"]

    # Log case expansion info
    case_count = len([k for k in casesArgs.keys() if k != "global"])
    logger.info(f"Loaded {case_count} cases (after list expansion)")
    for case_id in [k for k in casesArgs.keys() if k != "global"]:
        logger.info(f"  - {case_id}: {casesArgs[case_id]}")

    # make an lwfm workflow to bundle all these cases
    log_with_time("Creating workflow object", [workflow_start_time])
    wf = Workflow("winter challenge", "ornl winter challenge", globalArgs)
    if (wf := lwfManager.putWorkflow(wf)) is None: sys.exit(1)
    log_with_time(f"Registered workflow {wf.getWorkflowId()}", [workflow_start_time])

    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())
    # will be altered per case, so keep a copy of the root
    keepSaveDir = globalArgs["savedir"]

    # create workflow output directory
    workflow_out_dir = Path(keepSaveDir).expanduser()
    workflow_out_dir.mkdir(parents=True, exist_ok=True)

    # warm up lwfm sandboxes we use by updating their respective dependencies
    if globalArgs.get("warmup_sites", True):
        log_with_time("Warming up lwfm sites (updating dependencies)", [workflow_start_time])
        lwfManager.updateSite()  # this projct folder ("./.venv")
        lwfManager.updateSite(globalArgs["preprocess_site"])  # circuits
        lwfManager.updateSite(globalArgs["exec_site"])  # runs circuits
        log_with_time("Site warmup complete", [workflow_start_time])

    logger.info(f"DEBUG: Attempting to get preprocess_site: {globalArgs['preprocess_site']}")
    preprocess_site = lwfManager.getSite(globalArgs["preprocess_site"])
    logger.info(f"DEBUG: preprocess_site result: {preprocess_site}")
    logger.info(f"DEBUG: preprocess_site type: {type(preprocess_site)}")
    
    logger.info(f"DEBUG: Attempting to get exec_site: {globalArgs['exec_site']}")
    exec_site = lwfManager.getSite(globalArgs["exec_site"])
    logger.info(f"DEBUG: exec_site result: {exec_site}")


    # ******************************************************************************

    # keep track of the results for each case
    caseResults: list[QiskitJobResult] = []
    quantum_solutions = []
    classical_solutions = []  # Store classical solution for each case
    exec_status = None  # Initialize for cases where circuit execution is skipped

    # we know this workflow will run the same circuit multiple times, so we'll test if
    # this is the first case and do all the preprocessing just once
    matrix = None       # the A in A x = b
    vector = None       # the b in A x = b

    # Circuit cache: key is (case, NQ_MATRIX, nx, ny) -> (qpy_path, pkl_path)
    # This avoids regenerating circuits when only shots differ
    circuit_cache = {}


    # 0. populate ORNL code property file template for the case
    # 1. circuit generation
    # 2. circuit execution
    # 3. post processing
    # we'll also do postprocessing for the workflow as a whole at the end

    # for each case in the workflow toml
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        case_start_time = time.time()
        log_with_time(f"========== Starting case {caseId} ==========", [workflow_start_time])

        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        # put all artifacts for this case in its own subdir of workflow root
        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)

        # **************************************************************************
        # 0. populate ORNL code property file template for the case

        log_with_time(
            f"[{caseId}] Phase 0: Populating ORNL input template",
            [workflow_start_time, case_start_time]
        )
        # take the templatized ORNL input_vars.yaml, fill it in with
        # the case args, save it
        with open("./input_vars.yaml", "r", encoding="utf-8") as f:
            input_vars = f.read()
        for key, value in caseArgs.items():
            input_vars = input_vars.replace("$" + key, str(value))
        out_dir = caseOutDir
        out_path = out_dir.joinpath(f"input_vars_{caseId}.yaml")
        circuit_qpy_path = out_dir.joinpath(
            f"{caseArgs['case']}_circ_nqmatrix{caseArgs['NQ_MATRIX']}.qpy"
        )
        # If the ORNL code expects the case to be the second YAML document
        # (doc index 1) for 'hele-shaw', we write a two-document YAML file
        # where the first document is a minimal placeholder and the second
        # is the actual filled template. Other cases remain single-document.
        if caseArgs.get('case') == 'hele-shaw':
            placeholder = "placeholder: true\ncase_name: placeholder\n"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(placeholder)
                f.write("---\n")
                f.write(input_vars)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(input_vars)
        # associate the input_vars file with the workflow
        lwfManager.notatePut(out_path.as_posix(),
            JobContext().initialize("template", wf.getWorkflowId()), {"case": caseId})
        log_with_time(f"[{caseId}] Input template created: {out_path}", [workflow_start_time, case_start_time])


        # **************************************************************************
        # 1. circuit generation/preprocessing

        log_with_time(f"[{caseId}] Phase 1: Starting circuit generation/preprocessing",
            [workflow_start_time, case_start_time])

        # in circuit generation, we discretize the governing Hele-Shaw
        # equations into their Ax=B linear form. the matrix A represents
        # the equations, and vector b the boundary conditions. all of this
        # is done by the circuit_HHL.py script given parameters found in
        # a casefile - in goes things like grid resolution, #qubits, etc.
        # and out comes a quantum circuit in a Qiskit-portable QPY format.

        # Build cache key from parameters that affect circuit (not shots)
        cache_key = (
            caseArgs.get('case'),
            caseArgs.get('NQ_MATRIX'),
            caseArgs.get('nx'),  # for hele-shaw
            caseArgs.get('ny'),  # for hele-shaw
        )

        # Check if we already have this circuit cached
        if cache_key in circuit_cache:
            circuit_qpy_path, circuit_pkl_path = circuit_cache[cache_key]
            log_with_time(
                f"[{caseId}] Using cached circuit: {circuit_qpy_path.name}",
                [workflow_start_time, case_start_time]
            )
            caseArgs['_time_circuit_generation'] = 0.0  # cached, no generation time
            caseArgs['_circuit_cached'] = True
        else:
            # run the ORNL code to take the CFD casefile and generate the
            # circuit (.qpy), save its extra data (.pkl). we will not let it
            # transpile the circuit, as we will do that in the execution step.

            log_with_time(
                f"[{caseId}] Submitting circuit generation job", [workflow_start_time, case_start_time]
            )
            logger.info(f"DEBUG: About to submit job, preprocess_site is: {preprocess_site}")
            if preprocess_site is None:
                logger.error(f"[{caseId}] preprocess_site is None! Cannot submit job.")
                logger.error(f"[{caseId}] Site name was: {globalArgs['preprocess_site']}")
                continue
            
            circuit_gen_start = time.time()
            preprocess_status = preprocess_site.getRunDriver().submit(
                JobDefn(
                    f"python {caseArgs['circuit_hhl_path']}",
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
            if (preprocess_status is None):
                logger.error(f"Preprocess job submission failed {caseId}")
                continue  # to next case
            log_with_time(
                f"[{caseId}] Waiting for circuit generation to complete",
                [workflow_start_time, case_start_time]
            )
            preprocess_status = lwfManager.wait(preprocess_status.getJobId())
            if (
                (preprocess_status is None)
                or (preprocess_status.getStatus() != JobStatus.COMPLETE)
            ):
                logger.error(f"Preprocess job failed {caseId}")
                continue  # to next case
            circuit_gen_time = time.time() - circuit_gen_start
            caseArgs['_time_circuit_generation'] = circuit_gen_time
            caseArgs['_circuit_cached'] = False
            log_with_time(f"[{caseId}] Circuit generation complete ({circuit_gen_time:.2f}s)", [workflow_start_time, case_start_time])
            lwfManager.notateGet(out_path.as_posix(), preprocess_status.getJobContext(),
                {"case": caseId})

            # locate the QPY file produced by the preprocess step. The wciscc2025
            # code composes the filename based on the actual matrix size (may pad
            # up to a power of two), so don't assume NQ_MATRIX from the TOML.
            # pick the most recently modified QPY in case multiple exist
            qpy_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.qpy"))
            if not qpy_candidates:
                logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
                continue  # to next case
            circuit_qpy_path = max(qpy_candidates, key=lambda p: p.stat().st_mtime)
            lwfManager.notatePut(
                circuit_qpy_path.as_posix(),
                preprocess_status.getJobContext(),
                {"case": caseId}
            )
            # pick the most recently modified PKL in case multiple exist
            pkl_candidates = list(
                caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.pkl")
            )
            if not pkl_candidates:
                logger.error(f"No generated .pkl found for {caseId} in {caseOutDir}")
                continue  # to next case
            circuit_pkl_path = max(pkl_candidates, key=lambda p: p.stat().st_mtime)
            lwfManager.notatePut(
                circuit_pkl_path.as_posix(),
                preprocess_status.getJobContext(),
                {"case": caseId}
            )
            log_with_time(
                f"[{caseId}] Located circuit files: {circuit_qpy_path.name}",
                [workflow_start_time, case_start_time]
            )

            # Cache the circuit paths for reuse
            circuit_cache[cache_key] = (circuit_qpy_path, circuit_pkl_path)
            log_with_time(
                f"[{caseId}] Cached circuit for key {cache_key}",
                [workflow_start_time, case_start_time]
            )

        # Note: We don't load the circuit here in qt02's venv.
        # The QPY file is passed directly to ibm-site which will load,
        # transpile, and execute it. We'll get circuit metadata from
        # the transpiled version later.
        log_with_time(
            f"[{caseId}] Circuit QPY ready for execution site",
            [workflow_start_time, case_start_time]
        )
        
        # Placeholder values - will be filled from transpiled circuit
        num_qubits_circuit = None
        circuit_depth = None
        circuit_size = None

        # get the matrix and vector from the PKL
        log_with_time(
            f"[{caseId}] Loading matrix data and computing classical solution",
            [workflow_start_time, case_start_time]
        )
        with open(circuit_pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
            matrix = pkl_data["matrix"]
            vector = pkl_data["vector"]
            # Capture circuit construction time from frontier-qlsa
            if 't_circ' in pkl_data:
                caseArgs['_time_circuit_construction'] = pkl_data['t_circ']

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
                    logger.warning(
                        f"[{caseId}] Original matrix not found in metadata"
                    )
        else:
            logger.warning(f"[{caseId}] Metadata file not found")

        # based on the size of the matrix, we can infer the number of qubits
        n_qubits_matrix = int(np.log2(matrix.shape[0]))
        logger.info(
            f"[{caseId}] Matrix properties: "
            f"qubits_from_matrix={n_qubits_matrix}, "
            f"rows={matrix.shape[0]}, cols={matrix.shape[1]}"
        )
        logger.info(
            f"[{caseId}] Matrix sparsity: "
            f"nonzeros={np.count_nonzero(matrix)}, "
            f"density={np.count_nonzero(matrix) / matrix.size:.4f}"
        )
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
                f"(κ={condition_number:.4e} > {max_condition_number:.4e})"
            )
            continue

        # Compare input parameters with actual circuit/matrix
        # For hele-shaw, grid_size = nx*ny; for tridiagonal, use matrix size
        if 'nx' in caseArgs and 'ny' in caseArgs:
            grid_size = caseArgs['nx'] * caseArgs['ny']
            logger.info(
                f"[{caseId}] Comparison: grid_size={grid_size} (nx*ny), "
                f"NQ_MATRIX={caseArgs.get('NQ_MATRIX', 'N/A')}, "
                f"matrix_size={matrix.shape[0]}, "
                f"circuit_qubits={num_qubits_circuit}"
            )
        else:
            logger.info(
                f"[{caseId}] Comparison: "
                f"NQ_MATRIX={caseArgs.get('NQ_MATRIX', 'N/A')}, "
                f"matrix_size={matrix.shape[0]}, "
                f"circuit_qubits={num_qubits_circuit}"
            )

        # Calculate classical solution to use as reference
        classical_solution_vector = np.linalg.solve(
            matrix, vector/np.linalg.norm(vector)
        )
        classical_euclidean_norm = float(np.linalg.norm(classical_solution_vector))
        logger.info(f"Classical solution vector: {classical_solution_vector}")
        logger.info(f"Classical euclidean norm: {classical_euclidean_norm}")
        log_with_time(
            f"[{caseId}] Classical solution computed "
            f"(norm={classical_euclidean_norm:.6f})",
            [workflow_start_time, case_start_time]
        )


        # **************************************************************************
        # 2. circuit execution step - use a venv site for the target backend

        # Check if we should run the circuit
        run_circuit = caseArgs.get('run_circuit', True)

        if not run_circuit:
            log_with_time(
                f"[{caseId}] Phase 2: Transpiling circuit without execution "
                f"(run_circuit=false)",
                [workflow_start_time, case_start_time]
            )
            
            # Transpile the circuit to get transpiled depth
            computeType = caseArgs["qc_backend"]
            runArgs = {
                "measure_all": True,
                "optimization_level": 0,
                "transpile_only": True  # Only transpile, don't execute
            }
            
            log_with_time(
                f"[{caseId}] Submitting transpile-only job",
                [workflow_start_time, case_start_time]
            )
            exec_status = exec_site.getRunDriver().submit(
                JobDefn(
                    circuit_qpy_path.as_posix(),
                    JobDefn.ENTRY_TYPE_STRING,
                    {"format": ".qpy"}
                ),
                JobContext().initialize(
                    f"transpile_{caseId}",
                    wf.getWorkflowId(),
                    exec_site.getSiteName()
                ),
                computeType,
                runArgs
            )
            
            if exec_status is None:
                logger.error(f"Transpile job submission failed {caseId}")
                caseArgs['_circuit_depth_transpiled'] = None
            else:
                exec_status = lwfManager.wait(exec_status.getJobId())
                if (
                    (exec_status is None)
                    or (exec_status.getStatus() != JobStatus.COMPLETE)
                ):
                    logger.error(f"Transpile job failed {caseId}")
                    caseArgs['_circuit_depth_transpiled'] = None
                else:
                    # Load transpiled circuit to get depth
                    transpiled_qpy_path = (
                        circuit_qpy_path.parent
                        / f"{circuit_qpy_path.stem}_transpiled{circuit_qpy_path.suffix}"
                    )
                    if transpiled_qpy_path.exists():
                        with open(transpiled_qpy_path, "rb") as f:
                            transpiled_circuits = qpy_load(f)
                            transpiled_circuit = (
                                transpiled_circuits[0]
                                if isinstance(transpiled_circuits, list)
                                else transpiled_circuits
                            )
                            transpiled_depth = transpiled_circuit.depth()
                            transpiled_size = transpiled_circuit.size()
                            caseArgs['_circuit_depth_transpiled'] = transpiled_depth
                            caseArgs['_circuit_gates_transpiled'] = transpiled_size
                            logger.info(
                                f"[{caseId}] Transpiled: depth={transpiled_depth}, gates={transpiled_size}"
                            )
                    else:
                        logger.warning(
                            f"[{caseId}] Transpiled circuit not found"
                        )
                        caseArgs['_circuit_depth_transpiled'] = None
                        caseArgs['_circuit_gates_transpiled'] = None
            
            # Save circuit/matrix metadata for scaling analysis
            caseArgs['_circuit_qubits'] = num_qubits_circuit
            caseArgs['_circuit_depth'] = circuit_depth
            caseArgs['_circuit_gates'] = circuit_size
            caseArgs['_matrix_size'] = matrix.shape[0]

            # Incremental save: write checkpoint for transpile-only cases
            import json
            checkpoint_data = {
                'case_id': caseId,
                'params': dict(caseArgs),
                'quantum_solution': None,  # No execution in transpile-only mode
                'classical_solution': classical_solution_vector.tolist() if classical_solution_vector is not None else None,
                'metadata': caseArgs.get('_metadata', {})
            }
            checkpoint_file = Path(globalArgs["savedir"]).parent / f"checkpoint_{caseId}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"[{caseId}] Saved checkpoint to {checkpoint_file}")

            case_elapsed = time.time() - case_start_time
            log_with_time(
                f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
                [workflow_start_time, case_start_time]
            )
            continue

        log_with_time(
            f"[{caseId}] Phase 2: Starting circuit execution on "
            f"{caseArgs['qc_backend']}",
            [workflow_start_time, case_start_time]
        )

        computeType = caseArgs["qc_backend"]    # simulators or real machines

        runArgs = {
            "shots": caseArgs["qc_shots"],  # how many shot/samples per run
            "measure_all": True,  # circuit won't have measurement yet
            "optimization_level": 0,  # 0 none, 3 max, transpile opt
        }

        if "_sim_aer" in computeType and caseArgs.get("sim_custom_noise"):
            noise_model_type = caseArgs.get("noise_model", "heron")
            custom_noise_model = add_custom_noise(noise_model_type)
            runArgs["noise_model"] = lwfManager.serialize(custom_noise_model)
            log_with_time(
                f"[{caseId}] Added {noise_model_type} noise model to simulation",
                [workflow_start_time, case_start_time]
            )

        log_with_time(
            f"[{caseId}] Submitting circuit execution job "
            f"({caseArgs['qc_shots']} shots)",
            [workflow_start_time, case_start_time]
        )
        exec_start = time.time()
        exec_status = exec_site.getRunDriver().submit(
            JobDefn(
                circuit_qpy_path.as_posix(),  # run this circuit
                JobDefn.ENTRY_TYPE_STRING,
                {"format": ".qpy"}  # stored in this format
            ),
            JobContext().initialize(
                f"{caseArgs['qc_shots']}",  # in its own job context
                wf.getWorkflowId(),
                exec_site.getSiteName()
            ),
            computeType,  # on this backed
            runArgs  # with these args
        )
        if exec_status is None:
            logger.error(f"Circuit execution job submission failed {caseId}")
            continue    # to next case
        log_with_time(
            f"[{caseId}] Waiting for circuit execution to complete",
            [workflow_start_time, case_start_time]
        )
        exec_status = lwfManager.wait(exec_status.getJobId())
        if (
            (exec_status is None)
            or (exec_status.getStatus() != JobStatus.COMPLETE)
        ):
            logger.error(f"Circuit execution job failed {caseId}")
            continue    # to next case
        exec_time = time.time() - exec_start
        caseArgs['_time_execution'] = exec_time
        log_with_time(f"[{caseId}] Circuit execution complete ({exec_time:.2f}s)", [workflow_start_time, case_start_time])
        lwfManager.notateGet(circuit_qpy_path.as_posix(), exec_status.getJobContext(),
            {"case": caseId})


        # **************************************************************************
        # 3. per-case postprocess step

        # Check if we should do post-processing
        do_postprocess = caseArgs.get('postprocess', True)

        if not do_postprocess:
            log_with_time(
                f"[{caseId}] Skipping post-processing (postprocess=false)",
                [workflow_start_time, case_start_time]
            )
            case_elapsed = time.time() - case_start_time
            log_with_time(
                f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
                [workflow_start_time, case_start_time]
            )
            continue

        log_with_time(f"[{caseId}] Phase 3: Post-processing results", [workflow_start_time, case_start_time])

        result = cast(
            QiskitJobResult, lwfManager.deserialize(exec_status.getNativeInfo())
        )

        # Load transpiled circuit from file to get actual depth
        transpiled_depth = None
        transpiled_qpy_path = (
            circuit_qpy_path.parent
            / f"{circuit_qpy_path.stem}_transpiled{circuit_qpy_path.suffix}"
        )

        if transpiled_qpy_path.exists():
            try:
                with open(transpiled_qpy_path, "rb") as f:
                    transpiled_circuits = qpy_load(f)
                    transpiled_circuit = (
                        transpiled_circuits[0]
                        if isinstance(transpiled_circuits, list)
                        else transpiled_circuits
                    )
                    transpiled_depth = transpiled_circuit.depth()
                    transpiled_size = transpiled_circuit.size()
                    # Get original circuit metadata from transpiled circuit
                    # (since we skipped loading the original QPY in qt02)
                    num_qubits_circuit = transpiled_circuit.num_qubits
                    # Note: transpiled depth/size may differ from original
                    # We'll use transpiled values for both
                    circuit_depth = transpiled_depth
                    circuit_size = transpiled_size
                    # Save transpiled metrics
                    caseArgs['_circuit_gates_transpiled'] = transpiled_size
                    logger.info(
                        f"[{caseId}] Transpiled circuit: "
                        f"qubits={num_qubits_circuit}, "
                        f"depth={transpiled_depth}, gates={transpiled_size}"
                    )
                    logger.info(
                        f"[{caseId}] Transpiled gate breakdown: "
                        f"{transpiled_circuit.count_ops()}"
                    )
            except Exception as e:
                logger.warning(f"[{caseId}] Failed to load transpiled circuit: {e}")
                transpiled_depth = None
                caseArgs['_circuit_gates_transpiled'] = None
        else:
            logger.warning(
                f"[{caseId}] Transpiled circuit not found at: "
                f"{transpiled_qpy_path}"
            )
            caseArgs['_circuit_gates_transpiled'] = None

        # Extract solution from measurement counts
        # Handle different result types from IBM runtime vs simulators
        if hasattr(result, 'data') and callable(result.data):
            # QiskitJobResult from simulators
            counts = (
                result.data()["counts"]
                if "counts" in result.data()
                else result.get_counts()
            )
            theData = result.data()
        else:
            # PrimitiveResult from IBM runtime
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                theData = result
            else:
                # Handle BitArray from IBM runtime
                bit_array = result[0].data.meas
                # Convert BitArray to counts dictionary
                counts = bit_array.get_counts()
                theData = result
        logger.info(f"Case {caseId} - Measurement counts: {counts}")

        # For HHL with measurements, extract solution from middle register
        # (based on HHL structure). Solution qubits are located in the
        # middle of the register, not first or last
        total_shots = sum(counts.values())
        logger.info(f"Case {caseId} - Total shots: {total_shots}")
        n_solution = 2 ** n_qubits_matrix  # Matrix size = 2^n_qubits_matrix
        quantum_solution = np.zeros(n_solution)

        # HHL solution extraction based on observed measurement bitstring
        # pattern. Extract solution from the last n_qubits_matrix bits of
        # each measurement as the HHL circuit places the solution in some,
        # and uses others as ancillas.

        for bitstring, count in counts.items():
            # Convert bitstring to state index (handle both hex and binary formats)
            # Qiskit is going to return the bitstring in hex format
            if bitstring.startswith('0x'):
                state_index = int(bitstring, 16)  # Hexadecimal format
            else:
                state_index = int(bitstring, 2)   # Binary format

            # Extract solution register bits (last n_qubits_matrix bits)
            solution_bits = state_index & ((1 << n_qubits_matrix) - 1)

            if solution_bits < n_solution:
                quantum_solution[solution_bits] += count / total_shots

        # Filter near-zero components (like HHL reference implementation)
        quantum_solution[np.abs(quantum_solution) < 1e-10] = 0

        logger.info(f"Case {caseId} - Raw quantum solution: {quantum_solution}")

        # Normalize and scale to match classical solution
        if np.linalg.norm(quantum_solution) > 0:
            quantum_solution = (
                quantum_solution / np.linalg.norm(quantum_solution)
                * classical_euclidean_norm
            )
            solvec_hhl = quantum_solution
        else:
            logger.warning("Zero norm quantum solution from measurements")
            solvec_hhl = np.zeros(n_solution)

        logger.info(f"Case {caseId}, Solution vector: {solvec_hhl}")
        log_with_time(
            f"[{caseId}] Quantum solution extracted and normalized",
            [workflow_start_time, case_start_time]
        )

        # write result to file in case directory
        result_path = caseOutDir / "results.out"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(str(result))
            f.write(str(theData))
            f.write(str(solvec_hhl))
        lwfManager.notatePut(
            result_path.as_posix(), exec_status.getJobContext(), {"case": caseId}
        )

        # save the job info and solution for postprocessing
        caseResults.append(result)
        quantum_solutions.append(solvec_hhl)
        classical_solutions.append(classical_solution_vector)

        # Save circuit/matrix metadata for scaling analysis
        caseArgs['_circuit_qubits'] = num_qubits_circuit
        caseArgs['_circuit_depth'] = circuit_depth
        caseArgs['_circuit_gates'] = circuit_size
        caseArgs['_circuit_depth_transpiled'] = (
            transpiled_depth if transpiled_depth else circuit_depth
        )
        caseArgs['_matrix_size'] = matrix.shape[0]

        # Incremental save: write checkpoint after each case completes
        import json
        checkpoint_data = {
            'case_id': caseId,
            'params': dict(caseArgs),
            'quantum_solution': solvec_hhl.tolist() if solvec_hhl is not None else None,
            'classical_solution': classical_solution_vector.tolist() if classical_solution_vector is not None else None,
            'metadata': caseArgs.get('_metadata', {})
        }
        checkpoint_file = Path(globalArgs["savedir"]).parent / f"checkpoint_{caseId}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"[{caseId}] Saved checkpoint to {checkpoint_file}")

        case_elapsed = time.time() - case_start_time
        log_with_time(
            f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
            [workflow_start_time, case_start_time]
        )

        # **************************************************************************
        # end of case loop
        # **************************************************************************

    workflow_elapsed = time.time() - workflow_start_time
    log_with_time(
        f"End of case iterations (total workflow time: {workflow_elapsed:.2f}s)",
        [workflow_start_time]
    )
    log_with_time("Ready for workflow post-processing", [workflow_start_time])

    # create symlink to workflow logs
    lwfm_log_dir = Path.home() / ".lwfm" / "logs" / str(wf.getWorkflowId())
    if lwfm_log_dir.exists():
        log_symlink = workflow_out_dir / "workflow.log"
        if not log_symlink.exists():
            os.symlink(lwfm_log_dir, log_symlink)
            logger.info(f"Created log symlink: {log_symlink} -> {lwfm_log_dir}")

    return (
        wf, caseResults, quantum_solutions, classical_solutions,
        casesArgs, globalArgs, exec_status
    )
