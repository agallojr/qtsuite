        if not run_circuit:
            log_with_time(logger,
                f"[{caseId}] Phase 2: Transpiling circuit without execution "
                f"(run_circuit=false)",
                case_start_time)

            # Transpile the circuit to get transpiled depth
            computeType = caseArgs["qc_backend"]
            runArgs = {
                "measure_all": True,
                "optimization_level": 0,
                "transpile_only": True  # Only transpile, don't execute
            }

            log_with_time(logger,
                f"[{caseId}] Submitting transpile-only job",
                case_start_time)
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
                            caseArgs['_circuit_depth_transpiled'] = transpiled_depth
                            logger.info(
                                f"[{caseId}] Transpiled depth: {transpiled_depth}"
                            )
                    else:
                        logger.warning(
                            f"[{caseId}] Transpiled circuit not found"
                        )
                        caseArgs['_circuit_depth_transpiled'] = None

            # Save circuit/matrix metadata for scaling analysis
            caseArgs['_circuit_qubits'] = num_qubits_circuit
            caseArgs['_circuit_depth'] = circuit_depth
            caseArgs['_matrix_size'] = matrix.shape[0]










        log_with_time(logger,
            f"[{caseId}] Phase 2: Starting circuit execution on "
            f"{caseArgs['qc_backend']}",
            case_start_time)

        computeType = caseArgs["qc_backend"]    # simulators or real machines

        runArgs = {
            "shots": caseArgs["qc_shots"],  # how many shot/samples per run
            "measure_all": True,  # circuit won't have measurement yet
            "optimization_level": 0,  # 0 none, 3 max, transpile opt
        }

        log_with_time(logger,
            f"[{caseId}] Submitting circuit execution job "
            f"({caseArgs['qc_shots']} shots)",
            case_start_time)
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
        log_with_time(logger,
            f"[{caseId}] Waiting for circuit execution to complete",
            case_start_time)
        exec_status = lwfManager.wait(exec_status.getJobId())
        if (
            (exec_status is None)
            or (exec_status.getStatus() != JobStatus.COMPLETE)
        ):
            logger.error(f"Circuit execution job failed {caseId}")
            continue    # to next case