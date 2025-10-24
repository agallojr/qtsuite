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
                    logger.info(
                        f"[{caseId}] Transpiled circuit: "
                        f"depth={transpiled_depth}, gates={transpiled_size}"
                    )
                    logger.info(
                        f"[{caseId}] Transpiled gate breakdown: "
                        f"{transpiled_circuit.count_ops()}"
                    )
            except Exception as e:
                logger.warning(f"[{caseId}] Failed to load transpiled circuit: {e}")
                transpiled_depth = None
        else:
            logger.warning(
                f"[{caseId}] Transpiled circuit not found at: "
                f"{transpiled_qpy_path}"
            )

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
            case_start_time
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
        caseArgs['_circuit_depth_transpiled'] = (
            transpiled_depth if transpiled_depth else circuit_depth
        )
        caseArgs['_matrix_size'] = matrix.shape[0]