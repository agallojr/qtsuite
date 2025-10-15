"""
test of hhl with classiq
"""

from classiq import *
from classiq.qmod.symbolic import floor, log

import numpy as np
import scipy

a_matrix = np.array(
    [
        [0.135, -0.092, -0.011, -0.045, -0.026, -0.033, 0.03, 0.034],
        [-0.092, 0.115, 0.02, 0.017, 0.044, -0.009, -0.015, -0.072],
        [-0.011, 0.02, 0.073, -0.0, -0.068, -0.042, 0.043, -0.011],
        [-0.045, 0.017, -0.0, 0.043, 0.028, 0.027, -0.047, -0.005],
        [-0.026, 0.044, -0.068, 0.028, 0.21, 0.079, -0.177, -0.05],
        [-0.033, -0.009, -0.042, 0.027, 0.079, 0.121, -0.123, 0.021],
        [0.03, -0.015, 0.043, -0.047, -0.177, -0.123, 0.224, 0.011],
        [0.034, -0.072, -0.011, -0.005, -0.05, 0.021, 0.011, 0.076],
    ]
)

b_vector = np.array(
    [
        -0.00885448,
        -0.17725898,
        -0.15441119,
        0.17760157,
        0.41428775,
        0.44735303,
        -0.71137715,
        0.1878808,
    ]
)

sol_classical = np.linalg.solve(a_matrix, b_vector)  # classical solution

# number of qubits for the unitary
num_qubits = int(np.log2(len(b_vector)))
# exact unitary
my_unitary = scipy.linalg.expm(1j * 2 * np.pi * a_matrix)

transpilation_options = {"classiq": "auto optimize", "qiskit": 1}

@qfunc
def simple_eig_inv(phase: QNum, indicator: Output[QBit]):
    allocate(indicator)
    indicator *= (1 / 2**phase.size) / phase


@qfunc
def my_hhl(
    precision: CInt,
    b: CArray[CReal],
    unitary: QCallable[QArray],
    res: Output[QArray],
    phase: Output[QNum],
    indicator: Output[QBit],
) -> None:
    prepare_amplitudes(b, 0.0, res)
    allocate(precision, False, precision, phase)
    within_apply(
        lambda: qpe(unitary=lambda: unitary(res), phase=phase),
        lambda: simple_eig_inv(phase=phase, indicator=indicator),
    )


def get_classiq_hhl_results(precision):
    """
    This function models, synthesizes, executes an HHL example and returns the depth,
    cx-counts and fidelity
    """

    # SP params
    b_normalized = b_vector.tolist()
    sp_upper = 0.00  # precision of the State Preparation
    unitary_mat = my_unitary.tolist()
    size = (len(b_normalized) - 1).bit_length()

    @qfunc
    def main(res: Output[QNum], phase_var: Output[QNum], indicator: Output[QBit]):
        my_hhl(
            precision=precision,
            b=b_normalized,
            unitary=lambda target: unitary(elements=unitary_mat, target=target),
            res=res,
            phase=phase_var,
            indicator=indicator,
        )

    # Synthesize
    preferences = Preferences(
        custom_hardware_settings=CustomHardwareSettings(basis_gates=["cx", "u"]),
        transpilation_option=transpilation_options["classiq"],
    )
    qprog_hhl = synthesize(main, preferences=preferences)
    total_q = qprog_hhl.data.width  # total number of qubits of the whole circuit
    depth = qprog_hhl.transpiled_circuit.depth
    cx_counts = qprog_hhl.transpiled_circuit.count_ops["cx"]

    # Execute
    backend_preferences = ClassiqBackendPreferences(
        backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR
    )
    execution_preferences = ExecutionPreferences(
        num_shots=1, backend_preferences=backend_preferences
    )

    with ExecutionSession(qprog_hhl, execution_preferences) as es:
        result = es.sample()

    df = result.dataframe
    qsol = np.zeros(2**size, dtype=complex)

    # Post-process
    # Filter only the successful states.
    filtered_st = df[
        (df.indicator == 1) & (df.phase_var == 0) & (np.abs(df.amplitude) > 1e-12)
    ]

    # Allocate values
    qsol[filtered_st.res] = filtered_st.amplitude / (1 / 2**precision)

    fidelity = (
        np.abs(
            np.dot(
                sol_classical / np.linalg.norm(sol_classical),
                qsol / np.linalg.norm(qsol),
            )
        )
        ** 2
    )
    return total_q, depth, cx_counts, fidelity

if __name__ == "__main__":
    classiq_widths = []
    classiq_depths = []
    classiq_cx_counts = []
    classiq_fidelities = []
    for per in range(2, 9):
        total_q, depth, cx_counts, fidelity = get_classiq_hhl_results(per)
        classiq_widths.append(total_q)
        classiq_depths.append(depth)
        classiq_cx_counts.append(cx_counts)
        classiq_fidelities.append(fidelity)
    print("classiq overlap:", classiq_fidelities)
    print("classiq depth:", classiq_depths)
