"""
test of "quantum reservoir py"
"""

#pylint: disable=missing-docstring

from quantumreservoirpy.reservoirs import Static

class CustomRes(Static):
    def before(self, circuit):
        circuit.h(circuit.qubits)
        circuit.barrier()

    def during(self, circuit, timestep, reservoir_number):
        circuit.initialize(str(timestep), [0])
        circuit.h(0)
        circuit.cx(0, 1)

    def after(self, circuit):
        circuit.barrier()
        circuit.measure_all()

res = CustomRes(n_qubits=20)
print(res.circuit([0, 1]).draw('text'))
