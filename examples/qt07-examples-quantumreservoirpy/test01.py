from quantumreservoirpy.reservoirs import Static

class CustomRes(Static):
   def before(self, circuit):
      circuit.h(circuit.qubits)
      circuit.barrier()

   def during(self, circuit, timestep):
      circuit.initialize(str(timestep), [0])
      circuit.h(0)
      circuit.cx(0, 1)

   def after(self, circuit):
      circuit.barrier()
      circuit.measure_all()

res = CustomRes(n_qubits=2)
res.circuit([0, 1]).draw('mpl')
