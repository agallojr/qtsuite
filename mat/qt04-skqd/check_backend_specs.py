"""Fetch and compare IBM Quantum backend specifications for Boston and Kingston"""

from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize service
service = QiskitRuntimeService()

backends = ['ibm_boston', 'ibm_kingston']

print("="*80)
print("IBM QUANTUM BACKEND SPECIFICATIONS COMPARISON")
print("="*80)

for backend_name in backends:
    try:
        backend = service.backend(backend_name)
        config = backend.configuration()
        props = backend.properties() if hasattr(backend, 'properties') else None
        
        print(f"\n{'='*80}")
        print(f"BACKEND: {backend_name.upper()}")
        print(f"{'='*80}")
        
        # Basic info
        print(f"\n📊 BASIC SPECIFICATIONS:")
        print(f"  Backend version: {config.backend_version}")
        print(f"  Number of qubits: {config.n_qubits}")
        print(f"  Quantum volume: {getattr(config, 'quantum_volume', 'N/A')}")
        print(f"  Processor type: {getattr(config, 'processor_type', 'N/A')}")
        
        # Gate information
        print(f"\n⚡ GATE SET:")
        print(f"  Basis gates: {config.basis_gates}")
        print(f"  Conditional: {config.conditional}")
        print(f"  Open pulse: {config.open_pulse}")
        
        # Coupling map info
        print(f"\n🔗 CONNECTIVITY:")
        if hasattr(config, 'coupling_map') and config.coupling_map:
            print(f"  Coupling map length: {len(config.coupling_map)} connections")
            print(f"  Sample couplings: {config.coupling_map[:5]}...")
        
        # Timing
        print(f"\n⏱️  TIMING:")
        if hasattr(config, 'dt'):
            print(f"  dt (sample time): {config.dt} seconds")
        if hasattr(config, 'rep_delay_range'):
            print(f"  Rep delay range: {config.rep_delay_range}")
        
        # Properties (error rates, etc.)
        if props:
            print(f"\n📉 ERROR RATES:")
            
            # Get gate errors
            gate_errors = []
            for gate in props.gates:
                if gate.gate == 'cx':  # Two-qubit gate
                    gate_errors.append(gate.parameters[0].value)
            
            if gate_errors:
                avg_cx_error = sum(gate_errors) / len(gate_errors)
                print(f"  Average CX gate error: {avg_cx_error:.6f}")
                print(f"  Min CX error: {min(gate_errors):.6f}")
                print(f"  Max CX error: {max(gate_errors):.6f}")
            
            # Readout errors
            readout_errors = [q.readout_error for q in props.qubits]
            avg_readout = sum(readout_errors) / len(readout_errors)
            print(f"  Average readout error: {avg_readout:.6f}")
            print(f"  Min readout error: {min(readout_errors):.6f}")
            print(f"  Max readout error: {max(readout_errors):.6f}")
            
            # T1 and T2 times
            t1_times = [q.t1 for q in props.qubits]
            t2_times = [q.t2 for q in props.qubits]
            avg_t1 = sum(t1_times) / len(t1_times)
            avg_t2 = sum(t2_times) / len(t2_times)
            print(f"\n🕐 COHERENCE TIMES:")
            print(f"  Average T1: {avg_t1*1e6:.2f} µs")
            print(f"  Average T2: {avg_t2*1e6:.2f} µs")
            print(f"  Min T1: {min(t1_times)*1e6:.2f} µs")
            print(f"  Max T1: {max(t1_times)*1e6:.2f} µs")
            
        # Status
        status = backend.status()
        print(f"\n🔴 STATUS:")
        print(f"  Operational: {status.operational}")
        print(f"  Pending jobs: {status.pending_jobs}")
        print(f"  Status msg: {status.status_msg}")
        
    except Exception as e:
        print(f"\n❌ Error fetching {backend_name}: {e}")

print(f"\n{'='*80}")
print("COMPARISON COMPLETE")
print(f"{'='*80}\n")
