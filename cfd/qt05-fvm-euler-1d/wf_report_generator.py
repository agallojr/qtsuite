"""
Workflow Report Generator
Collects and formats comprehensive workflow execution reports.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class WorkflowReportGenerator:
    """Generate comprehensive reports for workflow execution."""
    
    def __init__(self, casedir: str, case_id: str):
        self.casedir = Path(casedir)
        self.case_id = case_id
        self.report_data: Dict[str, Any] = {
            'metadata': {
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
                'casedir': str(casedir)
            },
            'cfd_problem': {},
            'run_parameters': {},
            'quantum_parameters': {},
            'iterations': [],
            'results': {},
            'performance': {}
        }
    
    def collect_cfd_problem_info(self, case_args: Dict[str, Any]):
        """Collect CFD problem definition.
        
        Note: Physics constants, geometry, and flow conditions are hardcoded here
        to match the hardcoded values in case.py (SimCase and Nozzle1D classes).
        If case.py changes, these must be updated manually.
        """
        self.report_data['cfd_problem'] = {
            'description': '1D Compressible Euler flow in supersonic diverging nozzle',
            'geometry': {
                'type': 'supersonic_diverging_nozzle',
                'area_function': 'S(x) = 1.398 + 0.347*tanh(0.8x - 4)',  # From case.py line 80
                'domain': f"x ∈ [0, {case_args.get('x_domain', [0, 10])[1]}]",
                'mesh_elements': case_args.get('nelem', 200)
            },
            'physics': {
                'equations': 'Compressible Euler (inviscid gas dynamics)',
                'gamma': 1.4,           # From case.py line 8
                'gas_constant': 1716,  # From case.py line 9
                'dimensions': 1
            },
            'flow_conditions': {
                'inlet_mach': 1.5,      # From case.py line 13
                'inlet_pressure': 2000,  # From case.py line 10
                'inlet_temperature': 520,  # From case.py line 11
                'boundary_conditions': {
                    'inlet': 'supersonic_inlet',  # From case.py line 20
                    'outlet': 'extrapolation'  # From case.py line 21
                }
            }
        }
    
    def collect_run_parameters(self, case_args: Dict[str, Any]):
        """Collect numerical method parameters."""
        self.report_data['run_parameters'] = {
            'time_scheme': case_args.get('scheme', 'BDF1'),
            'linear_solver': case_args.get('linsolver', 'LU'),
            'cfl_number': case_args.get('cfl', 1e10),
            'mesh': {
                'elements': case_args.get('nelem', 200),
                'domain': case_args.get('x_domain', [0, 10])
            },
            'convergence': {
                'max_outer_iterations': case_args.get('max_iters', 2000),
                'max_inner_iterations': case_args.get('max_inner_iters', 10),
                'residual_tolerance': case_args.get('res_tol', 1e-2),
                'solution_tolerance': case_args.get('conv_tol', 1e-12)
            },
            'flags': {
                'local_timestepping': case_args.get('localdt', False),
                'nondimensional': True
            }
        }
    
    def collect_quantum_parameters(self, case_args: Dict[str, Any]):
        """Collect quantum execution parameters."""
        self.report_data['quantum_parameters'] = {
            'enabled': case_args.get('linsolver') == 'HHL',
            'backend': case_args.get('quantum_backend', 'automatic_sim_aer'),
            'shots': case_args.get('quantum_shots', 1024),
            'transpilation': {
                'optimization_level': case_args.get('quantum_transpile_opt', 1)
            }
        }
    
    def add_iteration_data(self, iter_num: int, subiter_num: int, stats_file: Path):
        """Add data from a circuit execution iteration."""
        if not stats_file.exists():
            return
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # Diagnostic logging for None values
        if stats.get('pre_transpile', {}).get('depth') is None:
            print(f"[WARNING] pre_transpile depth is None in {stats_file}")
        if stats.get('post_transpile', {}).get('depth') is None:
            print(f"[WARNING] post_transpile depth is None in {stats_file}")
            print(f"[DEBUG] post_transpile keys: {stats.get('post_transpile', {}).keys()}")
            print(f"[DEBUG] post_transpile values: {stats.get('post_transpile', {})}")
        if stats.get('pre_transpile', {}).get('num_gates') is None:
            print(f"[WARNING] pre_transpile num_gates is None in {stats_file}")
        if stats.get('post_transpile', {}).get('num_gates') is None:
            print(f"[WARNING] post_transpile num_gates is None in {stats_file}")
        
        iter_data = {
            'outer_iteration': iter_num,
            'inner_iteration': subiter_num,
            'matrix': stats.get('matrix', {}),
            'circuit_pre_transpile': {
                'logical_qubits': stats['pre_transpile']['num_qubits'],
                'logical_depth': stats['pre_transpile']['depth'],
                'logical_gates': stats['pre_transpile']['num_gates'],
                'gate_breakdown': stats['pre_transpile']['gate_breakdown']
            },
            'circuit_post_transpile': {
                'physical_qubits': stats['post_transpile']['num_qubits'],
                'physical_depth': stats['post_transpile']['depth'],
                'physical_gates': stats['post_transpile']['num_gates'],
                'gate_breakdown': stats['post_transpile']['gate_breakdown']
            },
            'transpilation_impact': {
                'depth_ratio': (
                    (stats['post_transpile']['depth'] / stats['pre_transpile']['depth'])
                    if (stats['pre_transpile']['depth'] and
                        stats['post_transpile']['depth'] and
                        stats['pre_transpile']['depth'] > 0) else 0
                ),
                'gates_ratio': (
                    (stats['post_transpile']['num_gates'] / stats['pre_transpile']['num_gates'])
                    if (stats['pre_transpile']['num_gates'] and
                        stats['post_transpile']['num_gates'] and
                        stats['pre_transpile']['num_gates'] > 0) else 0
                )
            },
            'solution_metrics': stats.get('solution_metrics', {}),
            'timing': stats.get('timing', {})
        }
        
        self.report_data['iterations'].append(iter_data)
    
    def collect_all_iterations(self):
        """Scan directory for all iteration statistics files."""
        stats_files = sorted(self.casedir.glob('circuit_stats_iter*_subiter*.json'))
        for stats_file in stats_files:
            # Parse filename to get iter and subiter numbers
            name = stats_file.stem
            parts = name.replace('circuit_stats_iter', '').replace('_subiter', '_').split('_')
            if len(parts) >= 2:
                iter_num = int(parts[0])
                subiter_num = int(parts[1])
                self.add_iteration_data(iter_num, subiter_num, stats_file)
    
    def compute_aggregate_results(self):
        """Compute aggregate results across all iterations."""
        if not self.report_data['iterations']:
            return
        
        iterations = self.report_data['iterations']
        
        # Matrix condition numbers and sizes over time
        condition_numbers = [
            it['matrix']['condition_number'] for it in iterations 
            if 'matrix' in it and 'condition_number' in it['matrix']
        ]
        original_sizes = [
            it['matrix'].get('original_size', it['matrix']['shape'][0])
            for it in iterations if 'matrix' in it
        ]
        padded_sizes = [
            it['matrix'].get('padded_size', it['matrix']['shape'][0])
            for it in iterations if 'matrix' in it
        ]
        
        # Circuit complexity metrics (filter out None values)
        logical_depths = [
            it['circuit_pre_transpile']['logical_depth'] for it in iterations
            if it.get('circuit_pre_transpile', {}).get('logical_depth') is not None
        ]
        physical_depths = [
            it['circuit_post_transpile']['physical_depth'] for it in iterations
            if it.get('circuit_post_transpile', {}).get('physical_depth') is not None
        ]
        physical_gates = [
            it['circuit_post_transpile']['physical_gates'] for it in iterations
            if it.get('circuit_post_transpile', {}).get('physical_gates') is not None
        ]
        
        # Solution accuracy
        relative_errors = [
            it['solution_metrics']['relative_error'] for it in iterations
            if 'solution_metrics' in it 
            and it['solution_metrics'].get('relative_error') is not None
        ]
        
        # Timing statistics
        timing_data = {}
        for key in ['total_elapsed', 'file_loading', 'matrix_analysis', 'transpilation', 
                    'circuit_execution', 'solution_extraction', 'metrics_computation']:
            values = [
                it['timing'][key] for it in iterations
                if 'timing' in it and key in it['timing'] and it['timing'][key] is not None
            ]
            if values:
                timing_data[key] = {
                    'values': values,
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'total': float(np.sum(values))
                }
        
        self.report_data['results'] = {
            'total_iterations': len(iterations),
            'matrix_properties': {
                'original_sizes': original_sizes,
                'padded_sizes': padded_sizes,
                'condition_numbers': condition_numbers,
                'mean_condition': float(np.mean(condition_numbers)) if condition_numbers else None,
                'max_condition': float(np.max(condition_numbers)) if condition_numbers else None
            },
            'circuit_complexity': {
                'logical_depth': {
                    'values': logical_depths,
                    'mean': float(np.mean(logical_depths)) if logical_depths else None,
                    'max': int(np.max(logical_depths)) if logical_depths else None
                },
                'physical_depth': {
                    'values': physical_depths,
                    'mean': float(np.mean(physical_depths)) if physical_depths else None,
                    'max': int(np.max(physical_depths)) if physical_depths else None
                },
                'physical_gates': {
                    'values': physical_gates,
                    'mean': float(np.mean(physical_gates)) if physical_gates else None,
                    'max': int(np.max(physical_gates)) if physical_gates else None
                }
            },
            'solution_accuracy': {
                'relative_errors': relative_errors,
                'mean_relative_error': float(np.mean(relative_errors)) if relative_errors else None,
                'max_relative_error': float(np.max(relative_errors)) if relative_errors else None
            },
            'timing': timing_data
        }
    
    def save_json_report(self, filename: str = 'case_report.json'):
        """Save machine-readable JSON report."""
        report_path = self.casedir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2)
        print(f"[{self.case_id}] JSON report saved to {report_path}")
        return report_path
    
    def save_text_report(self, filename: str = 'case_report.txt'):
        """Save human-readable text report."""
        report_path = self.casedir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CASE REPORT: QUANTUM-CLASSICAL HYBRID CFD SOLVER\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"Case ID:     {self.report_data['metadata']['case_id']}\n")
            f.write(f"Timestamp:   {self.report_data['metadata']['timestamp']}\n")
            f.write(f"Directory:   {self.report_data['metadata']['casedir']}\n\n")
            
            # CFD Problem
            f.write("CFD PROBLEM DEFINITION\n")
            f.write("-" * 80 + "\n")
            cfd = self.report_data['cfd_problem']
            f.write(f"Description: {cfd.get('description', 'N/A')}\n")
            f.write(f"Geometry:    {cfd['geometry']['type']}\n")
            f.write(f"             {cfd['geometry']['area_function']}\n")
            f.write(f"Domain:      {cfd['geometry']['domain']}\n")
            f.write(f"Mesh:        {cfd['geometry']['mesh_elements']} elements\n")
            f.write(f"Equations:   {cfd['physics']['equations']}\n")
            f.write(f"Inlet Mach:  {cfd['flow_conditions']['inlet_mach']}\n\n")
            
            # Run Parameters
            f.write("NUMERICAL METHOD PARAMETERS\n")
            f.write("-" * 80 + "\n")
            run = self.report_data['run_parameters']
            f.write(f"Time Scheme:        {run['time_scheme']}\n")
            f.write(f"Linear Solver:      {run['linear_solver']}\n")
            f.write(f"CFL Number:         {run['cfl_number']:.2e}\n")
            f.write(f"Mesh Elements:      {run['mesh']['elements']}\n")
            f.write(f"Max Outer Iters:    {run['convergence']['max_outer_iterations']}\n")
            f.write(f"Max Inner Iters:    {run['convergence']['max_inner_iterations']}\n")
            f.write(f"Local Timestepping: {run['flags']['local_timestepping']}\n\n")
            
            # Quantum Parameters
            if self.report_data['quantum_parameters']['enabled']:
                f.write("QUANTUM EXECUTION PARAMETERS\n")
                f.write("-" * 80 + "\n")
                qp = self.report_data['quantum_parameters']
                f.write(f"Backend:             {qp['backend']}\n")
                f.write(f"Shots:               {qp['shots']}\n")
                f.write(f"Transpile Opt Level: {qp['transpilation']['optimization_level']}\n\n")
            
            # Results Summary
            if self.report_data['results']:
                f.write("RESULTS SUMMARY\n")
                f.write("-" * 80 + "\n")
                res = self.report_data['results']
                f.write(f"Total Iterations: {res['total_iterations']}\n\n")
                
                # Matrix properties
                if res['matrix_properties']['mean_condition']:
                    f.write(f"Matrix Properties:\n")
                    orig_sizes = res['matrix_properties']['original_sizes']
                    padded_sizes = res['matrix_properties']['padded_sizes']
                    
                    # Check if sizes vary
                    if len(set(orig_sizes)) == 1:
                        f.write(f"  Size (original): {orig_sizes[0]}x{orig_sizes[0]}\n")
                    else:
                        f.write(f"  Size (original): {orig_sizes}\n")
                    
                    if len(set(padded_sizes)) == 1:
                        f.write(
                f"  Size (padded):   {padded_sizes[0]}x{padded_sizes[0]}\n"
            )
                    else:
                        f.write(f"  Size (padded):   {padded_sizes}\n")
                    
                    f.write(
                        f"  Condition number (mean): "
                        f"{res['matrix_properties']['mean_condition']:.4e}\n"
                    )
                    f.write(
                        f"  Condition number (max):  "
                        f"{res['matrix_properties']['max_condition']:.4e}\n\n"
                    )
                
                if res['circuit_complexity']['physical_depth']['mean']:
                    f.write("Circuit Complexity (Physical Gates):\n")
                    f.write(
                        f"  Mean Depth: {res['circuit_complexity']['physical_depth']['mean']:.0f}\n"
                    )
                    f.write(
                        f"  Max Depth:  {res['circuit_complexity']['physical_depth']['max']}\n"
                    )
                    f.write(
                        f"  Mean Gates: {res['circuit_complexity']['physical_gates']['mean']:.0f}\n"
                    )
                    f.write(
                        f"  Max Gates:  {res['circuit_complexity']['physical_gates']['max']}\n\n"
                    )
                
                if res['solution_accuracy']['mean_relative_error']:
                    f.write("Solution Accuracy (vs Classical):\n")
                    accuracy = res['solution_accuracy']
                    f.write(
                        f"  Mean Relative Error: {accuracy['mean_relative_error']:.4e}\n"
                    )
                    f.write(
                        f"  Max Relative Error:  {accuracy['max_relative_error']:.4e}\n\n"
                    )
                
                # Timing summary
                if res.get('timing'):
                    f.write(f"Performance Timing:\n")
                    timing = res['timing']
                    if 'total_elapsed' in timing:
                        if 'circuit_execution' in timing:
                            exec_time = timing['circuit_execution']
                            total = exec_time.get('total', 0)
                            mean = exec_time.get('mean', 0)
                            f.write(
                                f"  Circuit Execution:    {total:.3f} s "
                                f"(mean: {mean:.3f} s)\n"
                            )
                    timing_breakdown = []
                    if 'transpilation' in timing:
                        timing_breakdown.append(f"trans={timing['transpilation']:.3f}s")
                    if 'circuit_execution' in timing:
                        timing_breakdown.append(f"exec={timing['circuit_execution']:.3f}s")
                    if 'matrix_analysis' in timing:
                        timing_breakdown.append(f"matrix={timing['matrix_analysis']:.3f}s")
                    
                    if timing_breakdown:
                        f.write(f"    ({', '.join(timing_breakdown)})\n")
                    
                    # Show per-iteration details if available
                    if 'iterations' in res and res['iterations']:
                        f.write("\nPer-Iteration Details:\n")
                        f.write("-" * 80 + "\n")
                        
                        for it in res['iterations']:
                            # Matrix info
                            matrix = it.get('matrix', {})
                            orig_size = matrix.get('original_size', 
                                matrix.get('shape', [0])[0] if matrix.get('shape') else 0)
                            padded_size = matrix.get('padded_size', 
                                matrix.get('shape', [0])[0] if matrix.get('shape') else 0)
                            
                            f.write(
                                f"Iteration {it.get('outer_iteration')}:"
                                f"{it.get('inner_iteration', '')}\n"
                            )
                            
                            if matrix:
                                f.write(
                                    f"  Matrix size:      {orig_size}x{orig_size} "
                                    f"(padded to {padded_size}x{padded_size})\n"
                                )
                                if 'condition_number' in matrix:
                                    f.write(
                                        f"  Matrix condition: {matrix['condition_number']:.4e}\n"
                                    )
                            
                            # Circuit info
                            pre = it.get('circuit_pre_transpile', {})
                            post = it.get('circuit_post_transpile', {})
                            
                            if pre:
                                f.write(
                                    f"  Logical circuit:  "
                                    f"{pre.get('logical_qubits', 'N/A')} qubits, "
                                    f"{pre.get('logical_depth', 'N/A')} depth\n"
                                )
                            
                            if post:
                                f.write(
                                    f"  Physical circuit: "
                                    f"{post.get('physical_qubits', 'N/A')} qubits, "
                                    f"{post.get('physical_depth', 'N/A')} depth, "
                                    f"{post.get('physical_gates', 'N/A')} gates\n"
                                )
                            
                            # Solution metrics
                            metrics = it.get('solution_metrics', {})
                            rel_err = metrics.get('relative_error')
                            if rel_err is not None:
                                f.write(f"  Relative error:   {rel_err:.4e}\n")
                            
                            # Timing info
                            timing = it.get('timing', {})
                            if timing:
                                breakdown = []
                                for key, label in [
                                    ('transpilation', 'trans'),
                                    ('circuit_execution', 'exec'),
                                    ('matrix_analysis', 'matrix')
                                ]:
                                    if key in timing and timing[key] is not None:
                                        breakdown.append(f"{label}={timing[key]:.3f}s")
                                
                                if breakdown:
                                    f.write(f"  Time: ({', '.join(breakdown)})\n")
                            
                            f.write("\n")  # Add spacing between iterations
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"[{self.case_id}] Text report saved to {report_path}")
        return report_path
    
    def generate_complete_report(self, case_args: Dict[str, Any]):
        """Generate complete workflow report."""
        print(f"[{self.case_id}] Generating workflow report...")
        
        # Collect all information
        self.collect_cfd_problem_info(case_args)
        self.collect_run_parameters(case_args)
        self.collect_quantum_parameters(case_args)
        self.collect_all_iterations()
        self.compute_aggregate_results()
        
        # Save reports
        json_path = self.save_json_report()
        text_path = self.save_text_report()
        
        print(f"[{self.case_id}] Report generation complete!")
        return json_path, text_path
