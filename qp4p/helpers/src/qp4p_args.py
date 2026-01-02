"""
Common argument parser helpers for quantum scripts.

Provides standard argument groups for noise, backend, and execution parameters.
"""

import argparse


def add_noise_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard noise-related arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in µs (default: None = no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in µs (default: None = no noise)")


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard backend and coupling map arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--backend", type=str, default=None,
                        help="Fake backend name (e.g., 'manila', 'jakarta')")
    parser.add_argument("--coupling-map", type=str, default="default",
                        choices=["default", "all-to-all"],
                        help="Coupling map: default (backend's native) or all-to-all (full connectivity) (default: default)")


def add_execution_args(parser: argparse.ArgumentParser, default_shots: int = 1024) -> None:
    """
    Add standard execution arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        default_shots: Default number of shots
    """
    parser.add_argument("--shots", type=int, default=default_shots,
                        help=f"Number of shots (default: {default_shots})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")


def add_standard_quantum_args(parser: argparse.ArgumentParser, 
                               default_shots: int = 1024,
                               include_shots: bool = True) -> None:
    """
    Add all standard quantum execution arguments to parser.
    
    Includes: t1, t2, backend, coupling-map, shots (optional), seed
    
    Args:
        parser: ArgumentParser to add arguments to
        default_shots: Default number of shots
        include_shots: Whether to include shots and seed arguments
    """
    add_noise_args(parser)
    add_backend_args(parser)
    if include_shots:
        add_execution_args(parser, default_shots)
