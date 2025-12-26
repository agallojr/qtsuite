"""
Chemistry helper functions and molecule definitions.

Based on PennyLane's "Top 20 molecules for quantum computing":
https://pennylane.ai/blog/2024/01/top-20-molecules-for-quantum-computing
"""

import numpy as np
from pyscf import gto, scf, fci, ao2mo

# Predefined molecule geometries (atom string templates with {d} for bond length)
MOLECULES = {
    # === Small molecules (2-4 electrons) ===
    "H2": {
        "atoms": "H 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 0.74,
        "charge": 0,
        "spin": 0,
        "description": "Hydrogen - the 'hello world' of quantum chemistry"
    },
    "HeH+": {
        "atoms": "He 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 0.93,
        "charge": 1,
        "spin": 0,
        "description": "Helium hydride cation"
    },
    "LiH": {
        "atoms": "Li 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 1.6,
        "charge": 0,
        "spin": 0,
        "description": "Lithium hydride"
    },
    "BeH2": {
        "atoms": "Be 0 0 0; H 0 0 {d}; H 0 0 -{d}",
        "basis": "sto-3g",
        "default_bond": 1.3,
        "charge": 0,
        "spin": 0,
        "description": "Beryllium dihydride - linear molecule"
    },
    # === Intermediate molecules (8-16 electrons) ===
    "CH2": {
        "atoms": "C 0 0 0; H 0 1.08 0; H 0 -0.54 0.935",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed geometry (bent ~102°)
        "charge": 0,
        "spin": 0,  # Singlet state
        "description": "Methylene - diradical, singlet-triplet gap benchmark"
    },
    "CH2_triplet": {
        "atoms": "C 0 0 0; H 0 1.08 0; H 0 -0.54 0.935",
        "basis": "sto-3g",
        "default_bond": None,
        "charge": 0,
        "spin": 2,  # Triplet state (2S = 2)
        "description": "Methylene triplet state"
    },
    "H2O": {
        "atoms": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed geometry
        "charge": 0,
        "spin": 0,
        "description": "Water - benchmark for double dissociation"
    },
    "NH3": {
        "atoms": "N 0 0 0; H 0 0.939 -0.382; H 0.813 -0.470 -0.382; H -0.813 -0.470 -0.382",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed pyramidal geometry
        "charge": 0,
        "spin": 0,
        "description": "Ammonia - pyramidal inversion barrier benchmark"
    },
    "C2": {
        "atoms": "C 0 0 0; C 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 1.24,
        "charge": 0,
        "spin": 0,
        "description": "Dicarbon - strong static correlation, interstellar molecule"
    },
    "N2": {
        "atoms": "N 0 0 0; N 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 1.10,
        "charge": 0,
        "spin": 0,
        "description": "Nitrogen - triple bond dissociation benchmark"
    },
    "C2H4": {
        "atoms": "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed planar geometry
        "charge": 0,
        "spin": 0,
        "description": "Ethylene - double bond rotation, most produced organic compound"
    },
    "CH2S": {
        "atoms": "C 0 0 0; S 0 0 1.61; H 0 0.94 -0.59; H 0 -0.94 -0.59",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed geometry
        "charge": 0,
        "spin": 0,
        "description": "Thioformaldehyde - excited state benchmark, intersystem crossing"
    },
    # === Larger molecules (24 electrons) - computationally demanding ===
    "O3": {
        "atoms": "O 0 0 0; O 0 1.08 0.70; O 0 -1.08 0.70",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed bent geometry (~117°)
        "charge": 0,
        "spin": 0,
        "description": "Ozone - strong static correlation, atmosphere chemistry"
    }
}


def build_hamiltonian(molecule: str | dict = "H2", bond_length: float = None):
    """
    Build molecular Hamiltonian using PySCF FCI.
    
    Args:
        molecule: Either a molecule name (str) from MOLECULES, or a dict with keys:
                  atoms, basis, charge, spin, default_bond (optional), description (optional)
        bond_length: Bond length in Angstroms (None = use default)
    
    Returns:
        hamiltonian_matrix: Hamiltonian as numpy matrix
        fci_energy: Exact FCI ground state energy
        scf_energy: SCF reference energy
        mol_info: dict with molecule details
    """
    # Handle molecule as string (lookup) or dict (direct definition)
    if isinstance(molecule, str):
        if molecule not in MOLECULES:
            raise ValueError(f"Unknown molecule: {molecule}. Available: {list(MOLECULES.keys())}")
        mol_def = MOLECULES[molecule]
        mol_name = molecule
    else:
        mol_def = molecule
        mol_name = mol_def.get("description", "custom")
    
    d = bond_length if bond_length is not None else mol_def.get("default_bond")
    
    mol = gto.Mole()
    mol.atom = mol_def["atoms"].format(d=d) if d else mol_def["atoms"]
    mol.basis = mol_def["basis"]
    mol.charge = mol_def.get("charge", 0)
    mol.spin = mol_def.get("spin", 0)
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)

    # Get integrals directly from mean-field calculation
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

    # Transform 2-electron integrals to MO basis
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    ecore = mf.energy_nuc()
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    # Construct full FCI Hamiltonian matrix
    fci_solver = fci.FCI(mf)

    # Get exact FCI energy for comparison
    fci_energy = fci_solver.kernel()[0]

    # Build Hamiltonian matrix in full CI space
    _, hamiltonian_matrix = fci_solver.pspace(h1e, h2e, norb, nelec, np=99)
    hamiltonian_matrix = hamiltonian_matrix + np.diag([ecore] * hamiltonian_matrix.shape[0])

    # Pad to next power of 2 if needed (required for qubit representation)
    dim = hamiltonian_matrix.shape[0]
    next_pow2 = 1 << (dim - 1).bit_length()  # Next power of 2
    if dim != next_pow2:
        padded = np.zeros((next_pow2, next_pow2), dtype=hamiltonian_matrix.dtype)
        padded[:dim, :dim] = hamiltonian_matrix
        # Fill diagonal padding with large energy to keep states inaccessible
        for i in range(dim, next_pow2):
            padded[i, i] = 1e6
        hamiltonian_matrix = padded

    num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
    
    mol_info = {
        "name": mol_name,
        "bond_length": d,
        "basis": mol_def["basis"],
        "num_qubits": num_qubits
    }
    
    return hamiltonian_matrix, fci_energy, mf.energy_tot(), mol_info
