"""
Chemistry helper functions and molecule definitions.

Based on PennyLane's "Top 20 molecules for quantum computing":
https://pennylane.ai/blog/2024/01/top-20-molecules-for-quantum-computing
"""

import numpy as np
from pyscf import gto, scf, fci, ao2mo
from qiskit.quantum_info import SparsePauliOp, Operator

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


def build_molecular_hamiltonian_fci(molecule: str | dict = "H2", bond_length: float = None, 
                                    return_matrix: bool = False):
    """
    Build molecular Hamiltonian using PySCF FCI (Full Configuration Interaction).
    
    Args:
        molecule: Either a molecule name (str) from MOLECULES, or a dict with keys:
                  atoms, basis, charge, spin, default_bond (optional), description (optional)
        bond_length: Bond length in Angstroms (None = use default)
        return_matrix: If True, return numpy matrix; if False, return SparsePauliOp (default: False)
    
    Returns:
        hamiltonian: Hamiltonian as SparsePauliOp (default) or numpy matrix (if return_matrix=True)
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
    
    # Return matrix or SparsePauliOp based on parameter
    if return_matrix:
        return hamiltonian_matrix, fci_energy, mf.energy_tot(), mol_info
    else:
        # Convert matrix to SparsePauliOp for VQE
        hamil_qop = SparsePauliOp.from_operator(Operator(hamiltonian_matrix))
        return hamil_qop, fci_energy, mf.energy_tot(), mol_info


def build_wannier_hamiltonian(jid: str, k_point: list = None, return_matrix: bool = False):
    """
    Build Wannier tight-binding Hamiltonian from JARVIS database.
    
    Args:
        jid: JARVIS material ID (e.g., "JVASP-816" for Al, "JVASP-51338" for FePbW2)
        k_point: k-point coordinates [kx, ky, kz] (default: [0.0, 0.0, 0.0] - Gamma point)
        return_matrix: If True, return numpy matrix; if False, return SparsePauliOp (default: False)
    
    Returns:
        hamiltonian: Hamiltonian as SparsePauliOp (default) or numpy matrix (if return_matrix=True)
        exact_ground_energy: Exact ground state energy from diagonalization (eV)
        wannier_info: dict with material details
    
    Raises:
        ImportError: If jarvis-tools is not installed
        ValueError: If material not found in JARVIS database
    """
    try:
        from jarvis.db.figshare import get_wann_electron, get_hk_tb
    except ImportError as exc:
        raise ImportError(
            "jarvis-tools is required for Wannier Hamiltonians. "
            "Install with: pip install jarvis-tools"
        ) from exc
    
    if k_point is None:
        k_point = [0.0, 0.0, 0.0]
    
    # Get Wannier data from JARVIS
    try:
        w, ef, atoms = get_wann_electron(jid=jid)
    except Exception as e:
        raise ValueError(f"Failed to retrieve material {jid} from JARVIS: {e}") from e
    
    # Compute tight-binding Hamiltonian at k-point
    hk = get_hk_tb(w=w, k=k_point)
    
    # Pad to power of 2 if needed
    n = hk.shape[0]
    is_power_of_2 = (n & (n-1)) == 0 and n != 0
    
    if not is_power_of_2:
        next_power_of_2 = 1 << (n - 1).bit_length()
        padded_hk = np.zeros((next_power_of_2, next_power_of_2), dtype=hk.dtype)
        padded_hk[:n, :n] = hk
        # Add large energy to padded states to keep them inaccessible
        for i in range(n, next_power_of_2):
            padded_hk[i, i] = 1e6
        hk = padded_hk
    
    # Get exact ground state energy
    eigenvalues = np.linalg.eigvalsh(hk)
    exact_ground_energy = float(eigenvalues[0])
    
    num_qubits = int(np.log2(hk.shape[0]))
    
    wannier_info = {
        "jid": jid,
        "k_point": k_point,
        "fermi_energy_ev": float(ef),
        "num_atoms": len(atoms.cart_coords),
        "formula": atoms.composition.reduced_formula,
        "num_qubits": num_qubits,
        "original_dimension": n,
        "padded": not is_power_of_2
    }
    
    # Return matrix or SparsePauliOp based on parameter
    if return_matrix:
        return hk, exact_ground_energy, wannier_info
    else:
        # Convert matrix to SparsePauliOp
        hamil_qop = SparsePauliOp.from_operator(Operator(hk))
        return hamil_qop, exact_ground_energy, wannier_info


def build_siam_hamiltonian(num_orbs: int, hopping: float = 1.0, onsite: float = 5.0,
                           hybridization: float = 1.0, chemical_potential: float = -2.5,
                           momentum_basis: bool = True):
    """
    Build Single Impurity Anderson Model (SIAM) Hamiltonian.
    
    The SIAM describes a magnetic impurity coupled to a bath of conduction electrons.
    Fundamental model for Kondo physics, quantum dots, and heavy fermion systems.
    
    Args:
        num_orbs: Number of spatial orbitals
        hopping: Hopping parameter t between bath sites (default: 1.0)
        onsite: Onsite Coulomb interaction U on impurity (default: 5.0)
        hybridization: Impurity-bath coupling V (default: 1.0)
        chemical_potential: Chemical potential μ (default: -2.5)
        momentum_basis: If True, return in momentum basis; if False, position basis
    
    Returns:
        h1e: One-electron integrals (num_orbs × num_orbs)
        h2e: Two-electron integrals (num_orbs^4 array)
        exact_energy: Exact FCI ground state energy
        siam_info: dict with model parameters
    """
    # Place impurity on first site (position basis)
    impurity_orb = 0
    
    # One-body matrix elements in position basis
    h1e = np.zeros((num_orbs, num_orbs))
    # Hopping between bath sites
    np.fill_diagonal(h1e[:, 1:], -hopping)
    np.fill_diagonal(h1e[1:, :], -hopping)
    # Hybridization between impurity and first bath site
    h1e[impurity_orb, impurity_orb + 1] = -hybridization
    h1e[impurity_orb + 1, impurity_orb] = -hybridization
    # Chemical potential on impurity
    h1e[impurity_orb, impurity_orb] = chemical_potential
    
    # Two-body matrix elements in position basis
    h2e = np.zeros((num_orbs, num_orbs, num_orbs, num_orbs))
    # Onsite interaction on impurity only
    h2e[impurity_orb, impurity_orb, impurity_orb, impurity_orb] = onsite
    
    # Transform to momentum basis if requested
    if momentum_basis:
        h1e, h2e = _siam_to_momentum_basis(h1e, h2e, num_orbs)
    
    # Compute exact energy by direct diagonalization
    # For SIAM, we can build the full Hamiltonian matrix and diagonalize
    # This is more reliable than trying to use PySCF with a dummy molecule
    from pyscf import fci as pyscf_fci
    
    # Build full Hamiltonian matrix in Fock space
    # Assume half-filling for electron count
    nelec = (num_orbs // 2, num_orbs // 2)
    
    # Use FCI to build Hamiltonian in the appropriate Fock space
    # Create FCI solver without molecular object
    norb = num_orbs
    na, nb = nelec
    
    # Build Hamiltonian matrix directly
    h1e_a = h1e
    h1e_b = h1e
    h2e_aa = h2e
    h2e_ab = h2e
    h2e_bb = h2e
    
    # Get FCI Hamiltonian dimension and build matrix
    from pyscf.fci import cistring
    na_states = cistring.num_strings(norb, na)
    nb_states = cistring.num_strings(norb, nb)
    dim = na_states * nb_states
    
    # Use direct_spin1 to build and diagonalize
    from pyscf.fci import direct_spin1
    h_fci = direct_spin1.pspace(h1e_a, h2e_aa, norb, nelec, np=dim)[1]
    
    # Diagonalize to get exact ground state energy
    eigenvalues = np.linalg.eigvalsh(h_fci)
    exact_energy = float(eigenvalues[0])
    
    siam_info = {
        "model": "SIAM",
        "num_orbs": num_orbs,
        "num_qubits": 2 * num_orbs,
        "hopping": hopping,
        "onsite": onsite,
        "hybridization": hybridization,
        "chemical_potential": chemical_potential,
        "basis": "momentum" if momentum_basis else "position"
    }
    
    return h1e, h2e, exact_energy, siam_info


def _siam_to_momentum_basis(h1e: np.ndarray, h2e: np.ndarray, num_orbs: int):
    """Transform SIAM Hamiltonian from position to momentum basis."""
    n_bath = num_orbs - 1
    
    # Diagonalize bath hopping matrix
    hopping_matrix = np.zeros((n_bath, n_bath))
    np.fill_diagonal(hopping_matrix[:, 1:], -1)
    np.fill_diagonal(hopping_matrix[1:, :], -1)
    _, vecs = np.linalg.eigh(hopping_matrix)
    
    # Canonicalize eigenvector signs for reproducibility
    signs = np.sign(vecs[0, :])
    zero_mask = np.isclose(signs, 0.0)
    if np.any(zero_mask):
        js = np.where(zero_mask)[0]
        i_max = np.argmax(np.abs(vecs[:, js]), axis=0)
        signs[js] = np.sign(vecs[i_max, js])
    signs[signs == 0] = 1.0
    vecs = vecs * signs
    
    # Build full orbital rotation (include impurity)
    orbital_rotation = np.zeros((num_orbs, num_orbs))
    orbital_rotation[0, 0] = 1  # Impurity on first site
    orbital_rotation[1:, 1:] = vecs
    
    # Move impurity to center
    new_index = n_bath // 2
    perm = np.r_[1:(new_index + 1), 0, (new_index + 1):num_orbs]
    orbital_rotation = orbital_rotation[:, perm]
    
    # Rotate Hamiltonian
    h1e_momentum = np.einsum('ab,Aa,Bb->AB', h1e, orbital_rotation, 
                             orbital_rotation.conj(), optimize='greedy')
    h2e_momentum = np.einsum('abcd,Aa,Bb,Cc,Dd->ABCD', h2e, 
                             orbital_rotation, orbital_rotation.conj(),
                             orbital_rotation, orbital_rotation.conj(), 
                             optimize='greedy')
    
    return h1e_momentum, h2e_momentum
