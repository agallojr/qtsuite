#!/usr/bin/env python3
"""
Generate a PySCF molecule that matches the FCIDUMP characteristics:
NORB=16, NELEC=10, MS2=0, all orbitals symmetric
"""

from pyscf import gto, scf, tools, mcscf

def create_molecule_for_fcidump():
    """
    Create a PySCF molecule that will generate an FCIDUMP with:
    - NORB = 16 (16 orbitals)
    - NELEC = 10 (10 electrons) 
    - MS2 = 0 (singlet state)
    - All orbital symmetries = 1 (totally symmetric)
    """

    # N2 molecule - 14 total electrons, but we'll use a minimal active space
    # to get exactly 10 electrons in 16 orbitals
    mol = gto.M(
        atom = 'N 0 0 0; N 0 0 1.1',  # N-N bond length in Angstrom
        basis = 'sto-3g',             # Minimal basis set
        symmetry = True,              # Enable point group symmetry
        spin = 0,                     # Singlet state (MS2 = 0)
        charge = 0                    # Neutral molecule
    )

    print("Molecule info:")
    print(f"  Atoms: {mol.atom}")
    print(f"  Basis: {mol.basis}")
    print(f"  Total electrons: {mol.nelectron}")
    print(f"  Number of AOs: {mol.nao}")
    print(f"  Point group: {mol.groupname}")
    print(f"  Spin: {mol.spin}")

    # Run SCF calculation
    mf = scf.RHF(mol)
    mf.kernel()

    print("\nSCF Results:")
    print(f"  Converged: {mf.converged}")
    print(f"  SCF Energy: {mf.e_tot:.10f}")
    print(f"  Number of MOs: {mf.mo_coeff.shape[1]}")

    # For the active space calculation, we need to select orbitals
    # that will give us exactly 10 electrons in 16 orbitals

    # Option 1: Use all valence orbitals (this should give close to what we want)
    # For N2 with STO-3G, we get 10 AOs per atom = 20 total
    # We can select a subset for our active space

    # Generate FCIDUMP file
    fcidump_file = 'test_fcidump.txt'
    tools.fcidump.from_scf(mf, fcidump_file)

    print(f"\nFCIDUMP written to: {fcidump_file}")

    # Read and display the header to verify
    with open(fcidump_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _, line in enumerate(lines):
            print(f"  {line.strip()}")
            if '&END' in line:
                break

    return mol, mf

def create_active_space_fcidump():
    """
    Alternative approach: Create FCIDUMP with specific active space
    to match exactly NORB=16, NELEC=10
    """

    # Start with a larger system and select active orbitals
    mol = gto.M(
        atom = 'N 0 0 0; N 0 0 1.1',
        basis = 'cc-pVDZ',  # Larger basis to have more orbitals to choose from
        symmetry = True,
        spin = 0,
        charge = 0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    print("\nFull system:")
    print(f"  Total MOs: {mf.mo_coeff.shape[1]}")
    print(f"  Total electrons: {mol.nelectron}")

    # Select active space: 10 electrons in 16 orbitals
    # This typically means 5 occupied + 11 virtual orbitals
    n_core = 2  # Keep 2 core electrons frozen (1s orbitals of each N)
    n_active_occ = 5  # 5 active occupied orbitals
    n_active_virt = 11  # 11 active virtual orbitals

    # Active space indices
    active_indices = list(range(n_core, n_core + n_active_occ + n_active_virt))

    if len(active_indices) == 16:
        print("\nActive space selection:")
        print(f"  Core orbitals: {n_core}")
        print(f"  Active orbitals: {len(active_indices)}")
        print(f"  Active electrons: 10 (excluding {n_core*2} core electrons)")

        # Create FCIDUMP for active space
        fcidump_active = 'active_space_fcidump.txt'

        # Extract active space molecular orbitals
        mo_active = mf.mo_coeff[:, active_indices]

        # Use from_mo to create FCIDUMP with specific active space
        # Note: from_mo determines nelec from the molecular orbitals automatically
        tools.fcidump.from_mo(mol, fcidump_active, mo_active)

        print(f"\nActive space FCIDUMP written to: {fcidump_active}")

        # Verify the header
        with open(fcidump_active, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                print(f"  {line.strip()}")
                if '&END' in line:
                    break

    return mol, mf

def create_casci_fcidump():
    """
    Use CASCI to create FCIDUMP with exactly NORB=16, NELEC=10, MS2=0
    This gives precise control over the active space.
    """

    # Start with N2 molecule
    mol = gto.M(
        atom = 'N 0 0 0; N 0 0 1.1',
        basis = 'sto-3g',  # Start with minimal basis
        symmetry = True,
        spin = 0,
        charge = 0
    )

    # Run SCF first
    mf = scf.RHF(mol)
    mf.kernel()

    print("\nCASCI approach:")
    print(f"  Total MOs: {mf.mo_coeff.shape[1]}")
    print(f"  Total electrons: {mol.nelectron}")

    # Define CASCI with exactly 10 electrons in 16 orbitals
    # For N2, we typically have 14 electrons total
    # So we want 10 active electrons (freeze 4 core electrons)
    ncas = 16  # Number of active orbitals
    nelecas = 10  # Number of active electrons

    # Create CASCI object
    cas = mcscf.CASCI(mf, ncas, nelecas)

    # Run CASCI
    cas.kernel()

    print("\nCASCI Results:")
    print(f"  Active orbitals: {ncas}")
    print(f"  Active electrons: {nelecas}")
    print(f"  CASCI energy: {cas.e_tot:.10f}")

    # Generate FCIDUMP from CASCI
    fcidump_casci = 'casci_fcidump.txt'

    # Use the CASCI object to create FCIDUMP
    # This will have exactly the right number of orbitals and electrons
    tools.fcidump.from_mcscf(cas, fcidump_casci)

    print(f"\nCASCI FCIDUMP written to: {fcidump_casci}")

    # Verify the header
    with open(fcidump_casci, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("FCIDUMP header:")
        for _, line in enumerate(lines):
            print(f"  {line.strip()}")
            if '&END' in line:
                break

    return mol, mf, cas

def create_exact_fcidump():
    """
    Alternative: Create FCIDUMP with exact control using smaller system
    """

    # Try with a system that naturally gives us close to what we want
    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22',  # H4 chain
        basis = 'cc-pVDZ',
        symmetry = False,  # Disable symmetry to get all orbitals as sym=1
        spin = 0,
        charge = 0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    print("\nH4 system:")
    print(f"  Total electrons: {mol.nelectron}")
    print(f"  Total MOs: {mf.mo_coeff.shape[1]}")

    # This should give us closer to the target
    if mf.mo_coeff.shape[1] >= 16:
        # Select exactly 16 orbitals
        selected_mos = mf.mo_coeff[:, :16]

        # Create FCIDUMP
        fcidump_exact = 'exact_fcidump.txt'
        tools.fcidump.from_mo(mol, fcidump_exact, selected_mos)

        print(f"\nExact FCIDUMP written to: {fcidump_exact}")

        # Verify
        with open(fcidump_exact, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("FCIDUMP header:")
            for _, line in enumerate(lines):
                print(f"  {line.strip()}")
                if '&END' in line:
                    break

    return mol, mf

if __name__ == "__main__":
    print("=== Method 1: Direct molecule ===")
    mol1, mf1 = create_molecule_for_fcidump()

    print("\n" + "="*50)
    print("=== Method 2: Active space selection ===")
    mol2, mf2 = create_active_space_fcidump()

    # print("\n" + "="*50)
    # print("=== Method 3: CASCI (Recommended) ===")
    # mol3, mf3, cas3 = create_casci_fcidump()

    print("\n" + "="*50)
    print("=== Method 4: Alternative system ===")
    mol4, mf4 = create_exact_fcidump()
