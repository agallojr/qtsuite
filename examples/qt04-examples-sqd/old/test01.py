"""
test01 - Sample-based Quantum Diagonalization (SQD) 
"Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer"
arXiv:2405.05068

The input data format (e.g. "n2_fci.txt") is a plain‑text FCIDUMP integral dump (the standard
Molpro/FCI “FCIDUMP” format). FCIDUMP contains:
A header like &FCI NORB=..., NELEC=..., MS2=..., ORBSYM=..., ISYM=... /
Followed by lines of integrals:
Two‑electron: value i j k l
One‑electron: value i j 0 0
Nuclear repulsion: value 0 0 0 0
This is a common interchange format in quantum chemistry for one‑ and two‑electron integrals,
which PySCF can read/write via pyscf.tools.fcidump.

"""

#pylint: disable=invalid-name, protected-access, consider-using-with
#pylint: disable=redefined-outer-name, too-many-arguments, too-many-positional-arguments
#pylint: disable=too-many-locals

import subprocess
import tomllib
import argparse
from pathlib import Path
import sys
from typing import Optional

# pyscf - python module for quantum chemistry - https://github.com/pyscf/pyscf
from pyscf import ao2mo, tools, scf, gto, fci

import matplotlib.pyplot as plt
import numpy as np

# IBM Qiskit addon library for SQD
from qiskit_addon_sqd.counts import generate_counts_uniform
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion

from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager, logger


def make_rhf(atom: str, basis: str, symmetry: bool, spin: int, charge: int,
    num_alpha: int, num_beta: int, dump_integrals: Optional[str] = None):
    """
    Take the description of an atom & basis set, run a restricted Hartree–Fock (RHF)
    calculation, returning the converged mean‑field object.
    """
    # Define the N2 molecule
    print(f"{atom} basis {basis} spin {spin} charge {charge} symmetry {symmetry}")
    mol = gto.M(atom=atom, basis=basis, symmetry=symmetry, spin=spin, charge=charge)
    # Run a Restricted Hartree-Fock (RHF) calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # Compute ground state energy using FCI
    total_electrons = num_alpha + num_beta

    print(f"Running FCI with {total_electrons} electrons (molecule has {mol.nelectron})")

    # Always use full FCI for now to avoid active space complications
    cisolver = fci.FCI(mol, mf.mo_coeff)
    e_fci, _ = cisolver.kernel()

    print(f"SCF energy: {mf.e_tot:.10f} Ha")
    print(f"FCI ground state energy: {e_fci:.10f} Ha")
    print(f"Correlation energy: {e_fci - mf.e_tot:.10f} Ha")
    print(f"Active electrons: {total_electrons} / {mol.nelectron} total")

    if dump_integrals is not None:
        tools.fcidump.from_scf(mf, dump_integrals)
        print(f"FCIDUMP written to {dump_integrals}")

    # Store the ground state energy in the mf object for later use
    return  mf, e_fci


if __name__ == '__main__':

    # *******************************************
    # parse CLI args for workflow input file (required)

    parser = argparse.ArgumentParser(description="Run SQD cases from a TOML definition")
    parser.add_argument("workflow_toml", metavar="WORKFLOW_TOML",
                        help="Path to TOML file")
    args = parser.parse_args()

    wf_toml_path = Path(args.workflow_toml)
    if not wf_toml_path.is_file():
        print(f"Error: {wf_toml_path} not found.")
        sys.exit(1)

    # load the TOML file of test case inputs
    try:
        with open(wf_toml_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)
    globalArgs = data["global"]

    # *******************************************
    # make an lwfm workflow to bundle all the cases
    wf = Workflow("SQD variations", "Fun with SQD", {})
    wf = lwfManager.putWorkflow(wf)
    if wf is None:
        print("Error: Failed to register workflow.")
        sys.exit(1)
    logger.setContext(wf)
    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())
    keepSaveDir = globalArgs["savedir"]
    # warm up sandboxes we use - here the latest Qiskit libs
    # lwfManager.updateSite("ibm-quantum-venv")

    # *******************************************
    # for each case in the workflow toml

    for caseId, caseArgs in ((k, v) for k, v in data.items() if k != "global"):
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId  # dir for this case
        caseArgs.update(globalArgs)
        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)
        dump_fci_path = caseOutDir / "FCIDUMP"

        # make the molecule and run SCF, get the reference ground state energy
        molecule_scf, g_energy = make_rhf(caseArgs['atom'], caseArgs['basis'],
            caseArgs['symmetry'], caseArgs['spin'], caseArgs['charge'],
            caseArgs['num_alpha'], caseArgs['num_beta'],
            dump_integrals=str(dump_fci_path) if caseArgs['dump_integrals'] else None)
