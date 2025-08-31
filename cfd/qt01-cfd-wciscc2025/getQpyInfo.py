from qiskit import qpy
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser(description="Inspect a QPY circuit file and print register sizes")
parser.add_argument("qpy_file", metavar="QPY_FILE", help="Exact path to the .qpy file to inspect")
args = parser.parse_args()

p = Path(args.qpy_file)
if not p.is_file():
    print(f"Error: {p} not found or is not a file.")
    sys.exit(1)

circ = qpy.load(p.open('rb'))[0]
print("total qubits:", circ.num_qubits)
for qreg in circ.qregs:
    print(f"register {qreg.name}: size={qreg.size}")
print("num ancillas attribute (if present):", getattr(circ, "num_ancillas", None))
