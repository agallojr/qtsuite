"""
Image flip demonstration

Sample execution:
    python src/image_flip.py --size 4 --shots 1024
    python src/image_flip.py --image input/sample.png --shots 2048
    python src/image_flip.py --size 8 --t1 50 --t2 40 --shots 4096
"""

import argparse
import json
import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

from qp4p_circuit import run_circuit, estimate_shots
from qp4p_util import load_or_generate_image, validate_power_of_2

# *****************************************************************************

def build_mirror_circuit(nq: int) -> QuantumCircuit:
    """
    Build a circuit that performs horizontal mirror (flip about Y-axis).
    For an s×s image with n=2*log2(s) qubits, the lower half are column qubits.
    
    Pixel index encoding:
      - Index = row * s + col
      - Lower n/2 qubits encode column (0 to s-1)
      - Upper n/2 qubits encode row (0 to s-1)
    
    To flip column c → column (s-1-c), we invert all column bits using X gates.
    Example for s=4 (2 column qubits):
      col 0 (00) → col 3 (11)
      col 1 (01) → col 2 (10)
      col 2 (10) → col 1 (01)
      col 3 (11) → col 0 (00)
    """
    cq = nq // 2  # Number of qubits encoding columns
    circ = QuantumCircuit(nq)
    
    # Apply X gate to all column qubits to invert column index
    for index in range(cq):
        circ.x(index)
    
    return circ


def make_mirror_circuit(pixels_normalized: np.ndarray) -> QuantumCircuit:
    """
    Create a circuit that amplitude encodes an image and applies horizontal mirror.
    
    Args:
        pixels_normalized: Normalized pixel values (sum of squares = 1)
    
    Returns:
        QuantumCircuit with amplitude encoding, mirror operation, and measurements
    """
    nq = int(np.log2(len(pixels_normalized)))
    circ = QuantumCircuit(nq)
    
    # Amplitude encode the original image
    circ.initialize(pixels_normalized, range(nq))
    circ.barrier()
    
    # Apply mirror: X gates on column qubits invert column index
    # For s×s image: lower log2(s) qubits are column bits
    cq = nq // 2
    for index in range(cq):
        circ.x(index)
    circ.barrier()
    
    # Measure all qubits
    circ.measure_all()
    
    return circ


def counts_to_image(sample_counts: dict, size: int) -> np.ndarray:
    """Reconstruct image from measurement counts."""
    npix = size * size
    pixel_counts = np.zeros(npix)
    
    for bs, c in sample_counts.items():
        # Qiskit bitstrings are big-endian (qubit 0 is rightmost)
        # Direct conversion gives the correct index
        index = int(bs, 2)
        if index < npix:
            pixel_counts[index] = c
    
    # Normalize to 0-255
    max_count = np.max(pixel_counts)
    if max_count > 0:
        image_array = (pixel_counts / max_count) * 255
    else:
        image_array = np.zeros(npix)
    
    return image_array.reshape((size, size)).astype(np.uint8)


def compute_fidelity(expected: np.ndarray, measured: np.ndarray) -> float:
    """
    Compute fidelity between expected and measured images.
    Uses normalized dot product (cosine similarity) of flattened pixel arrays.
    Returns value between 0 (no match) and 1 (perfect match).
    """
    exp_flat = expected.flatten().astype(float)
    meas_flat = measured.flatten().astype(float)
    
    # Normalize both vectors
    exp_norm = np.linalg.norm(exp_flat)
    meas_norm = np.linalg.norm(meas_flat)
    
    if exp_norm == 0 or meas_norm == 0:
        return 0.0
    
    # Cosine similarity
    fid = np.dot(exp_flat, meas_flat) / (exp_norm * meas_norm)
    return float(fid)


# *****************************************************************************
# main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quantum image flip demonstration")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image file")
    parser.add_argument("--size", type=validate_power_of_2, default=4,
                        help="Image size (must be power of 2, default: 4)")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of shots (default: auto-estimated)")
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in microseconds (default: no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in microseconds (default: no noise)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable graphical display of images")
    parser.add_argument("--backend", type=str, default=None,
                        help="Fake backend name (e.g., 'manila', 'jakarta')")
    args = parser.parse_args()
    display = not args.no_display

    # 1. Load and preprocess the image
    pixels = load_or_generate_image(args.image, args.size)

    # Normalize the pixel values to form a valid quantum state vector
    norm_pixels = pixels / np.linalg.norm(pixels)
    num_qubits = int(np.log2(len(norm_pixels)))

    # 2. Create circuit: amplitude encode original, mirror it, measure
    qc = make_mirror_circuit(norm_pixels)

    # 3. Estimate shots if not provided
    num_pixels = args.size * args.size
    if args.shots is None:
        shots = estimate_shots(num_pixels)
    else:
        shots = args.shots

    # 4. Run the circuit
    run_result = run_circuit(qc, shots=shots, t1=args.t1, t2=args.t2, backend=args.backend)
    counts = run_result["counts"]

    # 5. Reconstruct images
    # Original image (from pixel data)
    original_image = (pixels / np.max(pixels) * 255).\
        reshape((args.size, args.size)).astype(np.uint8)
    # Mirrored image (from measurement counts)
    mirrored_image = counts_to_image(counts, args.size)
    
    # Expected mirrored image (classical flip for fidelity comparison)
    expected_mirrored = np.fliplr(original_image)
    
    # Compute fidelity
    fidelity = compute_fidelity(expected_mirrored, mirrored_image)

    # 6. Build results dict
    results = {
        "image": {
            "size": args.size,
            "num_pixels": num_pixels,
            "source": args.image if args.image else "generated_gradient"
        },
        "circuit_stats": {
            "qubits": qc.num_qubits,
            "depth": qc.depth(),
            "gate_counts": dict(qc.count_ops())
        },
        "transpiled_stats": run_result["transpiled_stats"],
        "run": {
            "shots": shots,
            "fidelity": round(fidelity, 4),
            "t1_us": args.t1,
            "t2_us": args.t2
        },
        "backend_info": json.dumps(run_result["backend_info"], separators=(',', ':')) if run_result["backend_info"] else None
    }

    # 7. Print results as JSON
    print(json.dumps(results, indent=2))

    # 8. Display source and result images side by side
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(mirrored_image, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"Quantum Mirror (fidelity={fidelity:.3f})")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()


