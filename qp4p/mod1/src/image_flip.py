"""
Image flip demonstration

Sample execution:
    python mod1/src/image_flip.py --size 4
    python mod1/src/image_flip.py --image input/sample.png --shots 2048
    python mod1/src/image_flip.py --size 8 --t1 50 --t2 40 --backend manila
    python mod1/src/image_flip.py --size 4 --optimization-level 3 --shots 4096
"""

import argparse
import numpy as np
from qiskit import QuantumCircuit
from qp4p_circuit import transpile_circuit, execute_circuit
from qp4p_output import output_json
from qp4p_args import add_standard_quantum_args
from qp4p_util import load_or_generate_image, validate_power_of_2

# *****************************************************************************

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
    add_standard_quantum_args(parser)
    args = parser.parse_args()

    # 1. Load and preprocess the image
    pixels = load_or_generate_image(args.image, args.size)

    # Normalize the pixel values to form a valid quantum state vector
    norm_pixels = pixels / np.linalg.norm(pixels)
    num_qubits = int(np.log2(len(norm_pixels)))

    # 2. Create circuit: amplitude encode original, mirror it, measure
    qc = make_mirror_circuit(norm_pixels)

    # 3. Determine shots (use default 1024 from args, or estimate based on image size)
    num_pixels = args.size * args.size
    # For larger images, increase shots for better reconstruction
    if num_pixels > 16:  # For images larger than 4x4
        args.shots = max(args.shots, num_pixels * 10)

    # 4. Transpile and execute the circuit
    # Transpile
    transpile_result = transpile_circuit(qc, args)
    qc_transpiled, backend, noise_model, _ = transpile_result
    
    # Execute
    result = execute_circuit(qc_transpiled, backend, noise_model, args=args)
    counts = result["counts"]

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

    # 6. Output standardized JSON
    output_json(
        algorithm="image_flip",
        problem={
            "image": {
                "source": args.image,
                "size": args.size,
                "num_pixels": num_pixels
            }
        },
        config_args=args,
        results_data={
            **result,
            "original_image": original_image.tolist(),
            "mirrored_image": mirrored_image.tolist(),
            "expected_mirrored": expected_mirrored.tolist(),
            "fidelity": round(fidelity, 4)
        },
        original_circuit=qc,
        transpile_result=transpile_result
    )
