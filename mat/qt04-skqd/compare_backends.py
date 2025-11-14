import numpy as np
from qiskit.primitives import BitArray

print("="*70)
print("COMPARING BOSTON vs KINGSTON QUANTUM DATA")
print("="*70)

# Load both bit arrays
print("\nLoading bit_array_boston.npz...")
boston_data = np.load('bit_array_boston.npz')
boston_array = BitArray(boston_data['samples'], num_bits=int(boston_data['num_bits']))

print("Loading bit_array_kingston.npz...")
kingston_data = np.load('bit_array_kingston.npz')
kingston_array = BitArray(kingston_data['samples'], num_bits=int(kingston_data['num_bits']))

print("\n" + "="*70)
print("BASIC STATISTICS")
print("="*70)

print(f"\nBoston:")
print(f"  Shape: {boston_array.array.shape}")
print(f"  Num bits: {boston_array.num_bits}")
print(f"  Total shots: {boston_array.array.shape[0]}")

print(f"\nKingston:")
print(f"  Shape: {kingston_array.array.shape}")
print(f"  Num bits: {kingston_array.num_bits}")
print(f"  Total shots: {kingston_array.array.shape[0]}")

# Analyze unique bitstrings
boston_unique = np.unique(boston_array.array, axis=0)
kingston_unique = np.unique(kingston_array.array, axis=0)

print("\n" + "="*70)
print("UNIQUENESS (Diversity of quantum states)")
print("="*70)
print(f"\nBoston:")
print(f"  Unique bitstrings: {len(boston_unique)}")
print(f"  Diversity ratio: {len(boston_unique)/boston_array.array.shape[0]:.4f}")

print(f"\nKingston:")
print(f"  Unique bitstrings: {len(kingston_unique)}")
print(f"  Diversity ratio: {len(kingston_unique)/kingston_array.array.shape[0]:.4f}")

# Analyze Hamming weights (number of 1s in each bitstring)
# For fermion systems, this relates to particle number
boston_hamming = np.sum(boston_array.array, axis=1)
kingston_hamming = np.sum(kingston_array.array, axis=1)

print("\n" + "="*70)
print("HAMMING WEIGHT DISTRIBUTION (Particle number conservation)")
print("="*70)
print(f"\nBoston:")
print(f"  Mean Hamming weight: {np.mean(boston_hamming):.2f}")
print(f"  Std dev: {np.std(boston_hamming):.2f}")
print(f"  Min/Max: {np.min(boston_hamming)}/{np.max(boston_hamming)}")

print(f"\nKingston:")
print(f"  Mean Hamming weight: {np.mean(kingston_hamming):.2f}")
print(f"  Std dev: {np.std(kingston_hamming):.2f}")
print(f"  Min/Max: {np.min(kingston_hamming)}/{np.max(kingston_hamming)}")

# For a half-filled system with 24 orbitals, we expect ~24 particles (12 alpha + 12 beta)
expected_particles = 24
print(f"\nExpected particles for half-filled system: {expected_particles}")
print(f"Boston deviation from expected: {abs(np.mean(boston_hamming) - expected_particles):.2f}")
print(f"Kingston deviation from expected: {abs(np.mean(kingston_hamming) - expected_particles):.2f}")

# Analyze entropy/concentration
from collections import Counter

boston_counts = Counter([tuple(row) for row in boston_array.array])
kingston_counts = Counter([tuple(row) for row in kingston_array.array])

# Shannon entropy
def shannon_entropy(counts, total):
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

boston_entropy = shannon_entropy(boston_counts, boston_array.array.shape[0])
kingston_entropy = shannon_entropy(kingston_counts, kingston_array.array.shape[0])

print("\n" + "="*70)
print("SHANNON ENTROPY (Higher = more spread out, less concentrated)")
print("="*70)
print(f"Boston entropy: {boston_entropy:.2f} bits")
print(f"Kingston entropy: {kingston_entropy:.2f} bits")

# Most common bitstrings
print("\n" + "="*70)
print("TOP 5 MOST COMMON BITSTRINGS")
print("="*70)
print("\nBoston:")
for i, (bitstring, count) in enumerate(boston_counts.most_common(5), 1):
    prob = count / boston_array.array.shape[0]
    hamming = sum(bitstring)
    print(f"  {i}. Count: {count:5d} ({prob:.4f}) | Hamming: {hamming:2d}")

print("\nKingston:")
for i, (bitstring, count) in enumerate(kingston_counts.most_common(5), 1):
    prob = count / kingston_array.array.shape[0]
    hamming = sum(bitstring)
    print(f"  {i}. Count: {count:5d} ({prob:.4f}) | Hamming: {hamming:2d}")

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

# Determine which is "better"
better_diversity = "Boston" if len(boston_unique)/boston_array.array.shape[0] > len(kingston_unique)/kingston_array.array.shape[0] else "Kingston"
better_particle_conservation = "Boston" if abs(np.mean(boston_hamming) - expected_particles) < abs(np.mean(kingston_hamming) - expected_particles) else "Kingston"
better_entropy = "Boston" if boston_entropy > kingston_entropy else "Kingston"

print(f"\nBetter state diversity: {better_diversity}")
print(f"Better particle number conservation: {better_particle_conservation}")
print(f"Higher entropy (less noise concentration): {better_entropy}")

print("\n" + "="*70)
