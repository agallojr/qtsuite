import numpy as np
from qiskit.primitives import BitArray

print("="*70)
print("COMPARING BOSTON vs KINGSTON QUANTUM DATA")
print("="*70)

# Load both bit arrays
print("\nLoading bit_array_boston.npz...")
boston_data = np.load('bit_array_boston.npz')
boston_samples = boston_data['samples']
boston_bits = int(boston_data['num_bits'])

print("Loading bit_array_kingston.npz...")
kingston_data = np.load('bit_array_kingston.npz')
kingston_samples = kingston_data['samples']
kingston_bits = int(kingston_data['num_bits'])

print("\n" + "="*70)
print("RAW DATA STRUCTURE")
print("="*70)
print(f"\nBoston:")
print(f"  samples shape: {boston_samples.shape}")
print(f"  samples dtype: {boston_samples.dtype}")
print(f"  num_bits: {boston_bits}")
print(f"  First sample (raw): {boston_samples[0]}")

print(f"\nKingston:")
print(f"  samples shape: {kingston_samples.shape}")
print(f"  samples dtype: {kingston_samples.dtype}")
print(f"  num_bits: {kingston_bits}")
print(f"  First sample (raw): {kingston_samples[0]}")

# Reconstruct BitArray properly
boston_array = BitArray(boston_samples, num_bits=boston_bits)
kingston_array = BitArray(kingston_samples, num_bits=kingston_bits)

# Get unpacked bits
boston_unpacked = np.unpackbits(boston_array.array, axis=-1)[..., -boston_bits:].astype(bool)
kingston_unpacked = np.unpackbits(kingston_array.array, axis=-1)[..., -kingston_bits:].astype(bool)

print("\n" + "="*70)
print("UNPACKED BITSTRINGS")
print("="*70)
print(f"\nBoston unpacked shape: {boston_unpacked.shape}")
print(f"Kingston unpacked shape: {kingston_unpacked.shape}")

# Analyze Hamming weights (particle numbers) with unpacked data
boston_hamming = np.sum(boston_unpacked, axis=1)
kingston_hamming = np.sum(kingston_unpacked, axis=1)

print("\n" + "="*70)
print("PARTICLE NUMBER DISTRIBUTION (Hamming weights)")
print("="*70)
print(f"\nBoston:")
print(f"  Mean particle number: {np.mean(boston_hamming):.2f}")
print(f"  Std dev: {np.std(boston_hamming):.2f}")
print(f"  Min/Max: {np.min(boston_hamming)}/{np.max(boston_hamming)}")

print(f"\nKingston:")
print(f"  Mean particle number: {np.mean(kingston_hamming):.2f}")
print(f"  Std dev: {np.std(kingston_hamming):.2f}")
print(f"  Min/Max: {np.min(kingston_hamming)}/{np.max(kingston_hamming)}")

# For a half-filled system with 24 orbitals (48 bits = 24 alpha + 24 beta), we expect 24 particles
expected_particles = 24
print(f"\nExpected particles for half-filled system: {expected_particles}")
print(f"Boston deviation from expected: {abs(np.mean(boston_hamming) - expected_particles):.2f}")
print(f"Kingston deviation from expected: {abs(np.mean(kingston_hamming) - expected_particles):.2f}")

# Analyze uniqueness
boston_unique = np.unique(boston_unpacked, axis=0)
kingston_unique = np.unique(kingston_unpacked, axis=0)

print("\n" + "="*70)
print("STATE DIVERSITY")
print("="*70)
print(f"\nBoston:")
print(f"  Unique bitstrings: {len(boston_unique)}")
print(f"  Diversity ratio: {len(boston_unique)/len(boston_unpacked):.4f}")

print(f"\nKingston:")
print(f"  Unique bitstrings: {len(kingston_unique)}")
print(f"  Diversity ratio: {len(kingston_unique)/len(kingston_unpacked):.4f}")

# Shannon entropy
from collections import Counter

boston_tuples = [tuple(row) for row in boston_unpacked]
kingston_tuples = [tuple(row) for row in kingston_unpacked]

boston_counts = Counter(boston_tuples)
kingston_counts = Counter(kingston_tuples)

def shannon_entropy(counts, total):
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

boston_entropy = shannon_entropy(boston_counts, len(boston_unpacked))
kingston_entropy = shannon_entropy(kingston_counts, len(kingston_unpacked))

print("\n" + "="*70)
print("SHANNON ENTROPY (Higher = more uniform distribution)")
print("="*70)
print(f"Boston: {boston_entropy:.3f} bits")
print(f"Kingston: {kingston_entropy:.3f} bits")
print(f"Maximum possible entropy: {np.log2(len(boston_unpacked)):.3f} bits")

# Most concentrated states
print("\n" + "="*70)
print("TOP 5 MOST COMMON STATES")
print("="*70)
print("\nBoston (most concentrated):")
for i, (bitstring, count) in enumerate(boston_counts.most_common(5), 1):
    prob = count / len(boston_unpacked)
    particles = sum(bitstring)
    print(f"  {i}. Probability: {prob:.4f} ({count:4d} shots) | Particles: {particles:2d}")

print("\nKingston (most concentrated):")
for i, (bitstring, count) in enumerate(kingston_counts.most_common(5), 1):
    prob = count / len(kingston_unpacked)
    particles = sum(bitstring)
    print(f"  {i}. Probability: {prob:.4f} ({count:4d} shots) | Particles: {particles:2d}")

# Concentration metric (higher = more concentrated in fewer states)
boston_concentration = sum([(count/len(boston_unpacked))**2 for count in boston_counts.values()])
kingston_concentration = sum([(count/len(kingston_unpacked))**2 for count in kingston_counts.values()])

print("\n" + "="*70)
print("CONCENTRATION METRIC (Lower = more uniform, better exploration)")
print("="*70)
print(f"Boston: {boston_concentration:.6f}")
print(f"Kingston: {kingston_concentration:.6f}")

print("\n" + "="*70)
print("SUMMARY - WHICH IS BETTER?")
print("="*70)

better_diversity = "Boston" if len(boston_unique) > len(kingston_unique) else "Kingston"
better_particle = "Boston" if abs(np.mean(boston_hamming) - expected_particles) < abs(np.mean(kingston_hamming) - expected_particles) else "Kingston"
better_particle_std = "Boston" if np.std(boston_hamming) < np.std(kingston_hamming) else "Kingston"
better_entropy = "Boston" if boston_entropy > kingston_entropy else "Kingston"
better_uniformity = "Boston" if boston_concentration < kingston_concentration else "Kingston"

print(f"\n✓ More unique states: {better_diversity}")
print(f"✓ Better particle number conservation (closer to 24): {better_particle}")
print(f"✓ Tighter particle number spread (lower std): {better_particle_std}")
print(f"✓ Higher entropy (better state coverage): {better_entropy}")
print(f"✓ More uniform distribution (less concentrated): {better_uniformity}")

# Overall assessment
boston_score = sum([better_diversity == "Boston", better_particle == "Boston", 
                    better_particle_std == "Boston", better_entropy == "Boston",
                    better_uniformity == "Boston"])
kingston_score = 5 - boston_score

print("\n" + "="*70)
print(f"Overall: Boston {boston_score}/5, Kingston {kingston_score}/5")
if boston_score > kingston_score:
    print("★ Boston data appears higher quality")
elif kingston_score > boston_score:
    print("★ Kingston data appears higher quality")
else:
    print("★ Both backends produce similar quality data")
print("="*70)
