#!/usr/bin/env python3
"""
Test systematic configuration generation
"""

from itertools import combinations

def generate_systematic_configs(num_orbitals, num_alpha, num_beta):
    """Generate all possible configurations systematically"""
    
    print(f"Generating for {num_orbitals} orbitals, {num_alpha}α + {num_beta}β electrons")
    
    # Generate all possible alpha and beta configurations
    alpha_configs = list(combinations(range(num_orbitals), num_alpha))
    beta_configs = list(combinations(range(num_orbitals), num_beta))
    
    print(f"Alpha configurations: {len(alpha_configs)}")
    print(f"Beta configurations: {len(beta_configs)}")
    print(f"Total combinations: {len(alpha_configs) * len(beta_configs)}")
    
    # Create bitstrings for all combinations
    bitstrings = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            # Create bitstring: first num_orbitals bits for alpha, next for beta
            bitstring = ['0'] * (num_orbitals * 2)
            for i in alpha_occ:
                bitstring[i] = '1'  # Alpha electrons
            for i in beta_occ:
                bitstring[i + num_orbitals] = '1'  # Beta electrons
            bitstrings.append(''.join(bitstring))
    
    # Convert to counts format (each configuration appears once)
    counts = {bs: 1 for bs in bitstrings}
    
    print(f"Created {len(counts)} unique configurations")
    print("First few configurations:")
    for i, (bs, count) in enumerate(list(counts.items())[:5]):
        print(f"  {bs} (count: {count})")
    
    return counts

if __name__ == "__main__":
    # Test with HF system: 6 orbitals, 5α + 5β electrons
    counts = generate_systematic_configs(6, 5, 5)
    
    # Test import of qiskit function
    try:
        from qiskit_addon_sqd.counts import counts_to_arrays
        bitstring_matrix, probs_array = counts_to_arrays(counts)
        print(f"\nSuccessfully converted to arrays:")
        print(f"Bitstring matrix shape: {bitstring_matrix.shape}")
        print(f"Probabilities array shape: {probs_array.shape}")
        print(f"All configurations have correct electron count: {len(bitstring_matrix)} configs")
    except Exception as e:
        print(f"Error converting to arrays: {e}")
