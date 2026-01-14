import numpy as np
import pandas as pd
from espresso_core import compute_mp, calculate_semantic_density_matrix
from segmentation import separate_greedy_ig

def ESPRESSO(K, data, subsequence, chain_len=None):
    """
    ESPRESSO: Entropy and ShaPe awaRe timE-Series SegmentatiOn.

    Args:
        K: int, number of segments.
        data: numpy array of shape (n_dims, n_points) or (n_points,) for 1D.
        subsequence: int, window size (m).
        chain_len: int, maximum length of chain. Defaults to K if not provided?
                   MATLAB code uses 'chain_length' in `calculateSemanticDensityMatrix` call
                   but input argument is `chain_len`.
                   Assuming it should be passed.

    Returns:
        espTT: list of segment boundaries (indices).
    """
    # Handle data shape
    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_ts, len_ts = data.shape

    # Defaults
    if chain_len is None:
        chain_len = K # Reasonable default? Or maybe 3?
        # MATLAB code doesn't set default, expects input.

    print(f"Running ESPRESSO with K={K}, subsequence={subsequence}, chain_len={chain_len}")
    print(f"Data shape: {num_ts} x {len_ts}")

    # 1. Compute Matrix Profile
    # MATLAB: [MP, MPI] = computMP(data, subsequence);
    mp, mpi = compute_mp(data, subsequence)

    # 2. Calculate Semantic Density Matrix (wcac)
    # MATLAB: wcac(i,:) = calculateSemanticDensityMatrix(...)
    # Note: wcac size in MATLAB: (numTS, lenTS).
    # But calculateSemanticDensityMatrix returns vector of length 'lenTS'.

    mp_len = mp.shape[1]
    wcac = np.zeros((num_ts, mp_len))

    for i in range(num_ts):
        wcac[i, :] = calculate_semantic_density_matrix(mp[i], mpi[i], chain_len, subsequence)

    # 3. Greedy Segmentation with IG
    # MATLAB: [espTT,~] = separateGreedyIG(data, K, wcac, 0.01);
    esp_tt, best_ig = separate_greedy_ig(data, K, wcac, pdist_ratio=0.01)

    return esp_tt

if __name__ == "__main__":
    # Example usage / Simple test
    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    # 3 segments
    s1 = np.sin(2 * np.pi * 0.1 * t[:300])
    s2 = np.sin(2 * np.pi * 0.5 * t[300:700])
    s3 = np.sin(2 * np.pi * 0.1 * t[700:])

    data = np.concatenate([s1, s2, s3])
    data = data.reshape(1, -1) # 1 dimension

    K = 3
    subsequence = 50
    chain_len = 3

    # boundaries = ESPRESSO(K, data, subsequence, chain_len)
    # print("Found boundaries:", boundaries)
    # print("Expected around: [300, 700]")
