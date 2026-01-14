import numpy as np
import stumpy

def compute_mp(data, subsequence_length):
    """
    Computes Matrix Profile and MP Index for each dimension of the data.

    Args:
        data: numpy array of shape (n_dims, n_points).
        subsequence_length: int, length of subsequence.

    Returns:
        mp: numpy array of shape (n_dims, profile_length)
        mpi: numpy array of shape (n_dims, profile_length)
    """
    n_dims, n_points = data.shape

    # stumpy.stump returns profile of length n_points - m + 1
    # We need to handle dimensions.

    # Initialize lists (stumpy output shape varies?)
    # stumpy returns (n-m+1, 4).

    # We'll determine shape from first run
    mp_list = []
    mpi_list = []

    for i in range(n_dims):
        ts = data[i, :].astype(float)
        res = stumpy.stump(ts, m=subsequence_length)
        mp_list.append(res[:, 0])
        mpi_list.append(res[:, 1])

    mp = np.array(mp_list)
    mpi = np.array(mpi_list, dtype=int)

    return mp, mpi

def calculate_semantic_density_matrix(matrix_profile, mp_index, chain_length, subsequence_length):
    """
    Extracts k-chains of similar patterns and computes semantic density.

    Args:
        matrix_profile: numpy array (n_points,) (for one dimension)
        mp_index: numpy array (n_points,)
        chain_length: int (K)
        subsequence_length: int (m)

    Returns:
        crosscount: numpy array
    """
    # Ensure inputs are 1D
    matrix_profile = matrix_profile.flatten()
    mp_index = mp_index.flatten().astype(int)

    n = len(matrix_profile)
    dontcare = n # Use n (length) as invalid index (0-based: 0..n-1 are valid)

    # Threshold
    # MATLAB: threshold = 2*max(MatrixProfile);
    threshold = 2 * np.max(matrix_profile)

    # Initialize ArcSet and ArcCost
    # In MATLAB, ArcSet{i} is a list of indices.
    # We can use a list of lists.
    arc_set = [[idx] for idx in mp_index]
    arc_cost = [[cost] for cost in matrix_profile]

    last_arc_set = mp_index.copy()
    last_arc_cost = matrix_profile.copy()

    for k in range(chain_length):
        arc_set, arc_cost, last_arc_set, last_arc_cost = extract_new_arc_set(
            matrix_profile, mp_index, arc_set, arc_cost, threshold,
            last_arc_set, last_arc_cost, subsequence_length, dontcare
        )

        # Check if we should break
        if np.sum(last_arc_set < dontcare) == 0:
            break

    nnmark = np.zeros(n)

    # Compute Min and Max for normalization
    min_vals = np.array([np.min(chain) for chain in arc_set])
    max_vals = np.array([np.max(chain) for chain in arc_set])

    indices = np.arange(n)
    # distance from start index j
    dist_min = np.abs(min_vals - indices)
    dist_max = np.abs(max_vals - indices)

    totmin = np.min(dist_min)
    totmax = np.max(dist_max)

    if totmax == totmin:
        norm_factor = 1.0
    else:
        norm_factor = 1.0 / (totmax - totmin)

    # Accumulate counts
    for j in range(n):
        chain = arc_set[j]
        # Iterate over elements in chain
        for idx in chain:
            small = min(j, idx)
            large = max(j, idx)
            length = large - small

            # Python range: small to large+1 (inclusive of large)
            val = 1 - ((length - totmin) * norm_factor)
            nnmark[small:large+1] += val

    return nnmark

def extract_new_arc_set(matrix_profile, mp_index, arc_set, arc_cost, threshold,
                        last_arc_set, last_arc_cost, m, dontcare):
    """
    Helper function to extend arcs.
    """
    n = len(matrix_profile)

    initial_arcs = last_arc_set.copy()
    initial_arcs[last_arc_cost > threshold] = dontcare

    temp = np.append(initial_arcs, dontcare)
    new_arcs = temp[initial_arcs]

    temp_arc_cost = np.full(n, threshold + 1.0)
    temp_arc_set = np.full(n, dontcare, dtype=int)

    quarter_m = m / 4.0

    for i in range(n):
        target = new_arcs[i]

        if target != dontcare and (target > i + quarter_m or target < i - quarter_m):
            arc_set[i].append(target)

            cost = last_arc_cost[i] + last_arc_cost[last_arc_set[i]]
            arc_cost[i].append(cost)

            temp_arc_cost[i] = cost
            temp_arc_set[i] = target
        else:
            pass

    return arc_set, arc_cost, temp_arc_set, temp_arc_cost
