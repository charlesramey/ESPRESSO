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
    profile_length = n_points - subsequence_length + 1

    mp = np.zeros((n_dims, profile_length))
    mpi = np.zeros((n_dims, profile_length), dtype=int)

    for i in range(n_dims):
        # stumpy.stump returns (profile_length, 4)
        # columns: MP, MPI, Left MPI, Right MPI
        ts = data[i, :].astype(float)

        # stumpy expects 1D array
        res = stumpy.stump(ts, m=subsequence_length)

        mp[i, :] = res[:, 0]
        mpi[i, :] = res[:, 1]

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
        # MATLAB: if (sum(lastArcSet(:)<dontcare)==0) break;
        # In Python (0-based), valid indices are < n.
        # dontcare is n.
        if np.sum(last_arc_set < dontcare) == 0:
            break

    nnmark = np.zeros(n)

    # Compute Min and Max for normalization
    # MIN=cellfun(@(x) min(x),ArcSet);
    # MAX=cellfun(@(x) max(x),ArcSet);
    # totmin=min(abs(MIN-[1:size(MIN)]'));
    # totmax=max(abs(MAX-[1:size(MAX)]'));

    # In Python, ArcSet[j] corresponds to starting index j.
    # The 'indices' in ArcSet are 0-based.
    # The starting index is j.

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
        # MATLAB: for i=1:list_len
        # small=min(j,list(i)); large=max(j,list(i));

        for idx in chain:
            small = min(j, idx)
            large = max(j, idx)
            length = large - small

            # MATLAB: nnmark(small:large)=nnmark(small:large)+(1-((len-totmin)/(totmax-totmin)));
            # Note: MATLAB ranges are inclusive. Python: small:large+1

            val = 1 - ((length - totmin) * norm_factor)
            nnmark[small:large+1] += val

    return nnmark

def extract_new_arc_set(matrix_profile, mp_index, arc_set, arc_cost, threshold,
                        last_arc_set, last_arc_cost, m, dontcare):
    """
    Helper function to extend arcs.
    """
    n = len(matrix_profile)

    # InitialArcs = lastArcSet;
    # InitialArcs(lastArcCost(:)>threshold)=dontcare;
    initial_arcs = last_arc_set.copy()
    initial_arcs[last_arc_cost > threshold] = dontcare

    # temp = [InitialArcs; dontcare];
    # newArcs = temp(InitialArcs);
    # In Python, we can index with initial_arcs.
    # But initial_arcs contains 'dontcare' which is 'n'.
    # So we need an array of size n+1.

    temp = np.append(initial_arcs, dontcare)
    new_arcs = temp[initial_arcs] # Indexing with array

    temp_arc_cost = np.full(n, threshold + 1.0)
    temp_arc_set = np.full(n, dontcare, dtype=int)

    quarter_m = m / 4.0

    for i in range(n):
        # Check validity
        # MATLAB: if(newArcs(i)~=dontcare && (newArcs(i)>i+m/4 || newArcs(i)<i-m/4))
        target = new_arcs[i]

        if target != dontcare and (target > i + quarter_m or target < i - quarter_m):
            # Update ArcSet
            # ArcSet{i}=[ArcSet{i},newArcs(i)];
            arc_set[i].append(target)

            # Update ArcCost
            # ArcCost{i} = [ArcCost{i},lastArcCost(i)+ lastArcCost(lastArcSet(i))];
            # Note: lastArcSet(i) gives the index that we just jumped FROM?
            # No, lastArcSet is the index of the nearest neighbor of the *previous* step.
            # Wait.
            # lastArcSet holds the index of the neighbor found in the previous iteration.
            # matrix_profile/mp_index are constant (the global MP).
            # But lastArcSet is updated at the end of this function.

            # Let's trace carefully.
            # In MATLAB:
            # newArcs(i) is temp(InitialArcs(i)).
            # InitialArcs(i) is the index of the neighbor of i (from previous step).
            # So newArcs(i) is the neighbor of the neighbor of i. (Chain extension).

            # Cost calculation:
            # lastArcCost(i) + lastArcCost(lastArcSet(i))
            # lastArcCost(i) is the cost of the link i -> lastArcSet(i).
            # lastArcCost(lastArcSet(i)) is the cost of the link lastArcSet(i) -> its neighbor.
            # So we sum the costs.

            cost = last_arc_cost[i] + last_arc_cost[last_arc_set[i]]
            arc_cost[i].append(cost)

            temp_arc_cost[i] = cost
            temp_arc_set[i] = target
        else:
            # tempArcCost(i) = threshold+1; -> Already initialized
            # tempArcSet(i) = dontcare; -> Already initialized
            pass

    return arc_set, arc_cost, temp_arc_set, temp_arc_cost
