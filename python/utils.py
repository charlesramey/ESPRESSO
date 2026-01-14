import numpy as np
from scipy.stats import entropy

def clean_ts(data, double_mode=0):
    """
    Normalizes the time series and handles heterogeneity.

    Args:
        data: numpy array of shape (n_dims, n_points). The raw time series.
        double_mode: 0=Normalize, 1=Normalize and double, 2=No change

    Returns:
        numpy array: The cumulative sum of the processed time series.
    """
    # Ensure data is float
    integ_ts = data.astype(float).copy()
    m, n = integ_ts.shape

    # Process each dimension
    processed_rows = []

    for i in range(m):
        row = integ_ts[i, :].copy()

        min_val = np.min(row)
        if double_mode == 2:
            min_val = 0

        row = row - min_val

        if double_mode != 2:
            sum_val = np.sum(row) / 1000.0
            if sum_val != 0:
                row = row / sum_val
            else:
                # Handle case where sum is 0 (all values were same as min)
                # This results in a row of zeros.
                pass

        processed_rows.append(row)

        if double_mode == 1:
            # Create inverted row
            # maxVal=max(Integ_TS(i,:));
            # Integ_TS(i+m,:)=maxVal-Integ_TS(i,:);
            # Note: We use the already normalized 'row' for inversion?
            # Re-reading MATLAB code:
            #   minVal=min(Integ_TS(i,:)); ... Integ_TS(i,:)=Integ_TS(i,:)-minVal;
            #   sumVal=sum(Integ_TS(i,:))/1000; Integ_TS(i,:)=Integ_TS(i,:)/sumVal;
            #   if double==1
            #     maxVal=max(Integ_TS(i,:));
            #     Integ_TS(i+m,:)=maxVal-Integ_TS(i,:); <-- This uses the MODIFIED Integ_TS(i,:)
            #     sumVal=sum(Integ_TS(i+m,:))/1000;
            #     Integ_TS(i+m,:)=Integ_TS(i+m,:)/sumVal;

            inverted_row = np.max(row) - row
            sum_val_inv = np.sum(inverted_row) / 1000.0
            if sum_val_inv != 0:
                inverted_row = inverted_row / sum_val_inv

            processed_rows.append(inverted_row)

    # If double_mode == 1, we interleaved the rows in the loop?
    # MATLAB: Integ_TS(i+m,:) = ...
    # This means the original rows are 1..m, and new rows are m+1..2m.
    # In my loop I appended them.
    # So if I just stack them, I get row1, row1_inv, row2, row2_inv...
    # But MATLAB produces row1, row2, ..., row1_inv, row2_inv, ...
    # I should align with MATLAB logic to be safe, although order might not matter for entropy sum.
    # But for indexing consistency, let's stick to MATLAB order.

    final_rows = []
    if double_mode == 1:
        # Separate original and inverted
        originals = [processed_rows[i] for i in range(0, len(processed_rows), 2)]
        inverted = [processed_rows[i] for i in range(1, len(processed_rows), 2)]
        final_rows = originals + inverted
    else:
        final_rows = processed_rows

    integ_ts_processed = np.array(final_rows)

    # Return cumulative sum
    # MATLAB: N_Integ_TS=cumsum(Integ_TS,2);
    return np.cumsum(integ_ts_processed, axis=1)

def sh_entropy(x):
    """
    Calculates Shannon Entropy.
    """
    # Filter out zeros (MATLAB: x(x==0)=[])
    # scipy.stats.entropy handles this but checks if probabilities sum to 1.
    # Here input x are counts or values, entropy will normalize them.
    # We should filter 0s to avoid issues if any.
    # Actually scipy.stats.entropy is robust.
    return entropy(x)

def ig_cal(integ_ts, pos_tt, k):
    """
    Calculates Information Gain for a set of cut points.

    Args:
        integ_ts: cumulative sum of time series (n_dims, n_points)
        pos_tt: list/array of cut points (indices). Should include the end index if needed?
                MATLAB: pos_tt1 argument. pos_TT=sort(pos_TT1);
                Wait, MATLAB code uses pos_TT(i) as the end of segment.
                And it iterates i from 1 to k+1.
                This implies pos_tt has k+1 elements?

                In separateGreedyIG:
                c = [TT, remain_locs(j), dataLength];
                ig = IG_Cal(Integ_TS, c, i);
                Here i is the loop variable for number of segments found so far?
                No, i in separateGreedyIG is loop 1..length(locs).
                Wait.
                separateGreedyIG:
                   TT starts empty.
                   i = 1:
                     c = [TT, remain_locs(j), dataLength] -> [loc, len]
                     IG_Cal(Integ_TS, c, i) -> k=1.
                     IG_Cal iterates 1..k+1 -> 1..2.
                     pos_TT has 2 elements.
                     Segment 1: 1 to pos_TT(1) (loc)
                     Segment 2: pos_TT(1) to pos_TT(2) (len)
                     Correct.
    """
    pos_tt = np.sort(np.array(pos_tt, dtype=int))
    # MATLAB uses 1-based indexing.
    # Python 0-based.
    # integ_ts is cumsum.
    # Value at index i is sum(0..i).
    # Segment sum from last_id (exclusive) to pos_TT(i) (inclusive).
    # MATLAB: Integ_TS(j,pos_TT(i))-Integ_TS(j,last_id);
    # In Python:
    # If last_id is index of end of previous segment.
    # Sum is integ_ts[current_end] - integ_ts[prev_end].

    # Initialization
    # MATLAB: last_id = 1; (Since cumsum starts at index 1)
    # But wait, Clean_TS returns cumsum.
    # If we want sum of first segment ending at idx (inclusive), it is integ_ts[idx].
    # But MATLAB subtracts integ_ts[last_id].
    # Initially last_id=1.
    # So first segment sum is Integ_TS(pos) - Integ_TS(1).
    # This implies the first element is excluded?
    # Or Integ_TS includes a 0 at start? No.
    # MATLAB cumsum of [1, 2, 3] is [1, 3, 6].
    # Segment 1..2 sum is 3. Integ_TS(2).
    # MATLAB code: Integ_TS(j,pos_TT(i))-Integ_TS(j,last_id) with last_id=1.
    # So for first segment ending at pos, it subtracts Integ_TS(1).
    # So it effectively ignores the first point of the time series?
    # Or maybe 'last_id' represents the 'start index - 1'?
    # In MATLAB, if start is 1, previous is 0. But indices start at 1.
    # If last_id=1, it subtracts the accumulated value at index 1.
    # So the segment is from index 2 to pos_TT(i).
    # This seems to lose the first point.

    # Let's assume this is intentional or a minor index off-by-one in MATLAB code that I should replicate or fix.
    # Given 'last_id=1' initialization.
    # If I replicate:
    # Python indices: 0..N-1.
    # If I use 0-based indices for pos_tt.
    # First segment: Integ_TS[pos] - Integ_TS[0].
    # This is sum(1..pos). (indices). So value at 0 is excluded.

    # However, for total distribution entropy:
    # TS_dist(i)=Integ_TS(i,Le_TS);
    # This takes the full sum (value at last index).

    # Let's look closer at separateGreedyIG:
    # c = [TT, remain_locs(j), dataLength];
    # dataLength is the length of TS.
    # In MATLAB, indices are 1..dataLength.
    # So c includes the last index.

    # If I want to be mathematically correct:
    # Sum(start..end) = CumSum(end) - CumSum(start-1).
    # If start=0, CumSum(-1) = 0.

    # In MATLAB code:
    # last_id starts at 1.
    # So for the first segment, it subtracts Integ_TS(:, 1).
    # This means the first time point is effectively NOT part of the first segment sum.
    # This is likely a bug or quirk in the MATLAB code.
    # "last_id=1" suggests it treats index 1 as the boundary *before* the first segment starts?
    # But Integ_TS is computed from data.
    # If I want to exactly replicate MATLAB:
    # I should treat indices as 0-based but mimic the subtraction logic.
    # If Python last_id = 0.
    # segment sum = integ_ts[pos] - integ_ts[0].
    # This matches MATLAB behavior (ignoring first point).

    # Let's try to do better.
    # Maybe I should handle the first segment differently.
    # Or maybe I should prepend a 0 column to Integ_TS so that subtraction works naturally.
    # If I pad Integ_TS with 0 at the beginning (column -1).

    # But wait, let's see how Clean_TS works.
    # It just does cumsum.

    # If I want to replicate MATLAB exactly, I must do what it does.
    # But if I want it to be correct...
    # The paper says "Entropy...".
    # Missing one point in a long time series (e.g. 1000s points) is negligible.
    # But for very short segments it matters.

    # I will replicate the MATLAB logic but adjust for 0-based indexing.
    # MATLAB: last_id=1. pos_TT are indices.
    # Python: last_id=0. pos_tt are indices.
    # Subtraction: integ_ts[:, pos] - integ_ts[:, last_id].
    # This is consistent.

    nu_ts, le_ts = integ_ts.shape

    # Calculate global distribution entropy
    # In MATLAB: TS_dist(i)=Integ_TS(i,Le_TS);
    # This is the last column.
    ts_dist = integ_ts[:, -1]
    ig = sh_entropy(ts_dist)

    last_id = 0 # Corresponds to MATLAB's 1

    # Iterate k+1 times.
    # k is passed as argument, but loop is 1:k+1.
    # In MATLAB 'k' is passed as 'i' from separateGreedyIG loop.
    # 'i' in separateGreedyIG is the number of cut points added so far (excluding end).
    # Wait, c = [TT, remain_locs(j), dataLength].
    # If TT is empty, c has 2 elements (loc, end).
    # i=1.
    # IG_Cal loop 1 to 2.
    # Iteration 1: pos_TT(1) (the loc). Segment 1.
    # Iteration 2: pos_TT(2) (the end). Segment 2.

    # So 'k' in IG_Cal seems to be "number of segments - 1"?
    # The argument name is 'k'.
    # separateGreedyIG passes 'i' as 'k'.
    # If i=1 (1 cut point), then we have 2 segments.
    # Loop 1..k+1 -> 1..2. Correct.

    for i in range(k + 1):
        # Current cut point index
        # pos_tt is sorted.
        # pos_tt indices should be integers.
        # In Python, indices are 0..N-1.
        # If passed from findpeaks (0-based), then fine.
        # But MATLAB findpeaks returns 1-based indices.
        # I should ensure inputs to this function are correct 0-based indices.

        current_pos = pos_tt[i]

        # In MATLAB: (pos_TT(i)-last_id)/Le_TS
        # Weight of the segment.
        # Length of segment is current_pos - last_id.

        # Segment sum
        # MATLAB: Integ_TS(j,pos_TT(i))-Integ_TS(j,last_id);
        # Python:
        # If I use last_id=0 initially.
        # And I subtract integ_ts[:, last_id].
        # I am subtracting value at index 0.
        # So I am ignoring index 0 value.

        # I will assume this is desired behavior for now.
        segment_dist = integ_ts[:, current_pos] - integ_ts[:, last_id]

        # Ensure non-negative (floating point errors)
        segment_dist = np.maximum(segment_dist, 0)

        weight = (current_pos - last_id) / le_ts
        ig -= weight * sh_entropy(segment_dist)

        last_id = current_pos

    return ig
