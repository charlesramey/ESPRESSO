import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from utils import clean_ts, ig_cal

def separate_greedy_ig(ts, num_segms, cc, pdist_ratio=0.01):
    """
    Greedy segmentation based on Information Gain.

    Args:
        ts: numpy array of raw time series (n_dims, n_points).
        num_segms: int, number of segments (K).
        cc: numpy array (n_dims, n_points - m + 1), Semantic Density Matrix (wcac).
        pdist_ratio: float, minimum peak distance ratio.

    Returns:
        best_tt: list of cut points (indices).
        best_ig: float, Information Gain of the best segmentation.
    """
    num_ts, data_length = ts.shape

    # Calculate Integral Time Series (CumSum)
    # MATLAB: Integ_TS = Clean_TS(ts,1);
    integ_ts = clean_ts(ts, double_mode=1)

    # Calculate min peak distance
    p_dist = int(np.floor(data_length * pdist_ratio))

    # Smooth CC (Gaussian, window 5)
    # MATLAB: CC = -1 * smoothdata(CC,'gaussian',5);
    # Invert CC because we want peaks in the "valley" of density?
    # Or maybe "crosscount" is density of crossings.
    # High crossing = high stability?
    # MATLAB code inverts it: -1 * ...
    # And then findpeaks.
    # So it looks for local MINIMA of the original CC (since -CC has peaks where CC has valleys).
    # Wait, 'calculateSemanticDensityMatrix' accumulates counts.
    # High count means many arcs cross this point -> stable regime?
    # Or transition?
    # If it's density of arcs, a cut point should interrupt fewer arcs?
    # Or cut point is where density is low?
    # If we look for peaks in -CC, we look for valleys in CC (low density).
    # This makes sense: cut where few arcs cross (transition between patterns).

    cc_smooth = -1 * pd.DataFrame(cc.T).rolling(window=5, win_type='gaussian', center=True).mean(std=1).fillna(0).values.T

    # The rolling mean might produce NaNs at edges if not handled, or partial windows.
    # fillna(0) is a quick fix. MATLAB smoothdata handles edges.

    best_tt = []
    best_ig = 0.0

    # Iterate over each dimension (each TS) to find candidate cut points
    for d in range(num_ts):
        row = cc_smooth[d, :]

        # Find peaks
        # MATLAB: findpeaks(CC(d,:),'SortStr','descend', 'minpeakdistance',pdist);
        peaks, properties = find_peaks(row, distance=p_dist)
        prominences = properties.get('prominences', row[peaks]) # find_peaks doesn't return prominences by default unless specified?
        # Need to call peak_prominences or use `prominence` parameter in find_peaks to filter?
        # MATLAB: [pks, locs, width, proms] = findpeaks(...)

        # Scipy find_peaks returns dict 'properties' if we ask for it.
        # But for sorting by prominence we need to compute it.
        # Let's simple sort by peak height (value) as a proxy if prominence is hard?
        # But MATLAB sorts by 'proms' (prominence) if requested?
        # Actually MATLAB `findpeaks(..., 'SortStr', 'descend')` sorts by Peak Value (Height).
        # Wait, the code says: `[p, idx] = sort(proms,'descend');`
        # So it manually sorts by prominence after getting them.

        # So I need peak prominences.
        proms, _, _ = peak_prominences(row, peaks)

        # Sort by prominence
        # idx = np.argsort(proms)[::-1]
        # locs = peaks[idx]

        # Combine locs and proms
        peak_data = list(zip(peaks, proms))
        peak_data.sort(key=lambda x: x[1], reverse=True) # Sort by prominence descending

        locs = [x[0] for x in peak_data]

        tt = []
        max_ig = np.zeros(len(locs))

        if len(locs) > 0:
            remain_locs = locs.copy()

            # Greedy selection
            # MATLAB: for i = 1 : length(locs)
            for i in range(len(locs)):
                temp_tt = None
                current_max_ig = -np.inf

                # Try adding each remaining candidate
                for loc in remain_locs:
                    # Construct candidate set
                    # c = [TT, loc, dataLength]
                    # In Python, we need to pass indices including the end.
                    # IG_Cal expects sorted cut points.
                    # Indices must be 0-based.
                    # dataLength in MATLAB is 1-based index of last point.
                    # In Python, we should probably pass the index 'n' which is out of bounds
                    # but represents the end of the last segment (n-1 inclusive).
                    # My IG_Cal implementation uses: segment_dist = integ_ts[:, current_pos] - ...
                    # So if current_pos is n (size), integ_ts[:, n] must exist?
                    # Clean_TS returns cumsum of shape (dims, n).
                    # Indices 0..n-1.
                    # integ_ts[:, n-1] is the total sum.
                    # If I pass n-1 as the end.
                    # Then segment is ... to n-1.
                    # So I should pass n-1?
                    # But separateGreedyIG passes dataLength.
                    # If dataLength is 100, indices 1..100.
                    # Integ_TS size is 100.
                    # Integ_TS(:, 100) is valid.
                    # So I should pass data_length - 1 (the last valid index).

                    c = tt + [loc, data_length - 1]

                    # Calculate IG
                    # IG_Cal(Integ_TS, c, i)
                    # i is the number of cut points (excluding end).
                    # len(tt) + 1.
                    # Wait, MATLAB loop i corresponds to number of added points?
                    # Yes.

                    ig = ig_cal(integ_ts, c, i) # i is 0-based index of loop, so it's 0, 1...
                    # but in MATLAB i=1 means 1 cut point.
                    # My IG_Cal takes k (num segments - 1).
                    # If i=0 (1 cut point added + end), then k should be 1.
                    # MATLAB: IG_Cal(..., i). i=1.
                    # Python loop i starts at 0. So pass i+1?
                    # MATLAB loop 1..length(locs).
                    # Python loop 0..length(locs)-1.
                    # So pass i+1.

                    if ig > current_max_ig:
                        current_max_ig = ig
                        temp_tt = loc

                if temp_tt is not None:
                    max_ig[i] = current_max_ig
                    tt.append(temp_tt)
                    remain_locs.remove(temp_tt)

            # Select best K
            # if(numSegms-1 > length(maxIG)) t = length(maxIG); else t = numSegms-1;
            # We want K segments, so K-1 cut points.

            target_cuts = num_segms - 1
            if target_cuts > len(max_ig):
                t = len(max_ig)
            else:
                t = target_cuts

            # Check if this TS provided a better IG than previous best (across all TS dimensions)
            # wait, MATLAB code:
            # if(maxIG(t) > bestIG) bestIG = maxIG(t); bestTT = TT(1:t); end
            # It updates global best if this dimension found a better segmentation.

            # max_ig array stores the best IG found for i cuts at index i.
            # But we are interested in exactly 't' cuts?
            # Yes. maxIG(t) (MATLAB) -> max_ig[t-1] (Python).

            if t > 0:
                score = max_ig[t-1]
                if score > best_ig:
                    best_ig = score
                    best_tt = tt[:t]

    return sorted(best_tt), best_ig
