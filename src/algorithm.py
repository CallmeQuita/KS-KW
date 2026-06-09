import numpy as np
from scipy.stats import chi2
import scikit_posthocs as sp

def perform_dunn_test(df, group_col, val_col):
    dunn_results = sp.posthoc_dunn(df, val_col=val_col, group_col=group_col, p_adjust='bonferroni')
    return dunn_results

def ks_1samp_manual(data, theoretical_cdf_func):
    n = len(data)
    data_sorted = np.sort(data)
    ecdf_top = np.arange(1, n + 1) / n
    ecdf_bottom = np.arange(0, n) / n
    theoretical_cdf = theoretical_cdf_func(data_sorted)
    d_stat = np.max([
        np.max(np.abs(ecdf_top - theoretical_cdf)),
        np.max(np.abs(ecdf_bottom - theoretical_cdf))
    ])
    d_crit = 1.358 / np.sqrt(n)
    reject_null = d_stat > d_crit
    return d_stat, d_crit, reject_null

def ks_2samp_manual(group_a, group_b):
    n1, n2 = len(group_a), len(group_b)
    data_all = np.sort(np.concatenate([group_a, group_b]))
    cdf1 = np.searchsorted(np.sort(group_a), data_all, side='right') / n1
    cdf2 = np.searchsorted(np.sort(group_b), data_all, side='right') / n2
    d_stat = np.max(np.abs(cdf1 - cdf2))
    d_crit = 1.358 * np.sqrt((n1 + n2) / (n1 * n2))
    reject_null = d_stat > d_crit
    return d_stat, d_crit, reject_null

def ks_2d_2samp_manual(data1, data2):
    n1, n2 = len(data1), len(data2)
    pooled_points = np.vstack((data1, data2))
    
    max_d_stat = 0
    best_x, best_y = 0, 0
    
    for point in pooled_points:
        x, y = point
        q1_diff = abs((np.sum((data1[:, 0] <= x) & (data1[:, 1] <= y)) / n1) -
                      (np.sum((data2[:, 0] <= x) & (data2[:, 1] <= y)) / n2))
        q2_diff = abs((np.sum((data1[:, 0] <= x) & (data1[:, 1] > y)) / n1) -
                      (np.sum((data2[:, 0] <= x) & (data2[:, 1] > y)) / n2))
        q3_diff = abs((np.sum((data1[:, 0] > x) & (data1[:, 1] <= y)) / n1) -
                      (np.sum((data2[:, 0] > x) & (data2[:, 1] <= y)) / n2))
        q4_diff = abs((np.sum((data1[:, 0] > x) & (data1[:, 1] > y)) / n1) -
                      (np.sum((data2[:, 0] > x) & (data2[:, 1] > y)) / n2))
        
        local_max = max(q1_diff, q2_diff, q3_diff, q4_diff)
        if local_max > max_d_stat:
            max_d_stat = local_max
            best_x, best_y = x, y
            
    d_crit = 1.628 * np.sqrt((n1 + n2) / (n1 * n2))
    reject_null = max_d_stat > d_crit
    return max_d_stat, d_crit, reject_null, best_x, best_y

def get_ranks(combined_data):
    N = len(combined_data)
    sorted_indices = np.argsort(combined_data)
    sorted_data = combined_data[sorted_indices]
    
    ranks = np.empty(N)
    i = 0
    ties_count = []
    
    while i < N:
        j = i
        while j < N and sorted_data[j] == sorted_data[i]:
            j += 1
        
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        
        if j - i > 1:
            ties_count.append(j - i)
            
        i = j
        
    original_ranks = np.empty(N)
    original_ranks[sorted_indices] = ranks
    
    return original_ranks, ties_count

def kruskal_wallis_custom(*groups, verbose=False):
    k = len(groups)
    n_i = [len(g) for g in groups]
    N = sum(n_i)
    
    combined_data = np.concatenate(groups)
    ranks, ties_count = get_ranks(combined_data)
    
    rank_sums = []
    start_idx = 0
    for n in n_i:
        group_rank_sum = np.sum(ranks[start_idx : start_idx + n])
        rank_sums.append(group_rank_sum)
        start_idx += n
        
    sum_term = sum((R**2) / n for R, n in zip(rank_sums, n_i))
    H_unadjusted = (12.0 / (N * (N + 1))) * sum_term - 3 * (N + 1)
    
    correction_factor = 1.0
    if ties_count:
        correction_factor = 1.0 - sum(t**3 - t for t in ties_count) / float(N**3 - N)
    
    if correction_factor == 0:
        H = 0.0
        p_value = 1.0
    else:
        H = H_unadjusted / correction_factor
        df = k - 1
        p_value = chi2.sf(H, df) 
    
    epsilon_sq = max(0.0, (H - k + 1) / (N - k)) if N > k else 0.0
    
    details = {}
    if verbose:
        details = {
            "N": N,
            "n_i": n_i,
            "k": k,
            "rank_sums": rank_sums,
            "H_unadjusted": H_unadjusted,
            "ties_count": len(ties_count),
            "correction_factor": correction_factor,
            "epsilon_sq": epsilon_sq
        }
        
    return H, p_value, epsilon_sq, k - 1, details