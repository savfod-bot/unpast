"""Cluster binarized genes"""

import numpy as np
import pandas as pd

from unpast.utils.backend import to_numpy as _to_numpy
from unpast.utils.backend import xp as _xp
from unpast.utils.logs import get_logger, log_function_duration

logger = get_logger(__name__)


@log_function_duration(name="Similarity ARI")
def get_similarity_ari(binarized_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Adjusted Rand Index between features based on binary patterns.

    Treats each feature column as a binary clustering of samples, then computes
    the ARI between all pairs of features using vectorized operations.

    Args:
        binarized_data: Binary matrix with samples as rows, features as columns.
            Values should be 0 or 1.

    Returns:
        Symmetric similarity matrix with ARI values in [-1, 1].
        Diagonal entries are 1.0.
    """
    # Backend-agnostic: runs on numpy (CPU) or cupy (GPU). On a CPU-only box
    # `xp` is numpy. The matmul `X.T @ X` is the O(features^2) hot path.
    X_np = binarized_data.to_numpy(dtype=int)
    xp = _xp(X_np)
    X = xp.asarray(X_np)
    n_samples = X.shape[0]

    # Contingency table components for all feature pairs
    n11 = X.T @ X
    n1 = X.sum(axis=0)
    n10 = n1[:, None] - n11
    n01 = n10.T
    n00 = n_samples - n01 - n10 - n11

    def comb2(k):
        return k * (k - 1) / 2

    index = comb2(n00) + comb2(n01) + comb2(n10) + comb2(n11)
    self_comb = comb2(n1) + comb2(n_samples - n1)
    si = self_comb[:, None]
    sj = self_comb[None, :]
    expected_index = si * sj / comb2(n_samples)
    max_index = (si + sj) / 2

    numerator = index - expected_index
    denominator = max_index - expected_index

    with np.errstate(divide="ignore", invalid="ignore"):
        # if denominator is 0, set ARI to 1.0 (perfect match)
        # (not expected to happen from the binarization step)
        # using 0.1 threshold to avoid floating point issues
        ari = xp.where(xp.abs(denominator) > 0.1, numerator / denominator, 1.0)

    ari = ari.clip(-1.0, 1.0)  # Ensure valid range in case of floating point errors
    # Convert back to numpy at the boundary for pandas / downstream CPU code.
    ari = _to_numpy(ari)
    ari_df = pd.DataFrame(
        ari, index=binarized_data.columns, columns=binarized_data.columns
    )

    logger.debug(
        f"ARI similarities for binarized data with shape {binarized_data.shape} computed."
    )
    return ari_df


@log_function_duration(name="Jaccard Similarity")
def get_similarity_jaccard(binarized_data):  # ,J=0.5
    """Calculate Jaccard similarity matrix between features based on binary expression patterns.

    Args:
        binarized_data (DataFrame): binary expression matrix with samples as rows and features as columns

    Returns:
        DataFrame: symmetric similarity matrix with Jaccard coefficients between all feature pairs
    """
    genes = binarized_data.columns.values
    n_samples = binarized_data.shape[0]
    size_threshold = int(min(0.45 * n_samples, (n_samples) / 2 - 10))
    # print("size threshold",size_threshold)
    n_genes = binarized_data.shape[1]

    # Backend-agnostic + matmul-vectorized. Runs on numpy (CPU) or cupy (GPU);
    # on a CPU-only box `xp` is numpy. The intersection matmul B @ B.T is the
    # O(features^2) hot path. Membership matrix B: features (rows) x samples.
    B_np = np.asarray(binarized_data.T, dtype=np.float64)
    xp = _xp(B_np)
    B = xp.asarray(B_np)

    rowsum = B.sum(axis=1)  # |g_i|, per feature
    inter = B @ B.T  # |g_i & g_j|
    union = rowsum[:, None] + rowsum[None, :] - inter  # |g_i | g_j|

    # Complement-matching variants (mirror the original elif logic):
    #   ~g1 & g2  -> |g2| - inter ;  |~g1 | g2| = n - |g1| + inter
    #   g1 & ~g2  -> |g1| - inter ;  |g1 | ~g2| = n - |g2| + inter
    union_c1 = n_samples - rowsum[:, None] + inter
    union_c2 = n_samples - rowsum[None, :] + inter

    # Divide only where the union (denominator) is non-empty; a 0/0 ratio in the
    # original collapses to 0 via max(jaccard, jaccard_c) anyway (both features
    # empty in that union -> no overlap), so map empty-union cells to 0.
    def _safe_div(num, den):
        return xp.where(den > 0, num / xp.where(den > 0, den, 1.0), 0.0)

    jaccard = _safe_div(inter, union)
    jac_c1 = _safe_div(rowsum[None, :] - inter, union_c1)
    jac_c2 = _safe_div(rowsum[:, None] - inter, union_c2)

    # Select complement variant per the original (asymmetric) elif rule. The
    # original loop fixes g1 = lower-indexed feature, g2 = higher-indexed, and
    # mirrors the result, so the complement preference goes to the smaller index
    # when both features exceed the threshold:
    #   if |g_row| > threshold: use jac_c1 (complement of the row feature)
    #   elif |g_col| > threshold: use jac_c2 (complement of the col feature)
    #   else: 0
    # Computed on the upper triangle (row < col), then symmetrized.
    big_i = rowsum[:, None] > size_threshold
    big_j = rowsum[None, :] > size_threshold
    jaccard_c = xp.where(big_i, jac_c1, xp.where(big_j, jac_c2, 0.0))

    upper = xp.maximum(jaccard, jaccard_c)
    upper = xp.triu(upper, k=1)  # keep strict upper triangle (row < col)
    results = upper + upper.T  # symmetric, zero diagonal
    # Diagonal is exactly 1 (a feature vs itself), as in the original.
    xp.fill_diagonal(results, 1.0)

    # Convert back to numpy at the boundary for pandas / downstream CPU code.
    results = _to_numpy(results)
    results = pd.DataFrame(data=results, index=genes, columns=genes)
    logger.debug(
        f"Jaccard similarities for {binarized_data.shape[1]} features computed."
    )
    return results


# @log_function_duration(name="Pearson Similarity")

# def get_similarity_corr(df, verbose=True):
#     """Calculate correlation-based similarity matrix between features.

#     Args:
#         df (DataFrame): expression matrix with features as columns
#         verbose (bool): whether to print progress information

#     Returns:
#         DataFrame: correlation similarity matrix with positive correlations only
#     """
#     corr = df.corr()  # .applymap(abs)
#     corr = corr[corr > 0]  # to consider only direct correlations
#     corr = corr.fillna(0)
#     return corr
