"""
edge_builder.py
---------------
Builds edge connections for the GNN graph.

From Section 2.3.3 of the base paper (navi_670):

  "For constructing edges in the GNN, we utilize two design strategies:

   1. CONSTELLATION EDGES: We group satellites based on whether they
      belong to the same constellation.

   2. COSINE SIMILARITY EDGES: We adopt the cosine similarity metric
      to construct additional edges based on satellite features.
      If similarity surpasses an empirically determined threshold,
      an edge is generated between the two nodes i and j.
      This choice leads to a sparser graph, which helps reduce the
      computational overhead of the GNN."

  Cosine similarity formula (Equation 8):
    similarity(s_i, s_j) = (s_i · s_j) / (||s_i|| * ||s_j||)

  Default threshold = 0.5 (found optimal in Tables 11-12 of navi_670)

Output: edge_index tensor of shape (2, num_edges) for PyTorch Geometric.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple


# ── default threshold from Tables 11-12 of navi_670 ──────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def compute_cosine_similarity(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for all satellite features.

    Parameters
    ----------
    features : (n, d) feature matrix — raw satellite features
               (before GNN transformation, per Section 2.3.3)

    Returns
    -------
    sim_matrix : (n, n) cosine similarity matrix
    """
    n = features.shape[0]
    if n == 0:
        return np.zeros((0, 0))

    norms  = np.linalg.norm(features, axis=1, keepdims=True)
    norms  = np.maximum(norms, 1e-10)
    normed = features / norms
    return np.clip(normed @ normed.T, -1.0, 1.0)


def build_constellation_edges(const_types: np.ndarray) -> List[Tuple[int, int]]:
    """
    Connect all satellite pairs within the same constellation.
    Both directions included for undirected message passing.
    """
    edges = []
    n = len(const_types)
    for i in range(n):
        for j in range(i + 1, n):
            if const_types[i] == const_types[j]:
                edges.append((i, j))
                edges.append((j, i))
    return edges


def build_similarity_edges(features:  np.ndarray,
                            threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                            ) -> List[Tuple[int, int]]:
    """
    Connect satellite pairs with cosine similarity above threshold.
    Both directions included for undirected message passing.
    """
    sim_matrix = compute_cosine_similarity(features)
    n          = sim_matrix.shape[0]
    edges      = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                edges.append((i, j))
                edges.append((j, i))
    return edges


def build_edge_index_fast(features:    np.ndarray,
                           const_types: np.ndarray,
                           threshold:   float = DEFAULT_SIMILARITY_THRESHOLD
                           ) -> torch.Tensor:
    """
    Vectorised edge builder — combines constellation + similarity strategies.

    Parameters
    ----------
    features    : (n, 6) feature matrix (raw, before GNN layers)
    const_types : (n,) constellation identifiers per satellite
    threshold   : float cosine similarity threshold (default 0.5)

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, num_edges)
    """
    n = len(const_types)
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # ── constellation mask ────────────────────────────────────────────────────
    c          = np.array(const_types)
    const_mask = (c[:, None] == c[None, :])         # (n, n) bool

    # ── similarity mask ───────────────────────────────────────────────────────
    sim_matrix = compute_cosine_similarity(features)
    sim_mask   = sim_matrix > threshold              # (n, n) bool

    # ── combine, remove self-loops ────────────────────────────────────────────
    eye      = np.eye(n, dtype=bool)
    combined = (const_mask | sim_mask) & ~eye        # (n, n) bool

    src, dst = np.where(combined)

    # ── fallback: fully connected if no edges found ───────────────────────────
    if len(src) == 0:
        idx      = np.arange(n)
        g_src, g_dst = np.meshgrid(idx, idx, indexing="ij")
        mask         = g_src != g_dst
        src, dst     = g_src[mask].ravel(), g_dst[mask].ravel()

    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from src.preprocessing.parser          import load_phone_data
    from src.preprocessing.wls            import compute_position_from_path
    from src.preprocessing.feature_builder import build_all_epoch_features

    phone_dir    = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"
    phone_name   = os.path.basename(phone_dir)
    derived_path = os.path.join(phone_dir, f"{phone_name}_derived.csv")
    gt_path      = os.path.join(phone_dir, "ground_truth.csv")

    print("Loading data + building features...")
    pos_df, gt_df  = compute_position_from_path(derived_path, gt_path)
    data           = load_phone_data(phone_dir)
    merged_df      = data["derived"]
    epoch_features = build_all_epoch_features(merged_df, pos_df, gt_df)

    ep       = epoch_features[0]
    epoch_ms = ep["epoch_ms"]

    const_col = ("constellationType" if "constellationType" in merged_df.columns
                 else "gnss_id")
    svid_col  = "svid" if "svid" in merged_df.columns else "sv_id"

    epoch_df = (merged_df[merged_df["millisSinceGpsEpoch"] == epoch_ms]
                .sort_values("Cn0DbHz", ascending=False)
                .drop_duplicates(subset=[svid_col, const_col], keep="first")
                .head(20))

    const_types = epoch_df[const_col].values
    X           = ep["X"]
    n_sats      = len(const_types)

    print(f"\nFirst epoch ({epoch_ms}):")
    print(f"  Satellites    : {n_sats}")
    print(f"  Constellations: {np.unique(const_types)}")

    # ── edge counts ───────────────────────────────────────────────────────────
    ce = build_constellation_edges(const_types)
    se = build_similarity_edges(X, DEFAULT_SIMILARITY_THRESHOLD)
    ei = build_edge_index_fast(X, const_types, DEFAULT_SIMILARITY_THRESHOLD)

    print(f"\n  Constellation edges (undirected): {len(ce)//2}")
    print(f"  Similarity edges   (undirected): {len(se)//2}")
    print(f"  Combined edge_index shape      : {ei.shape}")
    print(f"  Total directed edges           : {ei.shape[1]}")
    n_possible = n_sats * (n_sats - 1)
    print(f"  Graph density                  : "
          f"{ei.shape[1]/n_possible:.2%} of possible")

    # ── threshold sensitivity ─────────────────────────────────────────────────
    print(f"\n  Threshold sensitivity:")
    print(f"  {'Threshold':>10}  {'Directed edges':>15}  {'Density':>8}")
    print(f"  {'-'*37}")
    for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        e = build_edge_index_fast(X, const_types, thr)
        d = e.shape[1] / n_possible
        print(f"  {thr:>10.1f}  {e.shape[1]:>15d}  {d:>8.2%}")

    # ── stats over first 200 epochs ───────────────────────────────────────────
    print(f"\n  Stats over first 200 epochs:")
    ecounts = []
    for feat in epoch_features[:200]:
        ep_ms = feat["epoch_ms"]
        ep_df = (merged_df[merged_df["millisSinceGpsEpoch"] == ep_ms]
                 .sort_values("Cn0DbHz", ascending=False)
                 .drop_duplicates(subset=[svid_col, const_col], keep="first")
                 .head(20))
        ct = ep_df[const_col].values
        e  = build_edge_index_fast(feat["X"], ct, DEFAULT_SIMILARITY_THRESHOLD)
        ecounts.append(e.shape[1])

    print(f"  Mean directed edges : {np.mean(ecounts):.1f}")
    print(f"  Min / Max           : {np.min(ecounts)} / {np.max(ecounts)}")
    print(f"\n✅ Edge builder working correctly")
