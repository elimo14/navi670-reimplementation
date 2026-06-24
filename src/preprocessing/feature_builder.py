"""
feature_builder.py
------------------
Builds the 6D node feature vectors for the GNN graph.

From Section 2.3.2 of the base paper (navi_670):
  "This vector is formulated by concatenating:
     1. LOS vector (3D)
     2. Range residual (1D)
     3. C/N0 (1D)
     4. Pseudorange uncertainty (1D)"

Feature vector shape per satellite: (6,)
Max nodes per graph: 20 (Table 4 of navi_670)

Correct residual formula:
  residual = correctedPrM - ||sat_pos - rx_pos|| - cdt
  (clock bias cdt from WLS must be subtracted)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── constants (Table 4 of navi_670) ──────────────────────────────────────────
MAX_NODES   = 20
FEATURE_DIM = 6
MIN_SATS    = 4


# ═══════════════════════════════════════════════════════════════════
#  Core geometry functions
# ═══════════════════════════════════════════════════════════════════

def compute_los_and_residual(rx_pos:       np.ndarray,
                              sat_pos:      np.ndarray,
                              pseudoranges: np.ndarray,
                              cdt:          float = 0.0
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LOS unit vectors and clock-corrected residuals.

    Parameters
    ----------
    rx_pos      : (3,) ECEF receiver position in metres
    sat_pos     : (n, 3) ECEF satellite positions in metres
    pseudoranges: (n,) corrected pseudoranges in metres
    cdt         : float receiver clock bias in metres (from WLS)

    Returns
    -------
    los_vecs  : (n, 3) unit vectors from receiver → satellite
    residuals : (n,) = correctedPrM - geometric_range - cdt
    """
    diff      = sat_pos - rx_pos                              # (n, 3)
    ranges    = np.linalg.norm(diff, axis=1)                  # (n,)
    los_vecs  = diff / (ranges[:, np.newaxis] + 1e-10)       # (n, 3)
    residuals = pseudoranges - ranges - cdt                   # (n,)
    return los_vecs, residuals


def build_feature_matrix(rx_pos:       np.ndarray,
                          sat_pos:      np.ndarray,
                          pseudoranges: np.ndarray,
                          cn0:          np.ndarray,
                          pr_unc:       np.ndarray,
                          cdt:          float = 0.0
                          ) -> np.ndarray:
    """
    Build (n, 6) feature matrix for one epoch.

    Layout (Figure 3 of navi_670):
      cols 0:3 — LOS vector      (3D unit vector)
      col  3   — range residual  (correctedPrM - range - cdt)
      col  4   — C/N0            (dB-Hz)
      col  5   — PR uncertainty  (metres)
    """
    # ── fill NaN values with reasonable defaults ──────────────────────────────
    cn0_f = cn0.copy().astype(float)
    valid_cn0 = ~np.isnan(cn0_f)
    if valid_cn0.any():
        cn0_f[~valid_cn0] = np.median(cn0_f[valid_cn0])
    else:
        cn0_f[:] = 25.0   # typical mid-quality signal

    unc_f = pr_unc.copy().astype(float)
    valid_unc = ~np.isnan(unc_f)
    if valid_unc.any():
        unc_f[~valid_unc] = np.max(unc_f[valid_unc])
    else:
        unc_f[:] = 10.0   # high uncertainty default

    los_vecs, residuals = compute_los_and_residual(
        rx_pos, sat_pos, pseudoranges, cdt
    )

    X = np.column_stack([
        los_vecs,                   # cols 0:3
        residuals.reshape(-1, 1),   # col  3
        cn0_f.reshape(-1, 1),       # col  4
        unc_f.reshape(-1, 1),       # col  5
    ])
    return X.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Epoch-level feature building from merged DataFrame
# ═══════════════════════════════════════════════════════════════════

def _select_and_dedup(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Deduplicate: one signal per physical satellite (best C/N0).
    2. Select: top MAX_NODES satellites by C/N0.
    """
    cn0_col   = "Cn0DbHz"       if "Cn0DbHz"       in epoch_df.columns else None
    unc_col   = "rawPrUncM"     if "rawPrUncM"     in epoch_df.columns else None
    svid_col  = "svid"          if "svid"          in epoch_df.columns else "sv_id"
    const_col = "constellationType" if "constellationType" in epoch_df.columns \
                else "gnss_id"

    # sort by quality
    if cn0_col and epoch_df[cn0_col].notna().any():
        df = epoch_df.sort_values(cn0_col, ascending=False)
    elif unc_col:
        df = epoch_df.sort_values(unc_col, ascending=True)
    else:
        df = epoch_df.copy()

    # one signal per physical satellite
    df = df.drop_duplicates(subset=[svid_col, const_col], keep="first")

    # cap at MAX_NODES
    if len(df) > MAX_NODES:
        df = df.head(MAX_NODES)

    return df


def build_epoch_features_from_row(epoch_df:   pd.DataFrame,
                                   rx_pos:     np.ndarray,
                                   cdt:        float,
                                   epoch_ms:   int,
                                   gt_pos:     Optional[np.ndarray] = None
                                   ) -> Optional[Dict]:
    """
    Build feature dict for one epoch using a merged DataFrame.

    Parameters
    ----------
    epoch_df  : rows for this epoch (from merged derived+log DataFrame)
    rx_pos    : (3,) WLS ECEF position
    cdt       : float clock bias from WLS (metres)
    epoch_ms  : int epoch timestamp
    gt_pos    : (3,) ground truth ECEF position (optional)

    Returns
    -------
    dict or None if insufficient satellites
    """
    df = _select_and_dedup(epoch_df)

    # ── extract satellite positions ───────────────────────────────────────────
    x_col = "xSatPosM" if "xSatPosM" in df.columns else "x_sv_m"
    y_col = "ySatPosM" if "ySatPosM" in df.columns else "y_sv_m"
    z_col = "zSatPosM" if "zSatPosM" in df.columns else "z_sv_m"
    sat_pos = df[[x_col, y_col, z_col]].values.astype(float)

    # ── pseudorange ───────────────────────────────────────────────────────────
    pr_col = ("correctedPrM" if "correctedPrM" in df.columns
              else "smoothedPrM" if "smoothedPrM" in df.columns
              else "corr_pr_m")
    prs = df[pr_col].values.astype(float)

    # ── C/N0 ─────────────────────────────────────────────────────────────────
    cn0_col = ("Cn0DbHz" if "Cn0DbHz" in df.columns
               else "cn0_dbhz" if "cn0_dbhz" in df.columns else None)
    cn0 = df[cn0_col].values.astype(float) if cn0_col \
          else np.full(len(df), np.nan)

    # ── pseudorange uncertainty ───────────────────────────────────────────────
    unc_col = ("rawPrUncM" if "rawPrUncM" in df.columns
               else "raw_pr_sigma_m" if "raw_pr_sigma_m" in df.columns else None)
    pr_unc = df[unc_col].values.astype(float) if unc_col \
             else np.full(len(df), np.nan)

    # ── filter valid rows ─────────────────────────────────────────────────────
    valid = (~np.isnan(sat_pos).any(axis=1) & ~np.isnan(prs))
    if valid.sum() < MIN_SATS:
        return None

    sat_pos = sat_pos[valid]
    prs     = prs[valid]
    cn0     = cn0[valid]
    pr_unc  = pr_unc[valid]

    # ── build features ────────────────────────────────────────────────────────
    X = build_feature_matrix(rx_pos, sat_pos, prs, cn0, pr_unc, cdt)
    los_vecs, residuals = compute_los_and_residual(rx_pos, sat_pos, prs, cdt)

    result = {
        "epoch_ms":   epoch_ms,
        "rx_pos_wls": rx_pos,
        "cdt":        cdt,
        "X":          X,                              # (n, 6) float32
        "sat_pos":    sat_pos,                        # (n, 3) float64
        "los_vecs":   los_vecs.astype(np.float32),   # (n, 3)
        "residuals":  residuals.astype(np.float32),  # (n,)
        "n_sats":     int(valid.sum()),
    }

    if gt_pos is not None and not np.isnan(gt_pos).any():
        result["gt_pos"]    = gt_pos
        result["correction"] = gt_pos - rx_pos       # what GNN+BKF must learn
    else:
        result["gt_pos"]    = np.full(3, np.nan)
        result["correction"] = np.full(3, np.nan)

    return result


def build_all_epoch_features(merged_df: pd.DataFrame,
                              pos_df:    pd.DataFrame,
                              gt_df:     Optional[pd.DataFrame] = None
                              ) -> List[Dict]:
    """
    Build feature dicts for ALL epochs using the merged DataFrame.

    Parameters
    ----------
    merged_df : output of parser.load_phone_data()['derived']
                must have: millisSinceGpsEpoch, xSatPosM, correctedPrM,
                           rawPrUncM, Cn0DbHz (if available from GnssLog)
    pos_df    : WLS positions from wls.compute_position_from_path()
    gt_df     : ground truth with xEcefM/yEcefM/zEcefM columns

    Returns
    -------
    list of feature dicts, one per valid epoch
    """
    epochs = np.sort(merged_df["millisSinceGpsEpoch"].unique())

    # index pos_df and gt_df by timestamp for fast lookup
    pos_index = pos_df.set_index("millisSinceGpsEpoch")
    gt_index  = (gt_df.set_index("millisSinceGpsEpoch")
                 if gt_df is not None and "xEcefM" in gt_df.columns
                 else None)

    epoch_features = []
    skipped = 0

    for epoch_ms in epochs:
        epoch_df = merged_df[merged_df["millisSinceGpsEpoch"] == epoch_ms]

        # ── WLS position for this epoch ───────────────────────────────────────
        pos_row = _nearest_index(pos_index, epoch_ms, tol=2000)
        if pos_row is None or np.isnan(pos_row["xWlsM"]):
            skipped += 1
            continue

        rx_pos = np.array([pos_row["xWlsM"],
                            pos_row["yWlsM"],
                            pos_row["zWlsM"]], dtype=np.float64)
        cdt    = float(pos_row["cdtM"]) if not np.isnan(pos_row["cdtM"]) else 0.0

        # ── ground truth ──────────────────────────────────────────────────────
        gt_pos = None
        if gt_index is not None:
            gt_row = _nearest_index(gt_index, epoch_ms, tol=2000)
            if gt_row is not None:
                gt_pos = np.array([gt_row["xEcefM"],
                                   gt_row["yEcefM"],
                                   gt_row["zEcefM"]], dtype=np.float64)

        # ── build features ────────────────────────────────────────────────────
        feat = build_epoch_features_from_row(
            epoch_df, rx_pos, cdt, int(epoch_ms), gt_pos
        )
        if feat is None:
            skipped += 1
            continue

        epoch_features.append(feat)

    print(f"  Built features: {len(epoch_features)} valid, {skipped} skipped")
    return epoch_features


def _nearest_index(index_df: pd.DataFrame,
                   epoch_ms:  int,
                   tol:       int = 2000):
    """Return row from epoch-indexed DataFrame nearest to epoch_ms."""
    if index_df.empty:
        return None
    idx   = index_df.index.values
    diffs = np.abs(idx - epoch_ms)
    if diffs.min() > tol:
        return None
    return index_df.iloc[diffs.argmin()]


# ═══════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing.parser       import load_phone_data
    from src.preprocessing.wls         import compute_position_from_path

    phone_dir    = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"
    phone_name   = os.path.basename(phone_dir)
    derived_path = os.path.join(phone_dir, f"{phone_name}_derived.csv")
    gt_path      = os.path.join(phone_dir, "ground_truth.csv")

    # ── Step 1: WLS positions ─────────────────────────────────────────────────
    print("Step 1: WLS positions...")
    pos_df, gt_df = compute_position_from_path(derived_path, gt_path)
    print(f"  {len(pos_df)} epochs, {pos_df['xWlsM'].isna().sum()} NaN")

    # ── Step 2: merged data (has C/N0 from GnssLog) ───────────────────────────
    print("\nStep 2: Loading merged data...")
    data      = load_phone_data(phone_dir)
    merged_df = data["derived"]
    print(f"  Rows: {len(merged_df)}")
    has_cn0   = "Cn0DbHz" in merged_df.columns
    cn0_valid = merged_df["Cn0DbHz"].notna().sum() if has_cn0 else 0
    print(f"  C/N0 available: {has_cn0}  "
          f"({cn0_valid}/{len(merged_df)} non-null)")

    # ── Step 3: build features ────────────────────────────────────────────────
    print("\nStep 3: Building features...")
    epoch_features = build_all_epoch_features(merged_df, pos_df, gt_df)

    # ── Step 4: inspect ───────────────────────────────────────────────────────
    if epoch_features:
        ep = epoch_features[0]
        cdt_val = ep["cdt"]

        print(f"\nFirst epoch ({ep['epoch_ms']}):")
        print(f"  Satellites    : {ep['n_sats']}")
        print(f"  Feature shape : {ep['X'].shape}")
        print(f"  WLS pos (ECEF): {ep['rx_pos_wls'].round(1)}")
        print(f"  Clock bias    : {cdt_val:.2f} m  "
              f"({cdt_val/299792458*1e6:.2f} μs)")
        print(f"  Correction (m): {ep['correction'].round(2)}")
        print(f"  |correction|  : {np.linalg.norm(ep['correction']):.2f} m")

        print(f"\n  Feature matrix (first 3 rows):")
        print(f"  {'LOS_x':>7} {'LOS_y':>7} {'LOS_z':>7} "
              f"{'Resid':>8} {'C/N0':>7} {'Unc':>6}")
        print(f"  {'-'*52}")
        for row in ep["X"][:3]:
            print("  " + "  ".join(f"{v:7.3f}" for v in row))

        # ── summary ───────────────────────────────────────────────────────────
        print(f"\nSummary over {len(epoch_features)} epochs:")
        all_n    = [ep["n_sats"] for ep in epoch_features]
        corrs    = np.array([ep["correction"] for ep in epoch_features
                              if not np.isnan(ep["correction"]).any()])
        all_X    = np.vstack([ep["X"] for ep in epoch_features])

        print(f"  Mean sats     : {np.mean(all_n):.1f} "
              f"(min={np.min(all_n)}, max={np.max(all_n)})")
        if len(corrs):
            norms = np.linalg.norm(corrs, axis=1)
            print(f"  Mean |corr|   : {norms.mean():.2f} m")
            print(f"  Median |corr| : {np.median(norms):.2f} m")

        print(f"\nFeature statistics:")
        names = ["LOS_x", "LOS_y", "LOS_z", "Residual", "C/N0", "Uncertainty"]
        for i, name in enumerate(names):
            col = all_X[:, i]
            print(f"  {name:<12}: mean={col.mean():8.2f}  "
                  f"std={col.std():7.2f}  "
                  f"min={col.min():8.2f}  max={col.max():8.2f}")

        # ── check residuals are small now (clock bias subtracted) ─────────────
        res_mean = all_X[:, 3].mean()
        res_std  = all_X[:, 3].std()
        status   = "✅ GOOD" if abs(res_mean) < 100 else "⚠️  Still large"
        print(f"\nResidual check: mean={res_mean:.1f} m  "
              f"std={res_std:.1f} m  {status}")
        print(f"(Expected: |mean| < 100 m after clock bias subtraction)")
