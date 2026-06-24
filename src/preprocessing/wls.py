"""
wls.py
------
Weighted Least Squares (WLS) GNSS positioning with two key improvements
over the previous gnss_lib_py version:

  1. Sagnac / Earth-rotation correction (Kaggle baseline notebook)
     The satellite positions in the derived CSV are in ECEF at the
     signal EMISSION time. While the signal travels ~67 ms, the Earth
     rotates by omega_e * 0.067 ≈ 0.00048 rad. At satellite altitude
     (~26 000 km), ignoring this rotation causes ~12 m of error per
     satellite, and can push WLS positions 50-200 m from the truth.

     Fix: rotate each satellite's x/y position by the angle the Earth
     turns during the signal flight time before calling the WLS solver.

  2. Warm-start across epochs (Kaggle simple_pipeline)
     Restarting the iterative WLS from [0,0,0,0] each epoch is slow
     and can converge to a wrong local minimum. Using the previous
     epoch's solution as the initial guess dramatically improves
     convergence speed and accuracy.

Reference:
  kaggle.com/c/google-smartphone-decimeter-challenge
  notebooks: gsdc-reproducing-baseline-wls-on-one-measurement
             least-squares-solution-from-gnss-derived-data
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.parsers.google_decimeter import (
    AndroidDerived2021,
    AndroidGroundTruth2021,
)

# ── physical constants ────────────────────────────────────────────────────────
OMEGA_E     = 7.2921151467e-5   # Earth rotation rate (rad/s)
LIGHT_SPEED = 299_792_458        # m/s
MIN_SATS    = 4                  # minimum satellites for valid WLS


# ═══════════════════════════════════════════════════════════════════
#  Sagnac / Earth-rotation correction
# ═══════════════════════════════════════════════════════════════════

def apply_sagnac_correction(sat_x: np.ndarray,
                             sat_y: np.ndarray,
                             sat_z: np.ndarray,
                             corr_pr: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate satellite ECEF positions to account for Earth's rotation
    during signal propagation (Sagnac / frame-dragging correction).

    Parameters
    ----------
    sat_x, sat_y, sat_z : (n,) satellite ECEF positions at emission (m)
    corr_pr             : (n,) corrected pseudoranges (m)

    Returns
    -------
    x_rot, y_rot, z_rot : (n,) corrected satellite positions (m)
    """
    # signal travel time (emission to reception)
    t_travel = corr_pr / LIGHT_SPEED          # ~67 ms

    # rotation angle (small angle, but ~30-100 m effect)
    angle = OMEGA_E * t_travel                # (n,) radians

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x_rot = cos_a * sat_x + sin_a * sat_y
    y_rot = -sin_a * sat_x + cos_a * sat_y
    z_rot = sat_z                             # z-axis unaffected

    return x_rot, y_rot, z_rot


# ═══════════════════════════════════════════════════════════════════
#  Custom iterative WLS  (warm-start capable)
# ═══════════════════════════════════════════════════════════════════

def iterative_wls(sat_pos:      np.ndarray,
                  pseudoranges: np.ndarray,
                  weights:      np.ndarray,
                  x0:           Optional[np.ndarray] = None,
                  max_iter:     int   = 15,
                  tol:          float = 1e-3
                  ) -> Tuple[np.ndarray, float]:
    """
    Iterative Weighted Least Squares for GNSS positioning.

    Estimates [x, y, z, cdt] where cdt = receiver clock bias in metres.

    Parameters
    ----------
    sat_pos      : (n, 3) satellite ECEF positions (Sagnac-corrected)
    pseudoranges : (n,)   corrected pseudoranges (m)
    weights      : (n,)   per-satellite weights (e.g. 1/rawPrUncM)
    x0           : (4,)   initial estimate [x, y, z, cdt]; zeros if None
    max_iter     : int    maximum iterations
    tol          : float  convergence threshold (m)

    Returns
    -------
    x_hat    : (4,) solution [x, y, z, cdt]
    residual : float mean absolute pseudorange residual (m)
    """
    n = len(pseudoranges)
    if n < MIN_SATS:
        return np.full(4, np.nan), np.nan

    x = np.zeros(4) if x0 is None else x0.copy().astype(float)
    W = np.diag(weights.astype(float))

    for _ in range(max_iter):
        diff   = sat_pos - x[:3]
        ranges = np.linalg.norm(diff, axis=1)

        # avoid division by zero
        ranges = np.maximum(ranges, 1.0)

        # pseudorange residuals
        dp = pseudoranges - ranges - x[3]

        # geometry / design matrix G: [-LOS_x, -LOS_y, -LOS_z, 1]
        G         = np.ones((n, 4))
        G[:, :3]  = -diff / ranges[:, None]

        # weighted normal equations: dx = (G^T W G)^{-1} G^T W dp
        try:
            GtW  = G.T @ W
            dx   = np.linalg.solve(GtW @ G, GtW @ dp)
        except np.linalg.LinAlgError:
            break

        x += dx

        if np.linalg.norm(dx[:3]) < tol:
            break

    # final residual
    ranges_final = np.linalg.norm(sat_pos - x[:3], axis=1)
    residual     = float(np.mean(np.abs(pseudoranges - ranges_final - x[3])))

    return x, residual


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline functions
# ═══════════════════════════════════════════════════════════════════

def compute_position_from_path(derived_csv_path: str,
                                ground_truth_path: str
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GSDC data and compute per-epoch WLS positions with:
      - Sagnac Earth-rotation correction on satellite positions
      - Warm-start across epochs (previous epoch → next epoch x0)

    Parameters
    ----------
    derived_csv_path  : path to *_derived.csv
    ground_truth_path : path to ground_truth.csv

    Returns
    -------
    pos_df : DataFrame — one row per epoch with WLS position & clock bias
    gt_df  : DataFrame — ground truth with ECEF columns
    """
    derived = AndroidDerived2021(derived_csv_path)
    gt      = AndroidGroundTruth2021(ground_truth_path)

    records     = []
    x_prev      = None          # warm start: None → zeros for first epoch
    epoch_count = 0

    for timestamp, _, subset in loop_time(derived, "gps_millis",
                                          delta_t_decimals=-2):
        # ── extract epoch data ────────────────────────────────────────────────
        try:
            sat_x   = np.atleast_1d(subset["x_sv_m"]).astype(float)
            sat_y   = np.atleast_1d(subset["y_sv_m"]).astype(float)
            sat_z   = np.atleast_1d(subset["z_sv_m"]).astype(float)
            corr_pr = np.atleast_1d(subset["corr_pr_m"]).astype(float)
            pr_unc  = np.atleast_1d(subset["raw_pr_sigma_m"]).astype(float)
        except Exception:
            records.append(_nan_row(int(timestamp)))
            epoch_count += 1
            continue

        # ── filter valid measurements ─────────────────────────────────────────
        valid = (
            ~np.isnan(corr_pr) &
            ~np.isnan(sat_x) &
            ~np.isnan(sat_y) &
            ~np.isnan(sat_z) &
            (pr_unc > 0) &
            np.isfinite(corr_pr) &
            np.isfinite(sat_x)
        )

        if valid.sum() < MIN_SATS:
            records.append(_nan_row(int(timestamp)))
            x_prev = None       # reset warm start after gap
            epoch_count += 1
            continue

        sat_x_v   = sat_x[valid]
        sat_y_v   = sat_y[valid]
        sat_z_v   = sat_z[valid]
        corr_pr_v = corr_pr[valid]
        pr_unc_v  = pr_unc[valid]

        # ── Fix A: reject very noisy signals ─────────────────────────────────
        quality_mask = pr_unc_v < 50.0          # keep signals with unc < 50m
        if quality_mask.sum() >= MIN_SATS:
            sat_x_v   = sat_x_v[quality_mask]
            sat_y_v   = sat_y_v[quality_mask]
            sat_z_v   = sat_z_v[quality_mask]
            corr_pr_v = corr_pr_v[quality_mask]
            pr_unc_v  = pr_unc_v[quality_mask]

        # ── Fix B: one signal per physical satellite (best quality) ───────────
        try:
            sv_ids     = np.atleast_1d(subset["sv_id"]).astype(str)[valid]
            gnss_ids   = np.atleast_1d(subset["gnss_id"]).astype(str)[valid]
            if quality_mask.sum() >= MIN_SATS:
                sv_ids   = sv_ids[quality_mask]
                gnss_ids = gnss_ids[quality_mask]
            sat_keys   = np.array([f"{g}_{s}" for g, s in zip(gnss_ids, sv_ids)])
            _, unique_idx = np.unique(sat_keys, return_index=True)
            # sort by pr_unc first so np.unique picks best signal
            sort_order  = np.argsort(pr_unc_v)
            sat_x_v     = sat_x_v[sort_order][np.sort(unique_idx)]
            sat_y_v     = sat_y_v[sort_order][np.sort(unique_idx)]
            sat_z_v     = sat_z_v[sort_order][np.sort(unique_idx)]
            corr_pr_v   = corr_pr_v[sort_order][np.sort(unique_idx)]
            pr_unc_v    = pr_unc_v[sort_order][np.sort(unique_idx)]
        except Exception:
            pass   # if sv_id not available, skip dedup

        if len(corr_pr_v) < MIN_SATS:
            records.append(_nan_row(int(timestamp)))
            x_prev = None
            epoch_count += 1
            continue

        weights_v = 1.0 / np.clip(pr_unc_v, 1e-3, None)



        # ── Sagnac correction ─────────────────────────────────────────────────
        sx, sy, sz = apply_sagnac_correction(
            sat_x_v, sat_y_v, sat_z_v, corr_pr_v
        )
        sat_pos = np.stack([sx, sy, sz], axis=1)    # (n, 3)

        # ── iterative WLS with warm start ─────────────────────────────────────
        try:
            x_hat, residual = iterative_wls(
                sat_pos, corr_pr_v, weights_v, x0=x_prev
            )
        except Exception:
            x_hat    = np.full(4, np.nan)
            residual = np.nan

        if np.isnan(x_hat).any() or np.linalg.norm(x_hat[:3]) < 1e3:
            # Solution looks invalid (e.g. converged near origin)
            records.append(_nan_row(int(timestamp)))
            x_prev = None
            epoch_count += 1
            continue

        x_prev = x_hat.copy()   # warm start for next epoch

        records.append({
            "millisSinceGpsEpoch": int(timestamp),
            "xWlsM":       float(x_hat[0]),
            "yWlsM":       float(x_hat[1]),
            "zWlsM":       float(x_hat[2]),
            "cdtM":        float(x_hat[3]),
            "n_sats_used": int(valid.sum()),
        })

        epoch_count += 1
        if epoch_count % 500 == 0:
            print(f"  Processed {epoch_count} epochs...")

    pos_df = pd.DataFrame(records)
    gt_df  = _gt_to_df(gt)
    return pos_df, gt_df


def _nan_row(timestamp: int) -> dict:
    return {
        "millisSinceGpsEpoch": timestamp,
        "xWlsM": np.nan, "yWlsM": np.nan, "zWlsM": np.nan,
        "cdtM":  np.nan, "n_sats_used": 0,
    }


def _gt_to_df(gt: AndroidGroundTruth2021) -> pd.DataFrame:
    """Convert AndroidGroundTruth2021 NavData → DataFrame."""
    gt_df  = gt.pandas_df()
    rename = {
        "gps_millis":    "millisSinceGpsEpoch",
        "lat_rx_gt_deg": "latDeg",
        "lon_rx_gt_deg": "lngDeg",
        "alt_rx_gt_m":   "heightAboveWgs84EllipsoidM",
        "x_rx_gt_m":     "xEcefM",
        "y_rx_gt_m":     "yEcefM",
        "z_rx_gt_m":     "zEcefM",
    }
    return gt_df.rename(columns={k: v for k, v in rename.items()
                                  if k in gt_df.columns})


def position_error(pos_df: pd.DataFrame,
                   gt_df:  pd.DataFrame) -> pd.DataFrame:
    """Compute 3D and horizontal (N/E) position errors vs ground truth."""
    merged = pd.merge_asof(
        pos_df.sort_values("millisSinceGpsEpoch"),
        gt_df[["millisSinceGpsEpoch",
               "xEcefM", "yEcefM", "zEcefM",
               "latDeg", "lngDeg"]].sort_values("millisSinceGpsEpoch"),
        on="millisSinceGpsEpoch",
        tolerance=2000,
        direction="nearest",
    )

    dx = merged["xWlsM"] - merged["xEcefM"]
    dy = merged["yWlsM"] - merged["yEcefM"]
    dz = merged["zWlsM"] - merged["zEcefM"]
    merged["posError3dM"] = np.sqrt(dx**2 + dy**2 + dz**2)

    lat = np.radians(merged["latDeg"].values)
    lon = np.radians(merged["lngDeg"].values)

    n_x = -np.sin(lat) * np.cos(lon)
    n_y = -np.sin(lat) * np.sin(lon)
    n_z =  np.cos(lat)
    e_x = -np.sin(lon)
    e_y =  np.cos(lon)

    dx_v = dx.values
    dy_v = dy.values
    dz_v = dz.values

    err_north = dx_v * n_x + dy_v * n_y + dz_v * n_z
    err_east  = dx_v * e_x + dy_v * e_y

    merged["posErrorNorthM"] = np.abs(err_north)
    merged["posErrorEastM"]  = np.abs(err_east)
    merged["posErrorHorizM"] = np.sqrt(err_north**2 + err_east**2)

    return merged


def compute_wls_all_epochs(smoothed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibility wrapper for feature_builder.py.
    Runs Sagnac-corrected + warm-start WLS on a smoothed DataFrame.
    """
    epochs  = np.sort(smoothed_df["millisSinceGpsEpoch"].unique())
    records = []
    x_prev  = None

    for i, epoch_ms in enumerate(epochs):
        ep = smoothed_df[smoothed_df["millisSinceGpsEpoch"] == epoch_ms].copy()

        cn0_col = "Cn0DbHz" if "Cn0DbHz" in ep.columns else None
        if cn0_col and ep[cn0_col].notna().any():
            ep = ep.sort_values(cn0_col, ascending=False)
        else:
            ep = ep.sort_values("rawPrUncM", ascending=True)
        ep = ep.drop_duplicates(
            subset=["svid", "constellationType"], keep="first"
        ).head(20)

        pr_col  = "smoothedPrM" if "smoothedPrM" in ep.columns else "correctedPrM"
        sat_x   = ep["xSatPosM"].values.astype(float)
        sat_y   = ep["ySatPosM"].values.astype(float)
        sat_z   = ep["zSatPosM"].values.astype(float)
        corr_pr = ep[pr_col].values.astype(float)
        pr_unc  = ep["rawPrUncM"].values.astype(float)

        valid   = (
            ~np.isnan(corr_pr) & ~np.isnan(sat_x) &
            (pr_unc > 0) & np.isfinite(corr_pr)
        )

        x = y = z = cdt = residual = np.nan
        n_sats = valid.sum()

        if n_sats >= MIN_SATS:
            sx, sy, sz = apply_sagnac_correction(
                sat_x[valid], sat_y[valid], sat_z[valid], corr_pr[valid]
            )
            sat_pos = np.stack([sx, sy, sz], axis=1)
            weights = 1.0 / np.clip(pr_unc[valid], 1e-3, None)

            try:
                x_hat, res = iterative_wls(
                    sat_pos, corr_pr[valid], weights, x0=x_prev
                )
                if not np.isnan(x_hat).any() and np.linalg.norm(x_hat[:3]) > 1e3:
                    x, y, z, cdt = x_hat
                    residual      = res
                    x_prev        = x_hat.copy()
            except Exception:
                pass

        records.append({
            "millisSinceGpsEpoch": epoch_ms,
            "xWlsM": x, "yWlsM": y, "zWlsM": z, "cdtM": cdt,
            "wlsResidualM": residual, "n_sats_used": n_sats,
        })
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(epochs)} epochs...")

    return pd.DataFrame(records)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    phone_dir    = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"
    phone_name   = os.path.basename(phone_dir)
    derived_path = os.path.join(phone_dir, f"{phone_name}_derived.csv")
    gt_path      = os.path.join(phone_dir, "ground_truth.csv")

    print("Running WLS with Sagnac correction + warm start...")
    pos_df, gt_df = compute_position_from_path(derived_path, gt_path)

    print(f"\nShape      : {pos_df.shape}")
    print(f"NaN pos    : {pos_df['xWlsM'].isna().sum()}")
    print(f"Mean sats  : {pos_df['n_sats_used'].mean():.1f}")

    print("\nComputing errors...")
    err_df = position_error(pos_df, gt_df)
    valid  = err_df.dropna(subset=["posError3dM"])

    print(f"\n{'Metric':<8} {'3D (m)':>8} {'Horiz (m)':>10} "
          f"{'North (m)':>10} {'East (m)':>10}")
    print("-" * 50)
    for stat, fn in [("Mean",   np.mean), ("Median", np.median),
                     ("Min",    np.min),  ("Max",    np.max)]:
        print(f"{stat:<8} "
              f"{fn(valid['posError3dM']):>8.2f} "
              f"{fn(valid['posErrorHorizM']):>10.2f} "
              f"{fn(valid['posErrorNorthM']):>10.2f} "
              f"{fn(valid['posErrorEastM']):>10.2f}")

    print(f"\nHorizontal error distribution:")
    bins   = [0, 2, 5, 10, 20, 50, 100, float("inf")]
    labels = ["0-2","2-5","5-10","10-20","20-50","50-100",">100"]
    counts = pd.cut(valid["posErrorHorizM"], bins=bins,
                    labels=labels).value_counts().sort_index()
    for label, count in counts.items():
        pct = 100 * count / len(valid)
        bar = "+" * int(pct / 2)
        print(f"  {label:>8} m : {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\nPrevious WLS (gnss_lib_py): ~35 m horiz mean, ~20 m median")
    print(f"Target (Kaggle baseline)  : ~10 m horiz mean")
