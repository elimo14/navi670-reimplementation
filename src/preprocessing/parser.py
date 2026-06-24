"""
parser.py
---------
Loads GSDC data using gnss_lib_py's AndroidDerived2021 and
AndroidGroundTruth2021 classes, which handle all known corrections:

  Correction 1 (timestamp shift): maps derived timestamps to previous
    epoch for alignment with ground truth and Raw log data.
  Correction 5 (timing outliers): removes measurements where signal
    travel time is outside 0-300ms (physically impossible values).
  Ground truth altitude correction: subtracts 61 m from reported
    altitude (competition host correction).

All column names follow gnss_lib_py standard after loading.
We also keep our own column aliases for backward compatibility with
hatch_filter.py and feature_builder.py.
"""

import os
import numpy as np
import pandas as pd
from gnss_lib_py.parsers.google_decimeter import (
    AndroidDerived2021,
    AndroidGroundTruth2021,
)


# ── column mapping: gnss_lib_py standard → our names ─────────────────────────
# (used to rename back for compatibility with downstream code)
GNSS_TO_OURS = {
    "gps_millis":        "millisSinceGpsEpoch",
    "x_sv_m":            "xSatPosM",
    "y_sv_m":            "ySatPosM",
    "z_sv_m":            "zSatPosM",
    "vx_sv_mps":         "xSatVelMps",
    "vy_sv_mps":         "ySatVelMps",
    "vz_sv_mps":         "zSatVelMps",
    "b_sv_m":            "satClkBiasM",
    "b_dot_sv_mps":      "satClkDriftMps",
    "raw_pr_m":          "rawPrM",
    "raw_pr_sigma_m":    "rawPrUncM",
    "intersignal_bias_m":"isrbM",
    "iono_delay_m":      "ionoDelayM",
    "tropo_delay_m":     "tropoDelayM",
    "corr_pr_m":         "correctedPrM",   # already computed by postprocess()
    "gnss_id":           "constellationType",
    "sv_id":             "svid",
    "signal_type":       "signalType",
    "rx_name":           "phoneName",
    "trace_name":        "collectionName",
}


def load_derived(derived_csv_path: str) -> pd.DataFrame:
    """
    Load *_derived.csv using AndroidDerived2021 (all corrections applied).

    Returns a DataFrame with BOTH gnss_lib_py column names AND our
    original column names (as aliases) for downstream compatibility.
    """
    # AndroidDerived2021 applies Correction 1, Correction 5, and
    # computes corr_pr_m automatically via postprocess()
    navdata = AndroidDerived2021(derived_csv_path)

    # convert NavData → pandas DataFrame
    df = navdata.pandas_df()

    # rename gnss_lib_py columns back to our naming convention
    df = df.rename(columns=GNSS_TO_OURS)

    # keep correctedPrM as the main pseudorange column
    # (already computed: raw_pr_m + b_sv_m - isrb - iono - tropo)
    return df


def load_ground_truth(ground_truth_path: str) -> pd.DataFrame:
    """
    Load ground_truth.csv using AndroidGroundTruth2021.

    Applies the -61 m altitude correction and converts to ECEF.
    Returns DataFrame with columns: millisSinceGpsEpoch, latDeg,
    lngDeg, heightAboveWgs84EllipsoidM, xEcefM, yEcefM, zEcefM.
    """
    gt_navdata = AndroidGroundTruth2021(ground_truth_path)
    gt_df      = gt_navdata.pandas_df()

    # rename to our convention
    gt_col_map = {
        "gps_millis":     "millisSinceGpsEpoch",
        "lat_rx_gt_deg":  "latDeg",
        "lon_rx_gt_deg":  "lngDeg",
        "alt_rx_gt_m":    "heightAboveWgs84EllipsoidM",
        "x_rx_gt_m":      "xEcefM",
        "y_rx_gt_m":      "yEcefM",
        "z_rx_gt_m":      "zEcefM",
    }
    gt_df = gt_df.rename(columns={k: v for k, v in gt_col_map.items()
                                   if k in gt_df.columns})
    return gt_df


def parse_gnss_log(gnss_log_path: str) -> pd.DataFrame:
    """
    Parse *_GnssLog.txt to extract Raw measurement rows.

    Extracts per-satellite per-epoch:
      Cn0DbHz, AccumulatedDeltaRangeMeters,
      AccumulatedDeltaRangeState, PseudorangeRateMetersPerSecond,
      CarrierFrequencyHz
    """
    rows   = []
    header = None

    with open(gnss_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Raw,"):
                header = line.lstrip("# ").split(",")[1:]   # drop "Raw"
            elif line.startswith("Raw,") and header is not None:
                parts = line.split(",")[1:]                  # drop "Raw"
                if len(parts) == len(header):
                    rows.append(parts)

    if not rows or header is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)

    numeric_cols = [
        "TimeNanos", "FullBiasNanos",
        "Cn0DbHz",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "AccumulatedDeltaRangeMeters",
        "AccumulatedDeltaRangeUncertaintyMeters",
        "AccumulatedDeltaRangeState",
        "CarrierFrequencyHz",
        "Svid", "ConstellationType",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FullBiasNanos" in df.columns and "TimeNanos" in df.columns:
        df["millisSinceGpsEpoch"] = (
            (df["TimeNanos"] - df["FullBiasNanos"]) / 1e6
        ).round().astype("int64")

    return df


def merge_derived_and_log(derived_df: pd.DataFrame,
                           log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge derived.csv columns with GnssLog Raw columns on
    (millisSinceGpsEpoch, svid, constellationType).
    Snaps timestamps to nearest second for matching.
    """
    if log_df.empty:
        return derived_df

    log_cols = [
        "millisSinceGpsEpoch", "Svid", "ConstellationType",
        "Cn0DbHz",
        "AccumulatedDeltaRangeMeters",
        "AccumulatedDeltaRangeUncertaintyMeters",
        "AccumulatedDeltaRangeState",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "CarrierFrequencyHz",
    ]
    log_cols = [c for c in log_cols if c in log_df.columns]

    # gnss_lib_py CONSTELLATION_ANDROID maps integers to lowercase strings
    CONST_INT_TO_STR = {
        1: "gps", 2: "sbas", 3: "glonass",
        4: "qzss", 5: "beidou", 6: "galileo", 7: "irnss"
    }
    log_sub = log_df[log_cols].copy().rename(
        columns={"Svid": "svid", "ConstellationType": "constellationType"}
    )
    # convert integer constellation type → string to match derived_df
    log_sub["constellationType"] = log_sub["constellationType"].map(
        CONST_INT_TO_STR
    ).fillna(log_sub["constellationType"].astype(str))
    



    derived_df["epochSnap"] = (derived_df["millisSinceGpsEpoch"] / 1000).round() * 1000
    log_sub["epochSnap"]    = (log_sub["millisSinceGpsEpoch"] / 1000).round() * 1000

    merged = pd.merge(
        derived_df,
        log_sub.drop(columns=["millisSinceGpsEpoch"]),
        on=["epochSnap", "svid", "constellationType"],
        how="left",
    )
    return merged.drop(columns=["epochSnap"])


def load_phone_data(phone_dir: str) -> dict:
    """
    Main entry point — load all data for one phone directory.

    Parameters
    ----------
    phone_dir : str
        e.g. data/raw/train/2021-04-02-US-SJC-1/Pixel4

    Returns
    -------
    dict:
        'derived'      → merged DataFrame (all corrections applied)
        'ground_truth' → ground truth DataFrame with ECEF columns
        'phone_name'   → str
        'drive_id'     → str
    """
    phone_name = os.path.basename(phone_dir)
    drive_id   = os.path.basename(os.path.dirname(phone_dir))

    derived_path = os.path.join(phone_dir, f"{phone_name}_derived.csv")
    log_path     = os.path.join(phone_dir, f"{phone_name}_GnssLog.txt")
    gt_path      = os.path.join(phone_dir, "ground_truth.csv")

    # ── load derived (with all corrections) ───────────────────────────────────
    derived_df = load_derived(derived_path)

    # ── merge GnssLog for ADR, C/N0, Doppler ──────────────────────────────────
    if os.path.exists(log_path):
        log_df = parse_gnss_log(log_path)
        merged = merge_derived_and_log(derived_df, log_df)
    else:
        print(f"  [WARNING] No GnssLog for {phone_name}, using derived only.")
        merged = derived_df

    # ── load ground truth ─────────────────────────────────────────────────────
    gt_df = load_ground_truth(gt_path)

    return {
        "derived":      merged,
        "ground_truth": gt_df,
        "phone_name":   phone_name,
        "drive_id":     drive_id,
    }


def get_epochs(derived_df: pd.DataFrame) -> np.ndarray:
    return np.sort(derived_df["millisSinceGpsEpoch"].unique())


def get_epoch_data(derived_df: pd.DataFrame, epoch_ms: int) -> pd.DataFrame:
    return derived_df[derived_df["millisSinceGpsEpoch"] == epoch_ms].copy()


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    phone_dir = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"
    if len(sys.argv) > 1:
        phone_dir = sys.argv[1]

    print(f"Loading: {phone_dir}")
    data = load_phone_data(phone_dir)

    derived = data["derived"]
    gt      = data["ground_truth"]

    print(f"\nDerived shape     : {derived.shape}")
    print(f"Ground truth shape: {gt.shape}")
    print(f"Epochs            : {derived['millisSinceGpsEpoch'].nunique()}")
    print(f"Satellites        : {derived['svid'].nunique()}")
    print(f"\nColumns:\n{derived.columns.tolist()}")

    if "Cn0DbHz" in derived.columns:
        non_null = derived["Cn0DbHz"].notna().sum()
        print(f"\nCn0DbHz non-null  : {non_null} / {len(derived)}")

    print(f"\nGround truth ECEF (first row):")
    print(f"  x={gt['xEcefM'].iloc[0]:.1f}  "
          f"y={gt['yEcefM'].iloc[0]:.1f}  "
          f"z={gt['zEcefM'].iloc[0]:.1f}")

    print(f"\nFirst correctedPrM values (first epoch):")
    first_ep = derived[derived["millisSinceGpsEpoch"] ==
                       derived["millisSinceGpsEpoch"].iloc[0]]
    print(first_ep[["svid", "signalType",
                    "correctedPrM", "rawPrUncM"]].head(5).to_string(index=False))
