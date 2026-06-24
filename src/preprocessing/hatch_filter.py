"""
hatch_filter.py
---------------
Implements carrier-phase smoothing (Hatch filter) as described in
Section 2.1 of the base paper (navi_670):

  "We reduce the noise in pseudorange measurements by using carrier phase
   smoothing with a Hatch filter after accounting for various error sources.
   We use a window of two measurements from the AccumulatedDeltaRange (ADR)
   field. In the event of a cycle slip, we use Doppler values to smooth the
   code phase measurements."

The Hatch filter formula (window size N):
  smoothed(t) = (1/N) * rawPr(t) + ((N-1)/N) * (smoothed(t-1) + ADR(t) - ADR(t-1))

Cycle slip detection (Android bitmask on AccumulatedDeltaRangeState):
  Bit 0 (value 1) = ADR_STATE_VALID   → must be SET for ADR to be usable
  Bit 1 (value 2) = ADR_STATE_RESET   → if SET, a reset occurred
  Bit 2 (value 4) = ADR_STATE_CYCLE_SLIP → if SET, cycle slip detected

If cycle slip is detected → fall back to Doppler-based smoothing.
"""

import numpy as np
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0   # m/s
HATCH_WINDOW   = 2               # paper uses window of 2 (Section 2.1)

# ADR state bitmask values
ADR_STATE_VALID      = 1   # bit 0
ADR_STATE_RESET      = 2   # bit 1
ADR_STATE_CYCLE_SLIP = 4   # bit 2


def _is_cycle_slip(adr_state: float) -> bool:
    """
    Return True if the ADR measurement should NOT be trusted.

    A measurement is invalid when:
      - VALID bit is NOT set, OR
      - RESET bit is set, OR
      - CYCLE_SLIP bit is set
    """
    if np.isnan(adr_state):
        return True
    state = int(adr_state)
    valid      = bool(state & ADR_STATE_VALID)
    reset      = bool(state & ADR_STATE_RESET)
    cycle_slip = bool(state & ADR_STATE_CYCLE_SLIP)
    return (not valid) or reset or cycle_slip


def _doppler_smoothed_pr(prev_smoothed: float,
                          doppler_mps: float,
                          dt_s: float) -> float:
    """
    Doppler-based pseudorange prediction when ADR is unavailable.

    smoothed(t) ≈ smoothed(t-1) + Doppler * dt
    (Doppler is the pseudorange rate in m/s)
    """
    if np.isnan(doppler_mps) or np.isnan(dt_s) or dt_s <= 0:
        return prev_smoothed
    return prev_smoothed + doppler_mps * dt_s


def apply_hatch_filter(epoch_group: pd.DataFrame,
                        prev_state: dict,
                        prev_epoch_ms: float,
                        window: int = HATCH_WINDOW) -> tuple:
    """
    Apply one step of the Hatch filter to all satellites in a single epoch.

    Parameters
    ----------
    epoch_group   : DataFrame — all satellite rows for the current epoch
    prev_state    : dict mapping (svid, signalType) → (prev_smoothed, prev_adr)
    prev_epoch_ms : float — timestamp of the previous epoch in milliseconds
    window        : int — smoothing window (N=2 per the paper)

    Returns
    -------
    epoch_group   : DataFrame with added column 'smoothedPrM'
    new_state     : dict — updated state for next epoch
    """
    smoothed_prs = []
    new_state    = {}

    # time delta between consecutive epochs (seconds)
    current_epoch_ms = epoch_group["millisSinceGpsEpoch"].iloc[0]
    dt_s = (current_epoch_ms - prev_epoch_ms) / 1000.0 if prev_epoch_ms is not None else 0.0

    for _, row in epoch_group.iterrows():
        key = (row["svid"], row["signalType"])

        raw_pr   = row["correctedPrM"]           # corrected code-phase pseudorange
        adr      = row.get("AccumulatedDeltaRangeMeters", np.nan)
        adr_state= row.get("AccumulatedDeltaRangeState",  np.nan)
        doppler  = row.get("PseudorangeRateMetersPerSecond", np.nan)

        # ── first appearance of this satellite → initialise ───────────────────
        if key not in prev_state:
            smoothed = raw_pr
            new_state[key] = (smoothed, adr)
            smoothed_prs.append(smoothed)
            continue

        prev_smoothed, prev_adr = prev_state[key]

        # ── detect cycle slip ─────────────────────────────────────────────────
        slip = _is_cycle_slip(adr_state)

        if slip or np.isnan(adr) or np.isnan(prev_adr):
            # ── Doppler fallback ──────────────────────────────────────────────
            smoothed = _doppler_smoothed_pr(prev_smoothed, doppler, dt_s)
            # reset ADR tracking after slip
            new_state[key] = (smoothed, adr)
        else:
            # ── standard Hatch update ─────────────────────────────────────────
            adr_delta = adr - prev_adr          # carrier phase change (m)
            alpha     = 1.0 / window
            smoothed  = alpha * raw_pr + (1.0 - alpha) * (prev_smoothed + adr_delta)
            new_state[key] = (smoothed, adr)

        smoothed_prs.append(smoothed)

    epoch_group = epoch_group.copy()
    epoch_group["smoothedPrM"] = smoothed_prs
    return epoch_group, new_state


def smooth_all_epochs(derived_df: pd.DataFrame,
                       window: int = HATCH_WINDOW) -> pd.DataFrame:
    """
    Apply the Hatch filter across ALL epochs for one phone/drive.

    Processes epochs in chronological order, carrying filter state forward.

    Parameters
    ----------
    derived_df : merged DataFrame from parser.load_phone_data()
    window     : smoothing window size (default 2 per paper)

    Returns
    -------
    DataFrame with added column 'smoothedPrM'
    """
    epochs     = np.sort(derived_df["millisSinceGpsEpoch"].unique())
    prev_state = {}        # (svid, signalType) → (smoothed_pr, adr)
    prev_epoch = None
    result_chunks = []

    for epoch_ms in epochs:
        epoch_data = derived_df[derived_df["millisSinceGpsEpoch"] == epoch_ms].copy()
        smoothed_epoch, prev_state = apply_hatch_filter(
            epoch_data, prev_state, prev_epoch, window=window
        )
        result_chunks.append(smoothed_epoch)
        prev_epoch = epoch_ms

    return pd.concat(result_chunks, ignore_index=True)


def select_satellites(df: pd.DataFrame,
                       min_cn0: float = 20.0,
                       min_elevation_deg: float = 10.0) -> pd.DataFrame:
    """
    Filter out unreliable satellites as described in Section 2.1 of the paper:
      "We first select satellites by thresholding the carrier frequency of the
       tracked signal and the elevation angle, thus eliminating unreliable signals."

    Filters applied:
      1. C/N0 >= min_cn0        (signal quality)
      2. Elevation >= min_elevation_deg  (if elevation available)
      3. Drop rows where smoothedPrM is NaN or clearly out of range

    Parameters
    ----------
    df              : DataFrame with 'Cn0DbHz', optionally 'elevationDeg'
    min_cn0         : minimum acceptable C/N0 in dB-Hz (typical: 20-25)
    min_elevation_deg: minimum satellite elevation in degrees

    Returns
    -------
    Filtered DataFrame
    """
    mask = pd.Series(True, index=df.index)

    # ── C/N0 filter ───────────────────────────────────────────────────────────
    if "Cn0DbHz" in df.columns:
        mask &= df["Cn0DbHz"].fillna(0) >= min_cn0

    # ── elevation filter (if column exists) ───────────────────────────────────
    if "elevationDeg" in df.columns:
        mask &= df["elevationDeg"].fillna(0) >= min_elevation_deg

    # ── remove NaN pseudoranges ───────────────────────────────────────────────
    if "smoothedPrM" in df.columns:
        mask &= df["smoothedPrM"].notna()
    else:
        mask &= df["correctedPrM"].notna()

    # ── sanity range check: pseudoranges should be ~20,000 km ─────────────────
    pr_col = "smoothedPrM" if "smoothedPrM" in df.columns else "correctedPrM"
    mask &= df[pr_col].between(1e6, 1e8)   # 1,000 km – 100,000 km

    return df[mask].copy()


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing.parser import load_phone_data, get_epochs

    phone_dir = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"

    print("Loading data...")
    data    = load_phone_data(phone_dir)
    derived = data["derived"]

    print(f"Total rows before smoothing : {len(derived)}")

    print("\nApplying Hatch filter...")
    smoothed_df = smooth_all_epochs(derived, window=HATCH_WINDOW)

    print(f"Total rows after  smoothing : {len(smoothed_df)}")
    print(f"NaN in smoothedPrM          : {smoothed_df['smoothedPrM'].isna().sum()}")

    # compare raw vs smoothed on first few satellites
    first_epoch = get_epochs(smoothed_df)[0]
    ep = smoothed_df[smoothed_df["millisSinceGpsEpoch"] == first_epoch]
    print("\nFirst epoch — raw vs smoothed pseudorange (m):")
    print(ep[["svid", "signalType", "correctedPrM", "smoothedPrM",
              "Cn0DbHz"]].to_string(index=False))

    # apply satellite selection
    print("\nApplying satellite selection filter...")
    filtered = select_satellites(smoothed_df)
    print(f"Rows after selection : {len(filtered)} / {len(smoothed_df)}")
    print(f"Epochs remaining     : {filtered['millisSinceGpsEpoch'].nunique()}")