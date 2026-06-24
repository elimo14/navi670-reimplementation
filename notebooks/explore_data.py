"""
src/preprocessing/
    parser.py          ← Step 2.1: load & merge derived.csv + GnssLog.txt
    hatch_filter.py    ← Step 2.2: carrier smoothing
    wls.py             ← Step 2.2: weighted least squares initial position
    feature_builder.py ← Step 2.3: build 6D node feature vectors
src/graph/
    edge_builder.py    ← Step 2.4: constellation + cosine similarity edges
    dataset.py         ← Step 2.4: PyG Data objects per epoch
"""


import pandas as pd
import os

# ── adjust this to one of your actual train folders ──────────────────────────
BASE = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"

derived_path      = os.path.join(BASE, "Pixel4_derived.csv")
ground_truth_path = os.path.join(BASE, "ground_truth.csv")

# ── load ─────────────────────────────────────────────────────────────────────
derived      = pd.read_csv(derived_path)
ground_truth = pd.read_csv(ground_truth_path)

# ── basic info ────────────────────────────────────────────────────────────────
print("=" * 60)
print("DERIVED CSV")
print("=" * 60)
print(f"Shape : {derived.shape}  ({derived.shape[0]} rows, {derived.shape[1]} cols)")
print(f"\nColumns:\n{derived.columns.tolist()}")
print(f"\nConstellations present : {derived['constellationType'].unique()}")
print(f"Signal types present   : {derived['signalType'].unique()}")
print(f"Unique epochs          : {derived['millisSinceGpsEpoch'].nunique()}")
print(f"Unique satellites      : {derived['svid'].nunique()}")

print("\nKey columns sample (first epoch):")
first_epoch = derived['millisSinceGpsEpoch'].iloc[0]
ep = derived[derived['millisSinceGpsEpoch'] == first_epoch]
print(ep[['svid','signalType','rawPrM','satClkBiasM',
          'ionoDelayM','tropoDelayM','isrbM',
          'rawPrUncM','xSatPosM','ySatPosM','zSatPosM']].to_string())

print("\n" + "=" * 60)
print("GROUND TRUTH CSV")
print("=" * 60)
print(f"Shape : {ground_truth.shape}")
print(f"\nColumns:\n{ground_truth.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(ground_truth[['millisSinceGpsEpoch','latDeg','lngDeg',
                     'heightAboveWgs84EllipsoidM']].head(3).to_string())

print("\n" + "=" * 60)
print("CORRECTED PSEUDORANGE CHECK (first epoch, first satellite)")
print("=" * 60)
row = ep.iloc[0]
corrPr = row['rawPrM'] + row['satClkBiasM'] - row['isrbM'] \
       - row['ionoDelayM'] - row['tropoDelayM']
print(f"rawPrM        : {row['rawPrM']:.3f} m")
print(f"satClkBiasM   : {row['satClkBiasM']:.3f} m")
print(f"isrbM         : {row['isrbM']:.3f} m")
print(f"ionoDelayM    : {row['ionoDelayM']:.3f} m")
print(f"tropoDelayM   : {row['tropoDelayM']:.3f} m")
print(f"correctedPrM  : {corrPr:.3f} m  ← this is what we use")
print(f"rawPrUncM     : {row['rawPrUncM']:.3f} m")