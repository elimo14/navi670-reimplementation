"""
dataset.py
----------
PyTorch Geometric Dataset that wraps one phone/drive into
a list of Data objects — one per epoch.

Each Data object contains:
  data.x          : (n, 6) float32  node feature matrix
  data.edge_index : (2, E) int64    edge connectivity
  data.sat_pos    : (n, 3) float64  satellite ECEF positions
  data.rx_pos_wls : (3,)  float64  WLS receiver position
  data.cdt        : float           WLS clock bias (m)
  data.y          : (3,)  float32  true correction (gt_pos - rx_pos_wls)
  data.gt_pos     : (3,)  float64  ground truth ECEF position
  data.epoch_ms   : int             epoch timestamp

This format plugs directly into PyTorch Geometric's DataLoader
for batched training.

Reference: navi_670 Section 2, Figure 1
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional

sys.path.insert(0, ".")
from src.preprocessing.parser          import load_phone_data
from src.preprocessing.wls            import compute_position_from_path
from src.preprocessing.feature_builder import build_all_epoch_features
from src.graph.edge_builder            import build_edge_index_fast, DEFAULT_SIMILARITY_THRESHOLD


class GNSSGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for GNSS smartphone positioning.

    Each item is one epoch's satellite graph with:
      - Node features (6D per satellite)
      - Edge connectivity (constellation + cosine similarity)
      - Target correction (ground truth - WLS position)

    Parameters
    ----------
    phone_dirs  : list of phone directory paths
                  e.g. ['data/raw/train/2021-04-02-US-SJC-1/Pixel4', ...]
    threshold   : float cosine similarity threshold for edges (default 0.5)
    transform   : optional PyG transform
    pre_transform: optional PyG pre-transform
    """

    def __init__(self,
                 phone_dirs:   List[str],
                 threshold:    float = DEFAULT_SIMILARITY_THRESHOLD,
                 transform=None,
                 pre_transform=None):
        super().__init__(root=None,
                         transform=transform,
                         pre_transform=pre_transform)
        self.threshold  = threshold
        self._data_list = []
        self._phone_count = 0

        for phone_dir in phone_dirs:
            self._load_phone(phone_dir)

        print(f"Dataset ready: {len(self._data_list)} graphs "
              f"from {len(phone_dirs)} phone(s)")

    def _load_phone(self, phone_dir: str):
        """Load one phone/drive and add its epochs to _data_list."""
        phone_name   = os.path.basename(phone_dir)
        derived_path = os.path.join(phone_dir, f"{phone_name}_derived.csv")
        gt_path      = os.path.join(phone_dir, "ground_truth.csv")

        if not os.path.exists(derived_path):
            print(f"  [SKIP] {phone_name}: derived CSV not found")
            return

        print(f"  Loading {phone_name}...")

        # ── WLS positions ─────────────────────────────────────────────────────
        pos_df, gt_df = compute_position_from_path(derived_path, gt_path)

        # ── merged data (for C/N0 from GnssLog) ──────────────────────────────
        data_dict = load_phone_data(phone_dir)
        merged_df = data_dict["derived"]

        # ── build features ────────────────────────────────────────────────────
        epoch_features = build_all_epoch_features(merged_df, pos_df, gt_df)

        # ── constellation types per epoch ─────────────────────────────────────
        const_col = ("constellationType" if "constellationType" in merged_df.columns
                     else "gnss_id")
        svid_col  = "svid" if "svid" in merged_df.columns else "sv_id"


        phone_id = self._phone_count    # ← ADD THIS
        self._phone_count += 1          # ← ADD THIS

        skipped = 0

        for feat in epoch_features:
            data_obj = self._feat_to_data(
                feat, merged_df, const_col, svid_col, phone_id   # ← ADD phone_id
            )
            if data_obj is not None:
                self._data_list.append(data_obj)
            else:
                skipped += 1


        print(f"    -> {len(epoch_features) - skipped} graphs "
              f"({skipped} skipped)")

    def _feat_to_data(self,
                      feat:      dict,
                      merged_df: object,
                      const_col: str,
                      svid_col:  str,
                      phone_id:  int = 0
                      ) -> Optional[Data]:
        """Convert one epoch feature dict -> PyG Data object."""
        epoch_ms = feat["epoch_ms"]
        X        = feat["X"]           # (n, 6)
        n        = X.shape[0]

        if n < 4:
            return None

        # ── get constellation types for this epoch ────────────────────────────
        ep_df = (merged_df[merged_df["millisSinceGpsEpoch"] == epoch_ms]
                 .sort_values("Cn0DbHz", ascending=False)
                 .drop_duplicates(subset=[svid_col, const_col], keep="first")
                 .head(20))

        if len(ep_df) != n:
            # mismatch — use fallback all-same constellation
            const_types = np.zeros(n, dtype=object)
        else:
            const_types = ep_df[const_col].values

        # ── build edges ───────────────────────────────────────────────────────
        edge_index = build_edge_index_fast(X, const_types, self.threshold)

        # ── correction target ─────────────────────────────────────────────────
        correction = feat.get("correction", np.full(3, np.nan))
        has_gt     = not np.isnan(correction).any()

        # ── build PyG Data object ─────────────────────────────────────────────
        data_obj = Data(
            x          = torch.tensor(X, dtype=torch.float32),
            edge_index = edge_index,
            sat_pos    = torch.tensor(feat["sat_pos"],    dtype=torch.float64),
            rx_pos_wls = torch.tensor(feat["rx_pos_wls"], dtype=torch.float64),
            cdt        = torch.tensor([feat["cdt"]],      dtype=torch.float64),
            y          = torch.tensor(correction,          dtype=torch.float32)
                         if has_gt else torch.zeros(3, dtype=torch.float32),
            gt_pos     = torch.tensor(feat.get("gt_pos", np.full(3, np.nan)),
                                      dtype=torch.float64),
            epoch_ms   = torch.tensor([epoch_ms], dtype=torch.long),
            has_gt     = torch.tensor([has_gt],   dtype=torch.bool),
        )
        data_obj.phone_id = torch.tensor([phone_id], dtype=torch.long)
        return data_obj

    # ── PyG Dataset interface ─────────────────────────────────────────────────
    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]


# ── standalone builder (faster than Dataset for batch processing) ─────────────

def build_dataset_from_dirs(phone_dirs: List[str],
                             threshold:  float = DEFAULT_SIMILARITY_THRESHOLD
                             ) -> List[Data]:
    """
    Build list of PyG Data objects from multiple phone directories.
    Equivalent to GNSSGraphDataset but returns a plain list.
    """
    ds = GNSSGraphDataset(phone_dirs, threshold=threshold)
    return ds._data_list


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    phone_dir = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"

    print("=" * 55)
    print("Building GNSSGraphDataset...")
    print("=" * 55)

    dataset = GNSSGraphDataset(
        phone_dirs=[phone_dir],
        threshold=DEFAULT_SIMILARITY_THRESHOLD,
    )

    print(f"\nDataset size : {len(dataset)} graphs")

    # ── inspect first graph ───────────────────────────────────────────────────
    d0 = dataset[0]
    print(f"\nFirst graph:")
    print(f"  x (node features)  : {d0.x.shape}    dtype={d0.x.dtype}")
    print(f"  edge_index         : {d0.edge_index.shape}  dtype={d0.edge_index.dtype}")
    print(f"  sat_pos            : {d0.sat_pos.shape}    dtype={d0.sat_pos.dtype}")
    print(f"  rx_pos_wls (ECEF)  : {d0.rx_pos_wls.numpy().round(1)}")
    print(f"  cdt (clock bias)   : {d0.cdt.item():.2f} m")
    print(f"  y (correction)     : {d0.y.numpy().round(2)}")
    print(f"  |y| (correction)   : {d0.y.norm().item():.2f} m")
    print(f"  has_gt             : {d0.has_gt.item()}")
    print(f"  epoch_ms           : {d0.epoch_ms.item()}")

    # ── feature ranges ────────────────────────────────────────────────────────
    print(f"\nFeature column stats (first graph):")
    names = ["LOS_x", "LOS_y", "LOS_z", "Residual", "C/N0", "Uncertainty"]
    for i, name in enumerate(names):
        col = d0.x[:, i].numpy()
        print(f"  {name:<12}: mean={col.mean():8.2f}  "
              f"std={col.std():7.2f}  "
              f"[{col.min():.2f}, {col.max():.2f}]")

    # ── DataLoader test ───────────────────────────────────────────────────────
    print(f"\nDataLoader batch test (batch_size=4):")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch  = next(iter(loader))
    print(f"  batch.x.shape         : {batch.x.shape}")
    print(f"  batch.edge_index.shape: {batch.edge_index.shape}")
    print(f"  batch.y.shape         : {batch.y.shape}")
    print(f"  batch.batch.shape     : {batch.batch.shape}")
    print(f"  unique batch indices  : {batch.batch.unique().tolist()}")

    # ── dataset statistics ────────────────────────────────────────────────────
    print(f"\nDataset statistics over all {len(dataset)} graphs:")
    n_nodes  = [d.x.shape[0]          for d in dataset]
    n_edges  = [d.edge_index.shape[1]  for d in dataset]
    corr_norms = [d.y.norm().item()    for d in dataset if d.has_gt.item()]

    print(f"  Nodes/graph  : mean={np.mean(n_nodes):.1f}  "
          f"min={np.min(n_nodes)}  max={np.max(n_nodes)}")
    print(f"  Edges/graph  : mean={np.mean(n_edges):.1f}  "
          f"min={np.min(n_edges)}  max={np.max(n_edges)}")
    if corr_norms:
        print(f"  |correction| : mean={np.mean(corr_norms):.2f} m  "
              f"median={np.median(corr_norms):.2f} m  "
              f"(what GNN+BKF learns)")

    print(f"\n✅ Phase 2 complete — dataset ready for GNN training (Phase 3)")
