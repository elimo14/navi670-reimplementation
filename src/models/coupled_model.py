"""
coupled_model.py
----------------
Tightly coupled GNN + Learned BKF for GNSS smartphone positioning.

This is the core contribution of navi_670 (Figure 1):

  "Our framework integrates a GNN and BKF to estimate a receiver's
   position with high accuracy. The GNN outputs a measurement vector
   (position correction) which is fed as the observation z_t into the
   BKF. The BKF refines this using temporal state estimates. Gradients
   flow back through BOTH the GNN and BKF parameters jointly."

Forward pass (Figure 1):
  1. GNN forward  → correction estimate z_t (measurement vector)
  2. BKF predict  → x_{t|t-1}, P_{t|t-1}
  3. BKF update   → x_{t|t} using z_t as observation
  4. Output       → final_pos = rx_pos_wls + x_{t|t}
  5. Loss         → MSE(final_pos, gt_pos)
  6. Backward     → updates GNN weights + BKF Q, R jointly

Unique backpropagation strategy (Section 2.2):
  "We implement a unique backpropagation strategy that uses real-time
   positioning corrections to refine the performance of both the GNN
   and the learned Kalman filter."
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from src.models.gnn import GNSSGraphSAGE
from src.models.bkf import LearnedBKF


class TightlyCoupledGNNBKF(nn.Module):
    """
    Tightly coupled GNN + BKF model (Figure 1 of navi_670).

    Parameters
    ----------
    in_dim      : int   GNN input feature dim (default 6)
    hidden_dim  : int   GNN hidden dim (default 128, Table 4)
    n_layers    : int   GNN SAGEConv layers (default 8, Table 4)
    state_dim   : int   BKF state dim (default 3, Table 3)
    init_Q_scale: float initial BKF process noise (default 1e-3, Table 3)
    init_R_scale: float initial BKF measurement noise (default 1e-2, Table 3)
    """

    def __init__(self,
                 in_dim:      int   = 6,
                 hidden_dim:  int   = 128,
                 n_layers:    int   = 8,
                 state_dim:   int   = 3,
                 init_Q_scale:float = 1e-3,
                 init_R_scale:float = 1e-2):
        super().__init__()

        # ── GNN module ────────────────────────────────────────────────────────
        self.gnn = GNSSGraphSAGE(
            in_dim     = in_dim,
            hidden_dim = hidden_dim,
            out_dim    = state_dim,
            n_layers   = n_layers,
        )

        # ── Learned BKF module ────────────────────────────────────────────────
        self.bkf = LearnedBKF(
            state_dim    = state_dim,
            init_Q_scale = init_Q_scale,
            init_R_scale = init_R_scale,
        )

        self.state_dim = state_dim

    def reset_filter(self,
                     x0: Optional[torch.Tensor] = None,
                     P0: Optional[torch.Tensor] = None):
        """Reset BKF state at start of new drive/sequence."""
        self.bkf.reset(x0, P0)

    def forward_single(self,
                       x:          torch.Tensor,
                       edge_index: torch.Tensor,
                       rx_pos_wls: torch.Tensor,
                       batch:      Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE epoch.

        Parameters
        ----------
        x          : (N, 6)  node features
        edge_index : (2, E)  edge connectivity
        rx_pos_wls : (3,)    WLS receiver position in ECEF (m)
        batch      : (N,)    batch vector (None for single graph)

        Returns
        -------
        final_pos     : (3,) estimated receiver position in ECEF
        gnn_correction: (3,) GNN-predicted correction (before BKF)
        """
        # ── Step 1: GNN forward → measurement vector z_t ─────────────────────
        # GNN predicts the position CORRECTION (not absolute position)
        gnn_correction = self.gnn(x, edge_index, batch)  # (1, 3) or (B, 3)
        gnn_correction = gnn_correction.squeeze(0)        # (3,)

        # ── Step 2-3: BKF predict + update ───────────────────────────────────
        # z_t = GNN correction is used as the observation
        bkf_correction = self.bkf.step(gnn_correction)   # (3,)

        # ── Step 4: final position ────────────────────────────────────────────
        # final_pos = WLS_position + BKF_filtered_correction
        rx_pos = rx_pos_wls.float()
        final_pos = rx_pos + bkf_correction               # (3,)

        return final_pos, gnn_correction

    def forward(self,
                data_sequence: List,
                reset_filter:  bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over a SEQUENCE of epochs (one drive).

        This enables temporal smoothing via the BKF across time steps.

        Parameters
        ----------
        data_sequence : list of PyG Data objects (one per epoch, in order)
        reset_filter  : bool  reset BKF at start of sequence

        Returns
        -------
        final_positions : (T, 3) estimated positions per epoch
        gnn_corrections : (T, 3) raw GNN corrections per epoch
        """
        if reset_filter:
            self.reset_filter()

        final_positions = []
        gnn_corrections = []

        for data in data_sequence:
            x          = data.x
            edge_index = data.edge_index
            rx_pos_wls = data.rx_pos_wls.float()

            # handle batch dimension
            batch = data.batch if hasattr(data, "batch") and \
                    data.batch is not None else None

            final_pos, gnn_corr = self.forward_single(
                x, edge_index, rx_pos_wls, batch
            )

            final_positions.append(final_pos.unsqueeze(0))
            gnn_corrections.append(gnn_corr.unsqueeze(0))

        final_positions = torch.cat(final_positions, dim=0)  # (T, 3)
        gnn_corrections = torch.cat(gnn_corrections, dim=0)  # (T, 3)

        return final_positions, gnn_corrections

    def count_parameters(self) -> dict:
        """Return parameter counts per component."""
        gnn_params = sum(p.numel() for p in self.gnn.parameters()
                         if p.requires_grad)
        bkf_params = sum(p.numel() for p in self.bkf.parameters()
                         if p.requires_grad)
        return {
            "gnn":   gnn_params,
            "bkf":   bkf_params,
            "total": gnn_params + bkf_params,
        }


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, numpy as np
    sys.path.insert(0, ".")
    from src.graph.dataset import GNSSGraphDataset

    phone_dir = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"

    print("Loading dataset (first 20 epochs for speed)...")
    dataset = GNSSGraphDataset([phone_dir])
    sequence = [dataset[i] for i in range(20)]

    print(f"\nBuilding TightlyCoupledGNNBKF (navi_670 Figure 1)...")
    model = TightlyCoupledGNNBKF(
        in_dim       = 6,
        hidden_dim   = 128,
        n_layers     = 8,
        state_dim    = 3,
        init_Q_scale = 1e-3,
        init_R_scale = 1e-2,
    )

    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  GNN   : {params['gnn']:,}")
    print(f"  BKF   : {params['bkf']}")
    print(f"  Total : {params['total']:,}")

    # ── single epoch forward ──────────────────────────────────────────────────
    print(f"\nSingle epoch forward pass:")
    model.reset_filter()
    d0 = sequence[0]
    model.eval()
    with torch.no_grad():
        pos, corr = model.forward_single(
            d0.x, d0.edge_index, d0.rx_pos_wls
        )
    print(f"  WLS position  : {d0.rx_pos_wls.numpy().round(1)}")
    print(f"  GNN correction: {corr.numpy().round(3)}")
    print(f"  Final position: {pos.numpy().round(1)}")
    if d0.has_gt.item():
        gt  = d0.gt_pos.numpy()
        err = np.linalg.norm(pos.numpy() - gt)
        print(f"  GT  position  : {gt.round(1)}")
        print(f"  3D error      : {err:.2f} m  (random init)")

    # ── sequence forward ──────────────────────────────────────────────────────
    print(f"\nSequence forward pass (20 epochs):")
    model.eval()
    with torch.no_grad():
        positions, corrections = model(sequence, reset_filter=True)

    print(f"  Output positions shape  : {positions.shape}")
    print(f"  Output corrections shape: {corrections.shape}")

    # ── backpropagation test ──────────────────────────────────────────────────
    print(f"\nBackpropagation test (joint GNN + BKF):")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    positions, _ = model(sequence[:5], reset_filter=True)
    gt_positions = torch.stack([s.gt_pos.float() for s in sequence[:5]])

    loss = ((positions - gt_positions) ** 2).mean()
    loss.backward()

    # check gradients flow to both GNN and BKF
    gnn_grad = model.gnn.input_proj.weight.grad
    bkf_grad = model.bkf.log_Q_diag.grad

    print(f"  Loss            : {loss.item():.4f}")
    print(f"  GNN grad (input): {gnn_grad.norm().item():.6f}  "
          f"{'✅' if gnn_grad is not None else '❌'}")
    print(f"  BKF grad (log_Q): {bkf_grad.numpy().round(8)}"
          f"  {'✅' if bkf_grad is not None else '❌'}")
    print(f"\n✅ Joint backpropagation works — GNN and BKF update together")
    print(f"✅ Coupled model ready for training loop (Phase 3 Step 4)")
