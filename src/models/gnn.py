"""
gnn.py
------
GraphSAGE-based GNN module for GNSS positioning corrections.

From Section 2.3 and Tables 4-5 of the base paper (navi_670):

  Architecture:
    Linear(6 → 128)          ← input projection
    [SAGEConv(128 → 128)
     BatchNorm1d(128)
     ReLU] × N_layers        ← N=8 graph convolution layers
    Mean pooling             ← aggregate all node embeddings
    Linear(128 → 3)          ← output: 3D position correction (measurement vector)

  Key design choices (Section 2.3.4):
    - GraphSAGE (Hamilton et al., 2017) for INDUCTIVE learning
    - Mean aggregator (Equation 11)
    - Inductive: generalizes to unseen graph structures / satellite counts

  Output of GNN is used as the MEASUREMENT VECTOR z_t fed into the BKF
  (Figure 1 and Section 2.3 of navi_670).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool


class GNSSGraphSAGE(nn.Module):
    """
    GraphSAGE GNN that produces a 3D position correction from satellite graphs.

    Parameters
    ----------
    in_dim      : int   input feature dimension (default 6 from paper)
    hidden_dim  : int   hidden dimension (default 128 from Table 4)
    out_dim     : int   output dimension (default 3 for 3D correction)
    n_layers    : int   number of SAGEConv layers (default 8 from Table 4)
    dropout     : float dropout probability (default 0.0)
    """

    def __init__(self,
                 in_dim:     int = 6,
                 hidden_dim: int = 128,
                 out_dim:    int = 3,
                 n_layers:   int = 8,
                 dropout:    float = 0.0):
        super().__init__()

        self.in_dim     = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.n_layers   = n_layers
        self.dropout    = dropout

        # ── input projection: Linear(6 → 128) ────────────────────────────────
        # "The architecture begins with a linear layer that transforms the
        #  feature space from 6 to 128 dimensions" (Table 5, navi_670)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ── GraphSAGE convolution layers ──────────────────────────────────────
        # "A sequence of multiple convolution layers within the GraphSAGE
        #  learning framework" (Section 2.3.4)
        self.convs      = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batchnorms.append(nn.BatchNorm1d(hidden_dim))

        # ── output layer: Linear(128 → 3) ─────────────────────────────────────
        # "An output linear layer condenses the 128-dimensional data back
        #  down to 3 dimensions" (Table 5, navi_670)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for linear layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self,
                x:          torch.Tensor,
                edge_index: torch.Tensor,
                batch:      torch.Tensor = None
                ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x          : (N, 6)     node feature matrix (all nodes in batch)
        edge_index : (2, E)     edge connectivity
        batch      : (N,)       maps each node to its graph index
                                None for single-graph inference

        Returns
        -------
        out : (B, 3)  position correction per graph in batch
              where B = number of graphs
        """
        # ── input projection ──────────────────────────────────────────────────
        h = F.relu(self.input_proj(x))           # (N, 128)

        # ── graph convolution layers ──────────────────────────────────────────
        # Each layer: SAGEConv → BatchNorm → ReLU
        for conv, bn in zip(self.convs, self.batchnorms):
            h = conv(h, edge_index)              # (N, 128) — message passing
            h = bn(h)                            # (N, 128) — normalise
            h = F.relu(h)                        # (N, 128) — activate
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout)

        # ── mean pooling: aggregate all node embeddings ───────────────────────
        # "The node embeddings are aggregated by a mean aggregation method"
        # (Figure 2 caption, navi_670)
        if batch is None:
            # single graph: mean over all nodes
            h_graph = h.mean(dim=0, keepdim=True)  # (1, 128)
        else:
            # batched: mean per graph
            h_graph = global_mean_pool(h, batch)    # (B, 128)

        # ── output projection → 3D measurement vector ─────────────────────────
        out = self.output_layer(h_graph)            # (B, 3)

        return out

    def get_node_embeddings(self,
                             x:          torch.Tensor,
                             edge_index: torch.Tensor
                             ) -> torch.Tensor:
        """
        Return node-level embeddings (before pooling).
        Used for inspecting what the GNN has learned.

        Returns
        -------
        h : (N, 128) node embeddings
        """
        h = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.batchnorms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        return h

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from torch_geometric.loader import DataLoader
    from src.graph.dataset import GNSSGraphDataset

    phone_dir = r"data\raw\train\2021-04-22-US-SJC-1\Pixel4"

    print("Loading dataset...")
    dataset = GNSSGraphDataset([phone_dir])
    loader  = DataLoader(dataset, batch_size=4, shuffle=False)
    batch   = next(iter(loader))

    print(f"\nBuilding GNN (Table 4/5 of navi_670):")
    gnn = GNSSGraphSAGE(
        in_dim     = 6,
        hidden_dim = 128,
        out_dim    = 3,
        n_layers   = 8,
    )
    print(gnn)
    print(f"\nTotal parameters: {gnn.count_parameters():,}")

    # ── verify parameter counts match Table 5 ─────────────────────────────────
    print(f"\nLayer parameter breakdown:")
    for name, param in gnn.named_parameters():
        print(f"  {name:<40} {str(list(param.shape)):>20}  "
              f"{param.numel():>8,} params")

    # ── forward pass test ─────────────────────────────────────────────────────
    print(f"\nForward pass test:")
    print(f"  Input x shape      : {batch.x.shape}")
    print(f"  edge_index shape   : {batch.edge_index.shape}")
    print(f"  batch vector shape : {batch.batch.shape}")

    gnn.eval()
    with torch.no_grad():
        out = gnn(batch.x, batch.edge_index, batch.batch)

    print(f"  Output shape       : {out.shape}  (expected: [4, 3])")
    print(f"  Output (corrections in m):\n  {out.numpy().round(3)}")

    # ── single graph test ──────────────────────────────────────────────────────
    d0 = dataset[0]
    with torch.no_grad():
        out_single = gnn(d0.x, d0.edge_index)
    print(f"\nSingle graph output: {out_single.numpy().round(3)}")
    print(f"  True correction  : {d0.y.numpy().round(3)}")
    print(f"  (Random init → large error, training will fix this)")

    # ── node embedding inspection ──────────────────────────────────────────────
    with torch.no_grad():
        embeddings = gnn.get_node_embeddings(d0.x, d0.edge_index)
    print(f"\nNode embeddings shape : {embeddings.shape}  (n_sats × 128)")
    print(f"  Mean activation   : {embeddings.mean().item():.4f}")
    print(f"  Std  activation   : {embeddings.std().item():.4f}")

    print(f"\n✅ GNN ready for tight coupling with BKF (Phase 3 Step 2)")
