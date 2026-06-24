"""
train.py
--------
Joint training loop for the tightly coupled GNN + BKF model.

From Section 3 (Experimental Setup) of navi_670:

  Optimizer   : Adam (Table 4)
  LR          : 0.001 (Table 4)
  Weight decay: 5e-4  (Table 4)
  Loss        : MSE between predicted and true 3D position
  Epochs      : 11   (Table 4)

Training strategy (Section 2, Figure 1):
  For each drive (sequence of epochs):
    1. Reset BKF at start of drive
    2. For each epoch t in drive:
       a. GNN forward → correction z_t
       b. BKF predict + update → final position
       c. Accumulate MSE loss
    3. Backpropagate through entire sequence
    4. Update GNN weights + BKF Q, R jointly

Train/test split (Section 3):
  Train: 49 datasets from Mountain View / San Jose
  Test : San Jose + Sunnyvale datasets (Table 1)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from typing import List, Dict, Tuple
import json

sys.path.insert(0, ".")
from src.models.coupled_model import TightlyCoupledGNNBKF
from src.graph.dataset        import GNSSGraphDataset


# ═══════════════════════════════════════════════════════════════════
#  Helper: convert 3D error to horizontal (North/East)
# ═══════════════════════════════════════════════════════════════════

def ecef_error_to_horizontal(pred_pos:  torch.Tensor,
                              gt_pos:    torch.Tensor,
                              ref_lat:   float = 37.4,
                              ref_lon:   float = -121.9
                              ) -> Tuple[float, float, float]:
    """
    Decompose ECEF position error into North, East, 3D components.

    Parameters
    ----------
    pred_pos : (3,) predicted ECEF position
    gt_pos   : (3,) ground truth ECEF position
    ref_lat  : float reference latitude for NED frame (degrees)
    ref_lon  : float reference longitude for NED frame (degrees)

    Returns
    -------
    err_north, err_east, err_3d  (all in metres)
    """
    diff = (pred_pos - gt_pos).detach().cpu().numpy()
    err_3d = float(np.linalg.norm(diff))

    lat = np.radians(ref_lat)
    lon = np.radians(ref_lon)

    # North and East unit vectors in ECEF
    n = np.array([-np.sin(lat)*np.cos(lon),
                  -np.sin(lat)*np.sin(lon),
                   np.cos(lat)])
    e = np.array([-np.sin(lon),
                   np.cos(lon),
                   0.0])

    err_north = float(abs(diff @ n))
    err_east  = float(abs(diff @ e))
    return err_north, err_east, err_3d


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model:     TightlyCoupledGNNBKF,
                    drives:    List[List],
                    optimizer: torch.optim.Optimizer,
                    device:    torch.device
                    ) -> Dict:
    """
    Train over all drives for one epoch.

    Each drive is a chronological sequence of PyG Data objects.
    BKF is reset at the start of each drive.

    Returns dict with training metrics.
    """
    model.train()
    total_loss  = 0.0
    total_steps = 0
    errors_3d   = []

    for drive_idx, drive in enumerate(drives):
        if len(drive) == 0:
            continue

        # move data to device
        drive_device = [d.to(device) for d in drive]

        # ── forward pass over entire drive ────────────────────────────────────
        model.reset_filter()
        positions, _ = model(drive_device, reset_filter=False)

        # ── compute MSE loss ─────────────────────────────────────────────────
        gt_positions = torch.stack([
            d.gt_pos.float().to(device) for d in drive_device
        ])
        has_gt = torch.tensor([d.has_gt.item() for d in drive_device])

        if has_gt.sum() == 0:
            continue


        valid_pred = positions[has_gt]
        valid_gt   = gt_positions[has_gt]

        # ── work in CORRECTION space (relative to WLS) ────────────────────────
        # This avoids the large ECEF offset (~6.37M m) dominating gradients.
        # correction_pred ≈ 0-200m instead of 6.37M m
        rx_pos_batch    = torch.stack([
            d.rx_pos_wls.float().to(device) for d in drive_device
        ])
        valid_rx        = rx_pos_batch[has_gt]
        correction_pred = valid_pred - valid_rx   # predicted correction
        correction_true = valid_gt   - valid_rx   # true correction

        # ── Huber loss: robust to BKF divergence outliers ─────────────────────
        # MSE amplifies outliers (20,000 m error → 4×10^8 loss contribution)
        # Huber is MSE for errors < delta, linear for errors > delta
        loss = nn.functional.huber_loss(
            correction_pred, correction_true, delta=50.0
        )



        # ── backward + update ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # ── collect metrics ───────────────────────────────────────────────────
        total_loss  += loss.item()
        total_steps += 1

        with torch.no_grad():
            for i in range(len(valid_pred)):
                _, _, err3d = ecef_error_to_horizontal(
                    valid_pred[i], valid_gt[i]
                )
                errors_3d.append(err3d)


    return {
        "loss":        total_loss / max(total_steps, 1),
        "mean_err_3d": float(np.mean(errors_3d)) if errors_3d else float("nan"),
        "n_drives":    total_steps,
    }


@torch.no_grad()
def evaluate(model:  TightlyCoupledGNNBKF,
             drives:  List[List],
             device:  torch.device
             ) -> Dict:
    """
    Evaluate model on a list of drives.

    Returns per-drive and aggregate metrics matching paper's Table 6-7:
      mean, median, min, max horizontal error (North + East)
    """
    model.eval()
    all_north = []
    all_east  = []
    all_3d    = []

    for drive in drives:
        if len(drive) == 0:
            continue

        drive_device = [d.to(device) for d in drive]
        model.reset_filter()
        positions, _ = model(drive_device, reset_filter=False)

        for i, d in enumerate(drive_device):
            if not d.has_gt.item():
                continue
            gt = d.gt_pos.float().to(device)
            n, e, err3d = ecef_error_to_horizontal(positions[i], gt)
            all_north.append(n)
            all_east.append(e)
            all_3d.append(err3d)

    if not all_3d:
        return {}

    return {
        # Horizontal metrics (North direction, Table 7)
        "north_mean":   float(np.mean(all_north)),
        "north_median": float(np.median(all_north)),
        "north_min":    float(np.min(all_north)),
        "north_max":    float(np.max(all_north)),
        # Horizontal metrics (East direction, Table 6)
        "east_mean":    float(np.mean(all_east)),
        "east_median":  float(np.median(all_east)),
        "east_min":     float(np.min(all_east)),
        "east_max":     float(np.max(all_east)),
        # 3D error
        "err3d_mean":   float(np.mean(all_3d)),
        "err3d_median": float(np.median(all_3d)),
        "n_epochs":     len(all_3d),
    }


# ═══════════════════════════════════════════════════════════════════
#  Dataset splitting
# ═══════════════════════════════════════════════════════════════════

def dataset_to_drives(dataset: GNSSGraphDataset) -> List[List]:
    """
    Split dataset into per-phone drives.
    Each phone = one drive — BKF resets between phones.

    Falls back to time-gap splitting if phone_id not available.
    """
    # ── preferred: split by phone identity ───────────────────────────────────
    if hasattr(dataset._data_list[0], "phone_id"):
        drives = {}
        for data in dataset._data_list:
            pid = data.phone_id.item()
            if pid not in drives:
                drives[pid] = []
            drives[pid].append(data)
        result = [drives[pid] for pid in sorted(drives.keys())]
        print(f"  Drive split: {len(result)} phone-drives "
              f"from {len(dataset._data_list)} graphs")
        return result

    # ── fallback: time-gap splitting (legacy) ─────────────────────────────────
    drives = []
    current_drive = []
    for i, data in enumerate(dataset._data_list):
        if i == 0:
            current_drive.append(data)
            continue
        prev_ms = dataset._data_list[i-1].epoch_ms.item()
        curr_ms = data.epoch_ms.item()
        if curr_ms - prev_ms > 10_000:
            if current_drive:
                drives.append(current_drive)
            current_drive = [data]
        else:
            current_drive.append(data)
    if current_drive:
        drives.append(current_drive)
    return drives


def find_phone_dirs(base_dir: str) -> List[str]:
    """
    Recursively find all phone directories under base_dir.
    A phone directory contains a *_derived.csv file.
    """
    phone_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith("_derived.csv"):
                phone_dirs.append(root)
                break
    return sorted(phone_dirs)


# ═══════════════════════════════════════════════════════════════════
#  Main training function
# ═══════════════════════════════════════════════════════════════════

def train(config: Dict = None):
    """
    Main training entry point.

    Parameters
    ----------
    config : dict of hyperparameters (defaults match Table 4 of navi_670)
    """
    # ── default config (Table 4) ──────────────────────────────────────────────
    cfg = {
        "train_dirs":    [],              # list of phone dirs for training
        "test_dirs":     [],              # list of phone dirs for testing
        "n_epochs":      11,              # Table 4
        "lr":            0.001,           # Table 4
        "weight_decay":  5e-4,            # Table 4
        "hidden_dim":    128,             # Table 4
        "n_layers":      8,               # Table 4
        "threshold":     0.5,             # Table 4
        "device":        "cpu",
        "save_dir":      "checkpoints",
        "log_every":     1,               # print every N epochs
    }
    if config:
        cfg.update(config)

    device = torch.device(cfg["device"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    print("=" * 60)
    print("Tightly Coupled GNN + BKF Training (navi_670)")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Epochs      : {cfg['n_epochs']}")
    print(f"LR          : {cfg['lr']}")
    print(f"Weight decay: {cfg['weight_decay']}")
    print(f"Hidden dim  : {cfg['hidden_dim']}")
    print(f"GNN layers  : {cfg['n_layers']}")

    # ── build datasets ────────────────────────────────────────────────────────
    print(f"\nLoading training data...")
    if not cfg["train_dirs"]:
        raise ValueError("config['train_dirs'] must be a non-empty list of paths")

    train_dataset = GNSSGraphDataset(
        cfg["train_dirs"], threshold=cfg["threshold"]
    )
    train_drives = dataset_to_drives(train_dataset)
    print(f"  Train: {len(train_dataset)} graphs in {len(train_drives)} drives")

    test_drives = []
    if cfg["test_dirs"]:
        print(f"\nLoading test data...")
        test_dataset = GNSSGraphDataset(
            cfg["test_dirs"], threshold=cfg["threshold"]
        )
        test_drives = dataset_to_drives(test_dataset)
        print(f"  Test : {len(test_dataset)} graphs in {len(test_drives)} drives")

    # ── build model ───────────────────────────────────────────────────────────
    model = TightlyCoupledGNNBKF(
    in_dim          = 6,
    hidden_dim      = cfg["hidden_dim"],
    n_layers        = cfg["n_layers"],
    state_dim       = 3,
    init_Q_scale    = 1e-3,
    init_R_scale    = 1e-2,
    use_dynamic_cov = cfg.get("use_dynamic_cov", False),  # ← ADD THIS
).to(device)


    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  GNN : {params['gnn']:,}")
    print(f"  BKF : {params['bkf']}")
    print(f"  Total: {params['total']:,}")

    # ── optimizer (Table 4: Adam, lr=0.001, weight_decay=5e-4) ───────────────
    optimizer = Adam(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train 3D(m)':>12} "
          f"{'Test N(m)':>10} {'Test E(m)':>10}  Time")
    print("-" * 65)

    history = []
    best_test_horiz = float("inf")

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_drives, optimizer, device
        )

        # ── evaluate ──────────────────────────────────────────────────────────
        test_metrics = {}
        if test_drives:
            test_metrics = evaluate(model, test_drives, device)

        scheduler.step()
        elapsed = time.time() - t0

        # ── logging ───────────────────────────────────────────────────────────
        if epoch % cfg["log_every"] == 0:
            test_n = test_metrics.get("north_mean", float("nan"))
            test_e = test_metrics.get("east_mean",  float("nan"))
            print(f"{epoch:>6} "
                  f"{train_metrics['loss']:>12.4f} "
                  f"{train_metrics['mean_err_3d']:>12.2f} "
                  f"{test_n:>10.2f} "
                  f"{test_e:>10.2f} "
                  f" {elapsed:.1f}s")

        # ── save best model ───────────────────────────────────────────────────
        if test_metrics:
            horiz = test_metrics.get("north_mean", float("inf")) + \
                    test_metrics.get("east_mean",  float("inf"))
            if horiz < best_test_horiz:
                best_test_horiz = horiz
                ckpt_path = os.path.join(cfg["save_dir"], "best_model.pt")
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optim_state":  optimizer.state_dict(),
                    "test_metrics": test_metrics,
                    "config":       cfg,
                }, ckpt_path)

        # ── save checkpoint every epoch ───────────────────────────────────────
        ckpt_path = os.path.join(cfg["save_dir"], f"epoch_{epoch:02d}.pt")
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "train_metrics":train_metrics,
            "test_metrics": test_metrics,
            "config":       cfg,
        }, ckpt_path)

        record = {"epoch": epoch, **train_metrics, **test_metrics}
        history.append(record)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete!")
    if test_metrics:
        print(f"\nFinal test metrics (Table 6-7 format):")
        print(f"  {'Metric':<20} {'North (m)':>10} {'East (m)':>10}")
        print(f"  {'-'*42}")
        for stat in ["mean", "median", "min", "max"]:
            n = test_metrics.get(f"north_{stat}", float("nan"))
            e = test_metrics.get(f"east_{stat}",  float("nan"))
            print(f"  {stat.capitalize():<20} {n:>10.2f} {e:>10.2f}")

    # save history
    history_path = os.path.join(cfg["save_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")

    return model, history


# ── quick test (single phone, 3 epochs) ──────────────────────────────────────
if __name__ == "__main__":
    # ── find available train phone dirs ───────────────────────────────────────
    train_base = r"data\raw\train"
    all_dirs   = find_phone_dirs(train_base)

    if not all_dirs:
        print("No phone directories found. Check your data path.")
        sys.exit(1)

    print(f"Found {len(all_dirs)} phone directories")
    for d in all_dirs[:5]:
        print(f"  {d}")
    if len(all_dirs) > 5:
        print(f"  ... and {len(all_dirs)-5} more")

    # ── use first drive for both train and test (smoke test) ──────────────────
    # For real training, split into proper train/test per Section 3 of navi_670
    train_dirs = all_dirs[:1]   # one phone for quick test
    test_dirs  = all_dirs[:1]   # same phone to verify pipeline

    print(f"\nSmoke test: 3 epochs on 1 phone")
    print(f"  Train: {train_dirs[0]}")

    model, history = train({
        "train_dirs":  train_dirs,
        "test_dirs":   test_dirs,
        "n_epochs":    3,          # full run uses 11 (Table 4)
        "lr":          0.001,
        "weight_decay":5e-4,
        "hidden_dim":  128,
        "n_layers":    8,
        "threshold":   0.5,
        "device":      "cpu",
        "save_dir":    "checkpoints",
        "log_every":   1,
    })

    print(f"\nLoss curve (should decrease each epoch):")
    for h in history:
        print(f"  Epoch {h['epoch']}: loss={h['loss']:.4f}  "
              f"3D_err={h['mean_err_3d']:.2f} m")
