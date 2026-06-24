"""
visualize.py
------------
Phase 4 visualisation: Figures 4, 5, and supporting plots from navi_670.

Generates:
  1. Trajectory plots  — predicted vs ground truth per test phone
     (aerial view + magnified sections, Figure 4-5 style)
  2. Error over time   — per-epoch horizontal error along drive
  3. Learning curves   — training loss and 3D error vs epoch
  4. Per-phone bar chart — horizontal mean error comparison (Table 8 style)
  5. CDF of horizontal error — cumulative distribution across all epochs

Usage:
    python src/visualize.py

Output (saved to results/figures/):
    trajectory_<drive>_<phone>.png
    error_over_time_<drive>_<phone>.png
    learning_curves.png
    per_phone_bar.png
    error_cdf.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, ".")
from src.models.coupled_model import TightlyCoupledGNNBKF
from src.graph.dataset        import GNSSGraphDataset
from src.train                import (
    dataset_to_drives, ecef_error_to_horizontal
)
from src.evaluate             import load_best_model, evaluate_phone
from configs.train_config     import get_phone_dirs, TEST_DRIVES

# ── output directory ──────────────────────────────────────────────────────────
FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "lines.linewidth":  1.5,
})

# San Jose / Sunnyvale reference point for NED projection
REF_LAT = 37.35
REF_LON = -121.90


# ═══════════════════════════════════════════════════════════════════
#  Coordinate utilities
# ═══════════════════════════════════════════════════════════════════

def ecef_to_ned(ecef_pos: np.ndarray,
                ref_lat:  float = REF_LAT,
                ref_lon:  float = REF_LON
                ) -> np.ndarray:
    """
    Convert ECEF position(s) to local NED frame relative to reference.

    Parameters
    ----------
    ecef_pos : (3,) or (N, 3) ECEF positions in metres
    ref_lat, ref_lon : reference point for local NED frame

    Returns
    -------
    ned : (3,) or (N, 3)  [North, East, Down] in metres
    """
    flat = ecef_pos.ndim == 1
    pos  = ecef_pos.reshape(-1, 3)

    lat = np.radians(ref_lat)
    lon = np.radians(ref_lon)

    # NED unit vectors in ECEF
    n_hat = np.array([-np.sin(lat)*np.cos(lon),
                      -np.sin(lat)*np.sin(lon),
                       np.cos(lat)])
    e_hat = np.array([-np.sin(lon),
                       np.cos(lon),
                       0.0])
    d_hat = np.array([-np.cos(lat)*np.cos(lon),
                      -np.cos(lat)*np.sin(lon),
                      -np.sin(lat)])

    # project onto NED axes
    ned = np.stack([
        pos @ n_hat,
        pos @ e_hat,
        pos @ d_hat,
    ], axis=1)

    if flat:
        return ned[0]
    return ned


def get_trajectory(model:       TightlyCoupledGNNBKF,
                   phone_drive: List,
                   device:      torch.device
                   ) -> Dict[str, np.ndarray]:
    """
    Run model on a phone drive and collect position arrays.

    Returns dict with:
        pred  : (T, 3) predicted ECEF positions
        gt    : (T, 3) ground truth ECEF positions
        wls   : (T, 3) WLS initial positions
        epochs: (T,)   timestamps
        has_gt: (T,)   bool mask
    """
    model.eval()
    model.reset_filter()

    preds, gts, wls_pos, epoch_ms, has_gt_mask = [], [], [], [], []

    with torch.no_grad():
        positions, _ = model(
            [d.to(device) for d in phone_drive], reset_filter=False
        )

    for i, d in enumerate(phone_drive):
        preds.append(positions[i].cpu().numpy())
        gts.append(d.gt_pos.float().numpy())
        wls_pos.append(d.rx_pos_wls.float().numpy())
        epoch_ms.append(d.epoch_ms.item())
        has_gt_mask.append(d.has_gt.item())

    return {
        "pred":   np.array(preds),
        "gt":     np.array(gts),
        "wls":    np.array(wls_pos),
        "epochs": np.array(epoch_ms),
        "has_gt": np.array(has_gt_mask, dtype=bool),
    }


# ═══════════════════════════════════════════════════════════════════
#  Figure 1: Trajectory plot  (paper Figures 4 & 5)
# ═══════════════════════════════════════════════════════════════════

def plot_trajectory(traj:      Dict[str, np.ndarray],
                    drive_name: str,
                    phone_name: str,
                    save_dir:   str = FIG_DIR):
    """
    Aerial-view trajectory plot with two magnified insets.
    Reproduces the style of Figures 4 and 5 of navi_670.
    """
    mask = traj["has_gt"]
    if mask.sum() < 10:
        print(f"  Skipping trajectory plot (insufficient ground truth)")
        return

    # convert ECEF → NED (East = x-axis, North = y-axis for map view)
    pred_ned = ecef_to_ned(traj["pred"][mask])
    gt_ned   = ecef_to_ned(traj["gt"][mask])
    wls_ned  = ecef_to_ned(traj["wls"][mask])

    # shift to zero-mean (relative coordinates)
    origin  = gt_ned[0].copy()
    pred_ned -= origin
    gt_ned   -= origin
    wls_ned  -= origin

    east_pred, north_pred = pred_ned[:, 1], pred_ned[:, 0]
    east_gt,   north_gt   = gt_ned[:, 1],   gt_ned[:, 0]
    east_wls,  north_wls  = wls_ned[:, 1],  wls_ned[:, 0]

    # compute per-epoch horizontal errors
    horiz_err = np.sqrt(
        (north_pred - north_gt)**2 + (east_pred - east_gt)**2
    )
    mean_err  = horiz_err.mean()

    fig = plt.figure(figsize=(15, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── left: full trajectory ─────────────────────────────────────────────────
    ax_full = fig.add_subplot(gs[0])
    ax_full.plot(east_gt,   north_gt,   "k-",  lw=1.5, label="Ground truth", zorder=3)
    ax_full.plot(east_pred, north_pred, "b--", lw=1.2, label="Our approach", zorder=4)
    ax_full.plot(east_wls,  north_wls,  "g:",  lw=0.8,
                 alpha=0.6, label="WLS init", zorder=2)
    ax_full.set_xlabel("East (m)")
    ax_full.set_ylabel("North (m)")
    ax_full.set_title(f"Full trajectory\n{drive_name}/{phone_name}")
    ax_full.legend(loc="best", framealpha=0.8)
    ax_full.set_aspect("equal")
    ax_full.grid(True, alpha=0.3)

    # draw two magnification boxes
    n = len(east_gt)
    box_colors = ["#e74c3c", "#2ecc71"]
    zoom_starts = [int(n * 0.15), int(n * 0.65)]
    zoom_lens   = [max(30, n // 10), max(30, n // 10)]

    for j, (zs, zl, bc) in enumerate(zip(zoom_starts, zoom_lens, box_colors)):
        ze = min(zs + zl, n)
        xmin = east_gt[zs:ze].min() - 20
        xmax = east_gt[zs:ze].max() + 20
        ymin = north_gt[zs:ze].min() - 20
        ymax = north_gt[zs:ze].max() + 20
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor=bc, facecolor="none",
            linestyle="--"
        )
        ax_full.add_patch(rect)

        # ── right panels: magnified sections ─────────────────────────────────
        ax_z = fig.add_subplot(gs[j + 1])
        ax_z.plot(east_gt[zs:ze],   north_gt[zs:ze],
                  "k-",  lw=2.0,  label="Ground truth", zorder=3)
        ax_z.plot(east_pred[zs:ze], north_pred[zs:ze],
                  "b--", lw=1.5,  label="Our approach", zorder=4)
        ax_z.plot(east_wls[zs:ze],  north_wls[zs:ze],
                  "g:",  lw=1.0, alpha=0.7, label="WLS init", zorder=2)

        seg_err = horiz_err[zs:ze].mean()
        ax_z.set_xlabel("East (m)")
        ax_z.set_ylabel("North (m)")
        ax_z.set_title(f"Section {j+1} (mean err: {seg_err:.1f} m)",
                        color=bc)
        ax_z.legend(loc="best", framealpha=0.8)
        ax_z.set_aspect("equal")
        ax_z.grid(True, alpha=0.3)

        # add box border color
        for spine in ax_z.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2)

    fig.suptitle(
        f"Trajectory Tracking  |  {drive_name}  |  {phone_name}\n"
        f"Mean horizontal error: {mean_err:.2f} m",
        fontsize=12, y=1.02
    )

    fname = os.path.join(
        save_dir,
        f"trajectory_{drive_name}_{phone_name}.png"
    )
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2: Error over time
# ═══════════════════════════════════════════════════════════════════

def plot_error_over_time(traj:      Dict[str, np.ndarray],
                         drive_name: str,
                         phone_name: str,
                         save_dir:   str = FIG_DIR):
    """Per-epoch horizontal error over the drive."""
    mask = traj["has_gt"]
    if mask.sum() < 10:
        return

    pred_ned = ecef_to_ned(traj["pred"][mask])
    gt_ned   = ecef_to_ned(traj["gt"][mask])
    wls_ned  = ecef_to_ned(traj["wls"][mask])

    horiz_pred = np.sqrt(
        (pred_ned[:, 0] - gt_ned[:, 0])**2 +
        (pred_ned[:, 1] - gt_ned[:, 1])**2
    )
    horiz_wls = np.sqrt(
        (wls_ned[:, 0] - gt_ned[:, 0])**2 +
        (wls_ned[:, 1] - gt_ned[:, 1])**2
    )

    t = np.arange(len(horiz_pred))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, horiz_pred, "b-",  lw=1.0, alpha=0.8, label="Our approach")
    axes[0].plot(t, horiz_wls,  "g--", lw=0.8, alpha=0.6, label="WLS init")
    axes[0].axhline(horiz_pred.mean(), color="b", ls=":",
                    lw=1.5, label=f"Mean: {horiz_pred.mean():.1f} m")
    axes[0].set_ylabel("Horizontal Error (m)")
    axes[0].set_title(f"Horizontal Error over Time  |  "
                       f"{drive_name}/{phone_name}")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, min(horiz_pred.max() * 1.1, 300))
    axes[0].grid(True, alpha=0.3)

    # smoothed version (rolling mean)
    window = min(50, len(horiz_pred) // 10)
    if window > 1:
        smooth = pd.Series(horiz_pred).rolling(window, center=True).mean()
        axes[1].plot(t, smooth, "b-", lw=1.5, label=f"Rolling mean ({window}s)")
        smooth_wls = pd.Series(horiz_wls).rolling(window, center=True).mean()
        axes[1].plot(t, smooth_wls, "g--", lw=1.0, alpha=0.7,
                     label="WLS rolling mean")
        axes[1].set_ylabel("Smoothed Error (m)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch (seconds)")

    fname = os.path.join(
        save_dir,
        f"error_over_time_{drive_name}_{phone_name}.png"
    )
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3: Learning curves
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(history_path: str = "checkpoints/history.json",
                          save_dir: str = FIG_DIR):
    """Plot training loss and 3D error vs epoch."""
    if not os.path.exists(history_path):
        print(f"  History file not found: {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]        for h in history]
    losses     = [h["loss"]         for h in history]
    err3d      = [h["mean_err_3d"]  for h in history]
    north_mean = [h.get("north_mean", float("nan")) for h in history]
    east_mean  = [h.get("east_mean",  float("nan")) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # loss
    axes[0].semilogy(epochs, losses, "b-o", ms=5, lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss (log scale)")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # 3D error
    axes[1].plot(epochs, err3d, "r-o", ms=5, lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean 3D Error (m)")
    axes[1].set_title("Training 3D Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)

    # test horizontal
    valid = [(e, n, ea) for e, n, ea in zip(epochs, north_mean, east_mean)
             if not (np.isnan(n) or np.isnan(ea))]
    if valid:
        vep, vn, ve = zip(*valid)
        axes[2].plot(vep, vn, "g-o", ms=5, lw=2, label="North")
        axes[2].plot(vep, ve, "b-s", ms=5, lw=2, label="East")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Mean Error (m)")
        axes[2].set_title("Test Horizontal Error")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(list(vep))

    fig.suptitle("Learning Curves  |  Tightly Coupled GNN + BKF",
                  fontsize=13, y=1.02)

    fname = os.path.join(save_dir, "learning_curves.png")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 4: Per-phone bar chart  (Table 8 style)
# ═══════════════════════════════════════════════════════════════════

def plot_per_phone_bar(phone_results:  List[Dict],
                        save_dir:  str = FIG_DIR):
    """Horizontal error bar chart per phone (Table 8 format)."""
    if not phone_results:
        return

    labels = [f"{r['drive_name'][-12:]}\n{r['phone_name']}"
              for r in phone_results]
    horiz  = [r["horiz_mean"]   for r in phone_results]
    north  = [r["north_mean"]   for r in phone_results]
    east   = [r["east_mean"]    for r in phone_results]

    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5))

    bars_h = ax.bar(x - w, horiz, w, label="Horizontal", color="#3498db",
                     alpha=0.85, edgecolor="white")
    bars_n = ax.bar(x,     north, w, label="North",      color="#2ecc71",
                     alpha=0.85, edgecolor="white")
    bars_e = ax.bar(x + w, east,  w, label="East",       color="#e74c3c",
                     alpha=0.85, edgecolor="white")

    # value labels on bars
    for bar in list(bars_h) + list(bars_n) + list(bars_e):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # paper reference lines
    ax.axhline(1.1, color="#e74c3c", ls="--", lw=1, alpha=0.7,
               label="Paper East mean (1.1 m)")
    ax.axhline(1.9, color="#2ecc71", ls="--", lw=1, alpha=0.7,
               label="Paper North mean (1.9 m)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Mean Error (m)")
    ax.set_title("Per-Phone Horizontal Positioning Error")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(horiz) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    fname = os.path.join(save_dir, "per_phone_bar.png")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5: CDF of horizontal error
# ═══════════════════════════════════════════════════════════════════

def plot_error_cdf(all_trajectories: List[Tuple[Dict, str, str]],
                   save_dir: str = FIG_DIR):
    """
    Cumulative distribution of epoch-level horizontal errors
    for all test phones. Shows 50th, 75th, 90th percentiles.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(all_trajectories)))

    all_errs_combined = []

    for (traj, drive, phone), color in zip(all_trajectories, colors):
        mask = traj["has_gt"]
        if mask.sum() < 5:
            continue

        pred_ned = ecef_to_ned(traj["pred"][mask])
        gt_ned   = ecef_to_ned(traj["gt"][mask])

        horiz = np.sqrt(
            (pred_ned[:, 0] - gt_ned[:, 0])**2 +
            (pred_ned[:, 1] - gt_ned[:, 1])**2
        )
        all_errs_combined.extend(horiz.tolist())

        sorted_err = np.sort(horiz)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
        label = f"{drive[-14:]}/{phone}"
        ax.plot(sorted_err, cdf, color=color, lw=1.2,
                alpha=0.7, label=label)

    # combined CDF
    if all_errs_combined:
        all_arr = np.sort(all_errs_combined)
        cdf_all = np.arange(1, len(all_arr)+1) / len(all_arr)
        ax.plot(all_arr, cdf_all, "k-", lw=2.5, label="Combined", zorder=5)

        # percentile markers
        for pct, ls in [(50, "--"), (75, ":"), (90, "-.")]:
            val = np.percentile(all_errs_combined, pct)
            ax.axvline(val, color="gray", ls=ls, lw=1, alpha=0.8)
            ax.text(val + 0.5, 0.05 + pct/200,
                    f"P{pct}={val:.1f}m", fontsize=8, color="gray")

    # paper reference
    ax.axvline(1.1, color="red", ls="--", lw=1.5, alpha=0.8,
               label="Paper East mean (1.1 m)")

    ax.set_xlabel("Horizontal Error (m)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of Horizontal Positioning Error (all test phones)")
    ax.set_xlim(0, min(np.percentile(all_errs_combined, 95) * 1.5, 100)
                if all_errs_combined else 100)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fname = os.path.join(save_dir, "error_cdf.png")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu")

    print("="*60)
    print("Phase 4: Visualisation  (navi_670 Figures 4, 5 + extras)")
    print("="*60)

    # ── load model ────────────────────────────────────────────────────────────
    best_path = "checkpoints/best_model.pt"
    if not os.path.exists(best_path):
        ckpts = sorted([f for f in os.listdir("checkpoints")
                        if f.startswith("epoch_")])
        best_path = f"checkpoints/{ckpts[-1]}" if ckpts else None

    if not best_path:
        print("No checkpoint found. Run training first.")
        sys.exit(1)

    print(f"\nLoading model: {best_path}")
    model = load_best_model(best_path, device)

    # ── load evaluation results ───────────────────────────────────────────────
    eval_json = "checkpoints/evaluation_results.json"
    phone_results = []
    if os.path.exists(eval_json):
        with open(eval_json) as f:
            ev = json.load(f)
        phone_results = ev.get("per_phone", [])
        print(f"  Loaded per-phone results for {len(phone_results)} phones")

    # ── load test data ────────────────────────────────────────────────────────
    print(f"\nLoading test drives...")
    test_dirs    = get_phone_dirs(TEST_DRIVES)
    test_dataset = GNSSGraphDataset(test_dirs, threshold=0.5)
    test_drives  = dataset_to_drives(test_dataset)
    print(f"  {len(test_drives)} phone-drives")

    # ── collect trajectories ──────────────────────────────────────────────────
    print(f"\nCollecting trajectories...")
    all_trajs = []

    for i, (drive, phone_dir) in enumerate(zip(test_drives, test_dirs)):
        phone_name = os.path.basename(phone_dir)
        drive_name = os.path.basename(os.path.dirname(phone_dir))
        print(f"  [{i+1}/{len(test_drives)}] {drive_name}/{phone_name}")

        traj = get_trajectory(model, drive, device)
        all_trajs.append((traj, drive_name, phone_name))

    # ── Figure 1: learning curves ─────────────────────────────────────────────
    print(f"\nPlotting learning curves...")
    plot_learning_curves()

    # ── Figure 2: per-phone bar chart ─────────────────────────────────────────
    if phone_results:
        print(f"\nPlotting per-phone bar chart...")
        plot_per_phone_bar(phone_results)

    # ── Figure 3: CDF ────────────────────────────────────────────────────────
    print(f"\nPlotting error CDF...")
    plot_error_cdf(all_trajs)

    # ── Figures 4+: trajectory + error-over-time per phone ────────────────────
    print(f"\nPlotting trajectories and error-over-time...")
    for traj, drive_name, phone_name in all_trajs:
        print(f"  {drive_name}/{phone_name}")
        plot_trajectory(traj, drive_name, phone_name)
        plot_error_over_time(traj, drive_name, phone_name)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("="*60)
    print("Phase 4 complete!")
    print("="*60)


if __name__ == "__main__":
    main()
