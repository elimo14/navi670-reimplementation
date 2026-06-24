"""
evaluate.py
-----------
Phase 4 evaluation: reproduce Tables 6, 7, 8 from navi_670.

Loads the best trained checkpoint and evaluates on test drives,
reporting per-phone and aggregate horizontal positioning errors
in North and East directions.

Usage:
    python src/evaluate.py

Output:
    - Console: Tables 6, 7, 8 (paper format)
    - checkpoints/evaluation_results.json
    - checkpoints/per_phone_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple

sys.path.insert(0, ".")
from src.models.coupled_model import TightlyCoupledGNNBKF
from src.graph.dataset        import GNSSGraphDataset
from src.train                import (
    dataset_to_drives, find_phone_dirs, ecef_error_to_horizontal
)
from configs.train_config import get_phone_dirs, TEST_DRIVES, BASE


# ── reference coordinates for NED frame (San Jose / Sunnyvale area) ───────────
REF_LAT = 37.35
REF_LON = -121.90


def load_best_model(checkpoint_path: str,
                    device: torch.device) -> TightlyCoupledGNNBKF:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})

    model = TightlyCoupledGNNBKF(
        in_dim       = 6,
        hidden_dim   = cfg.get("hidden_dim", 128),
        n_layers     = cfg.get("n_layers",   8),
        state_dim    = 3,
        init_Q_scale = 1e-3,
        init_R_scale = 1e-2,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint from epoch {epoch}")
    if "test_metrics" in ckpt:
        tm = ckpt["test_metrics"]
        print(f"  Checkpoint test metrics: "
              f"N={tm.get('north_mean',float('nan')):.2f} m, "
              f"E={tm.get('east_mean', float('nan')):.2f} m")
    return model


@torch.no_grad()
def evaluate_phone(model:       TightlyCoupledGNNBKF,
                   phone_drive: List,
                   phone_dir:   str,
                   device:      torch.device
                   ) -> Dict:
    """
    Evaluate model on one phone-drive.

    Returns dict with per-epoch errors and aggregate metrics.
    """
    if not phone_drive:
        return {}

    drive_device = [d.to(device) for d in phone_drive]
    model.reset_filter()
    positions, _ = model(drive_device, reset_filter=False)

    north_errs = []
    east_errs  = []
    errs_3d    = []

    for i, d in enumerate(drive_device):
        if not d.has_gt.item():
            continue
        gt = d.gt_pos.float().to(device)
        n, e, err3d = ecef_error_to_horizontal(
            positions[i], gt, REF_LAT, REF_LON
        )
        north_errs.append(n)
        east_errs.append(e)
        errs_3d.append(err3d)

    if not north_errs:
        return {}

    n_arr = np.array(north_errs)
    e_arr = np.array(east_errs)
    h_arr = np.sqrt(n_arr**2 + e_arr**2)

    # extract drive and phone name from path
    parts     = phone_dir.replace("\\", "/").split("/")
    phone_name = parts[-1] if parts else "unknown"
    drive_name = parts[-2] if len(parts) > 1 else "unknown"

    return {
        "phone_dir":     phone_dir,
        "drive_name":    drive_name,
        "phone_name":    phone_name,
        "n_epochs":      len(north_errs),
        # North metrics (Table 7)
        "north_mean":    float(np.mean(n_arr)),
        "north_median":  float(np.median(n_arr)),
        "north_min":     float(np.min(n_arr)),
        "north_max":     float(np.max(n_arr)),
        # East metrics (Table 6)
        "east_mean":     float(np.mean(e_arr)),
        "east_median":   float(np.median(e_arr)),
        "east_min":      float(np.min(e_arr)),
        "east_max":      float(np.max(e_arr)),
        # Horizontal (Table 8)
        "horiz_mean":    float(np.mean(h_arr)),
        "horiz_median":  float(np.median(h_arr)),
        "horiz_min":     float(np.min(h_arr)),
        "horiz_max":     float(np.max(h_arr)),
        # 3D
        "err3d_mean":    float(np.mean(errs_3d)),
        "err3d_median":  float(np.median(errs_3d)),
        # percentile errors
        "north_p25":     float(np.percentile(n_arr, 25)),
        "north_p75":     float(np.percentile(n_arr, 75)),
        "east_p25":      float(np.percentile(e_arr, 25)),
        "east_p75":      float(np.percentile(e_arr, 75)),
    }


def aggregate_metrics(phone_results: List[Dict]) -> Dict:
    """Aggregate per-phone results into overall metrics (Tables 6 & 7)."""
    all_north = []
    all_east  = []
    all_horiz = []

    for r in phone_results:
        if not r:
            continue
        n = r["north_mean"]
        e = r["east_mean"]
        all_north.append(n)
        all_east.append(e)
        all_horiz.append(np.sqrt(n**2 + e**2))

    if not all_north:
        return {}

    # also collect epoch-level stats from individual records
    all_n_ep = []
    all_e_ep = []
    for r in phone_results:
        if r:
            all_n_ep.append(r["north_mean"])
            all_e_ep.append(r["east_mean"])

    return {
        "n_phones": len(all_north),
        # Table 6 — East
        "east_mean":   float(np.mean(all_east)),
        "east_median": float(np.median(all_east)),
        "east_min":    float(np.min(all_east)),
        "east_max":    float(np.max(all_east)),
        # Table 7 — North
        "north_mean":   float(np.mean(all_north)),
        "north_median": float(np.median(all_north)),
        "north_min":    float(np.min(all_north)),
        "north_max":    float(np.max(all_north)),
        # Horizontal
        "horiz_mean":   float(np.mean(all_horiz)),
        "horiz_median": float(np.median(all_horiz)),
    }


def print_table6(agg: Dict):
    """Print Table 6 — East direction errors."""
    paper = {"mean": 1.1, "median": 1.1, "min": 0.6, "max": 1.8}
    kf    = {"mean": 4.6, "median": 4.5, "min": 1.2, "max": 14.1}
    loose = {"mean": 3.5, "median": 3.4, "min": 2.1, "max": 4.8}

    print("\n" + "="*68)
    print("TABLE 6  |  Positioning Error in the EAST Direction (m)")
    print("="*68)
    print(f"{'Metric':<10} {'KF':>8} {'Loose GNN':>10} "
          f"{'Paper':>8} {'Ours':>8}")
    print("-"*68)
    for stat in ["mean", "median", "min", "max"]:
        ours = agg.get(f"east_{stat}", float("nan"))
        print(f"{stat.capitalize():<10} "
              f"{kf[stat]:>8.1f} "
              f"{loose[stat]:>10.1f} "
              f"{paper[stat]:>8.1f} "
              f"{ours:>8.2f}")
    print("="*68)


def print_table7(agg: Dict):
    """Print Table 7 — North direction errors."""
    paper = {"mean": 1.9, "median": 1.8, "min": 0.7, "max": 5.8}
    kf    = {"mean": 3.0, "median": 2.1, "min": 1.3, "max": 7.7}
    loose = {"mean": 2.0, "median": 1.8, "min": 1.0, "max": 5.7}

    print("\n" + "="*68)
    print("TABLE 7  |  Positioning Error in the NORTH Direction (m)")
    print("="*68)
    print(f"{'Metric':<10} {'KF':>8} {'Loose GNN':>10} "
          f"{'Paper':>8} {'Ours':>8}")
    print("-"*68)
    for stat in ["mean", "median", "min", "max"]:
        ours = agg.get(f"north_{stat}", float("nan"))
        print(f"{stat.capitalize():<10} "
              f"{kf[stat]:>8.1f} "
              f"{loose[stat]:>10.1f} "
              f"{paper[stat]:>8.1f} "
              f"{ours:>8.2f}")
    print("="*68)


def print_table8(phone_results: List[Dict]):
    """Print Table 8 — Per-phone horizontal errors."""
    paper_results = {
        ("2021-04-02-US-SJC-1", "Pixel4"):            ("3.7",  "2.7"),
        ("2021-04-02-US-SJC-1", "Pixel5"):            ("5.0",  "2.2"),
        ("2021-04-02-US-SJC-1", "SamsungS20Ultra"):   ("4.1",  "2.5"),
        ("2021-04-02-US-SJC-1", "Mi8"):               ("5.1",  "2.6"),
        ("2021-04-26-US-SVL-2", "SamsungS20Ultra"):   ("3.7",  "2.1"),
        ("2021-04-26-US-SVL-2", "Mi8"):               ("5.1",  "2.3"),
        ("2021-08-04-US-SJC-1", "Pixel4"):            ("2.5",  "1.4"),
        ("2021-08-04-US-SJC-1", "Pixel5"):            ("6.6",  "5.9"),
        ("2021-08-04-US-SJC-1", "SamsungS20Ultra"):   ("3.0",  "1.4"),
        ("2021-08-24-US-SVL-1", "Pixel4"):            ("4.2",  "1.3"),
        ("2021-08-24-US-SVL-1", "Pixel5"):            ("4.8",  "1.4"),
        ("2021-08-24-US-SVL-1", "SamsungS20Ultra"):   ("2.9",  "1.9"),
        ("2021-08-24-US-SVL-1", "Mi8"):               ("2.8",  "2.2"),
    }

    print("\n" + "="*75)
    print("TABLE 8  |  Horizontal Localization Errors (m) on Test Data Sets")
    print("="*75)
    print(f"{'No':>3}  {'Drive':<28} {'Phone':<18} "
          f"{'Baseline':>9} {'Ours':>8}")
    print("-"*75)

    for i, r in enumerate(phone_results, 1):
        if not r:
            continue
        drive = r["drive_name"]
        phone = r["phone_name"]
        ours  = r["horiz_mean"]

        # try to find paper baseline for this phone
        baseline = "--"
        for (d, p), (bl, pa) in paper_results.items():
            if p.lower() in phone.lower() or phone.lower() in p.lower():
                if any(x in drive for x in d.split("-")):
                    baseline = bl
                    break

        print(f"{i:>3}.  {drive:<28} {phone:<18} "
              f"{baseline:>9} {ours:>8.2f}")

    print("="*75)


def print_error_distribution(phone_results: List[Dict]):
    """Print horizontal error distribution across phones."""
    all_horiz = [r["horiz_mean"] for r in phone_results if r]
    if not all_horiz:
        return

    print("\n" + "="*50)
    print("Horizontal Error Distribution (per-phone mean)")
    print("="*50)
    bins   = [0, 2, 5, 10, 20, 50, float("inf")]
    labels = ["0-2 m", "2-5 m", "5-10 m", "10-20 m", "20-50 m", ">50 m"]
    counts = np.histogram(all_horiz, bins=bins)[0]
    for label, count in zip(labels, counts):
        pct = 100 * count / len(all_horiz)
        bar = "+" * int(pct / 5)
        print(f"  {label:>8}: {count:2d} phones ({pct:5.1f}%) {bar}")

    print(f"\n  Mean  horizontal: {np.mean(all_horiz):.2f} m")
    print(f"  Median horizontal: {np.median(all_horiz):.2f} m")
    print("="*50)


def main():
    device = torch.device("cpu")

    # ── find best checkpoint ──────────────────────────────────────────────────
    ckpt_dir  = "checkpoints"
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(best_path):
        # fall back to last epoch
        epoch_ckpts = sorted([
            f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")
        ])
        if not epoch_ckpts:
            print("No checkpoints found. Run training first.")
            sys.exit(1)
        best_path = os.path.join(ckpt_dir, epoch_ckpts[-1])

    print("="*60)
    print("Phase 4: Evaluation  (navi_670 Tables 6, 7, 8)")
    print("="*60)
    print(f"\nLoading model from: {best_path}")
    model = load_best_model(best_path, device)

    # ── load test data ────────────────────────────────────────────────────────
    print(f"\nLoading test drives...")
    test_dirs = get_phone_dirs(TEST_DRIVES)
    print(f"  Test phones: {len(test_dirs)}")

    test_dataset = GNSSGraphDataset(test_dirs, threshold=0.5)
    test_drives  = dataset_to_drives(test_dataset)
    print(f"  Test drives: {len(test_drives)}")

    # ── evaluate per phone ────────────────────────────────────────────────────
    print(f"\nEvaluating {len(test_drives)} phone-drives...")
    phone_results = []

    for i, (drive, phone_dir) in enumerate(zip(test_drives, test_dirs)):
        phone_name = os.path.basename(phone_dir)
        drive_name = os.path.basename(os.path.dirname(phone_dir))
        print(f"  [{i+1:2d}/{len(test_drives)}] {drive_name}/{phone_name}...",
              end="", flush=True)

        result = evaluate_phone(model, drive, phone_dir, device)
        phone_results.append(result)

        if result:
            print(f" N={result['north_mean']:.2f} m  "
                  f"E={result['east_mean']:.2f} m  "
                  f"Horiz={result['horiz_mean']:.2f} m")
        else:
            print(" (no ground truth)")

    # ── aggregate ─────────────────────────────────────────────────────────────
    valid_results = [r for r in phone_results if r]
    agg = aggregate_metrics(valid_results)

    # ── print tables ──────────────────────────────────────────────────────────
    print_table6(agg)
    print_table7(agg)
    print_table8(valid_results)
    print_error_distribution(valid_results)

    # ── overall summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY  |  Our Results vs Paper")
    print("="*60)
    print(f"{'Metric':<25} {'Our Result':>12} {'Paper':>8}")
    print("-"*60)
    rows = [
        ("East mean (m)",    agg.get("east_mean",   float("nan")), 1.1),
        ("East median (m)",  agg.get("east_median", float("nan")), 1.1),
        ("North mean (m)",   agg.get("north_mean",  float("nan")), 1.9),
        ("North median (m)", agg.get("north_median",float("nan")), 1.8),
        ("Horiz mean (m)",   agg.get("horiz_mean",  float("nan")), float("nan")),
        ("Horiz median (m)", agg.get("horiz_median",float("nan")), float("nan")),
    ]
    for name, ours, paper in rows:
        paper_str = f"{paper:.1f}" if not np.isnan(paper) else "  N/A"
        print(f"{name:<25} {ours:>12.2f} {paper_str:>8}")
    print("="*60)

    # ── save results ──────────────────────────────────────────────────────────
    out_dir = ckpt_dir

    # JSON
    results_dict = {
        "aggregate": agg,
        "per_phone": valid_results,
        "model_path": best_path,
    }
    json_path = os.path.join(out_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # CSV (Table 8 format)
    if valid_results:
        csv_path = os.path.join(out_dir, "per_phone_results.csv")
        pd.DataFrame(valid_results).to_csv(csv_path, index=False)
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")

    return results_dict


if __name__ == "__main__":
    main()
