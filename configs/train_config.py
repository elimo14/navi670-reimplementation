"""
Train/test split matching navi_670 Section 3.
Edit BASE to your data path if needed.
"""
import os

BASE = r"data\raw\train"

# ── TRAIN: MTV + RWC + SF (matches paper's training cities) ───────────────────
TRAIN_DRIVES = [
    "2020-05-14-US-MTV-1", "2020-05-14-US-MTV-2",
    "2020-05-21-US-MTV-1", "2020-05-21-US-MTV-2",
    "2020-05-29-US-MTV-1", "2020-05-29-US-MTV-2",
    "2020-06-04-US-MTV-1", "2020-06-05-US-MTV-1",
    "2020-06-05-US-MTV-2", "2020-06-11-US-MTV-1",
    "2020-07-08-US-MTV-1", "2020-07-17-US-MTV-1",
    "2020-07-17-US-MTV-2", "2020-08-03-US-MTV-1",
    "2020-08-06-US-MTV-2",
    "2020-09-04-US-SF-1",  "2020-09-04-US-SF-2",
    "2021-01-04-US-RWC-1", "2021-01-04-US-RWC-2",
    "2021-01-05-US-SVL-1", "2021-01-05-US-SVL-2",
    "2021-03-10-US-SVL-1",
    "2021-04-15-US-MTV-1",
    "2021-04-28-US-MTV-1", "2021-04-29-US-MTV-1",
]

# ── TEST: SJC + SVL (matches paper's Table 1) ─────────────────────────────────
TEST_DRIVES = [
    "2021-04-22-US-SJC-1",
    "2021-04-28-US-SJC-1",
    "2021-04-29-US-SJC-2",
    "2021-04-26-US-SVL-1",
]


def get_phone_dirs(drive_names):
    """Get all phone subdirectories for a list of drives."""
    from src.train import find_phone_dirs
    dirs = []
    for drive in drive_names:
        drive_path = os.path.join(BASE, drive)
        if os.path.exists(drive_path):
            dirs.extend(find_phone_dirs(drive_path))
    return dirs