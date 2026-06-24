"""
run_training.py
Full training run matching navi_670 Section 3.
Run from project root: python run_training.py
"""
import sys
sys.path.insert(0, ".")

from configs.train_config import get_phone_dirs, TRAIN_DRIVES, TEST_DRIVES
from src.train import train

train_dirs = get_phone_dirs(TRAIN_DRIVES)
test_dirs  = get_phone_dirs(TEST_DRIVES)

print(f"Train phones : {len(train_dirs)}")
print(f"Test  phones : {len(test_dirs)}")

train({
    "train_dirs":   train_dirs,
    "test_dirs":    test_dirs,
    "n_epochs":     11,
    "lr":           0.001,
    "weight_decay": 5e-4,
    "hidden_dim":   128,
    "n_layers":     8,
    "threshold":    0.5,
    "device":       "cpu",
    "save_dir":     "checkpoints",
    "log_every":    1,
})