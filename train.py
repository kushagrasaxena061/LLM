"""Root-level Production Training Script for MiniGPT-151M."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from model.config import canonical_151m_config
from training.train import train_production_151m

if __name__ == "__main__":
    print("🚀 Launching Canonical MiniGPT-151M Production Training Path...")
    train_production_151m(config=canonical_151m_config)
