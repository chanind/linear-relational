from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
