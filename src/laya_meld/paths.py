import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "meld"
RESULTS_DIR = ROOT / "results"
# Exported ONNX models are large; point LAYA_MODELS_DIR at an existing export to reuse it.
MODELS_DIR = Path(os.environ.get("LAYA_MODELS_DIR", ROOT / "models"))
