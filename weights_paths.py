import os
from pathlib import Path

_base_path = Path(__file__).parent.parent.resolve()
_models_path = os.environ.get("MODELS_PATH", _base_path / "models")

weight_uvr5_root = f"{_models_path}/voice-swap/rvc_uvr5_weights"
pretrained_pth_root = f"{_models_path}/voice-swap/rvc_pretrained"
