import os

_models_dir = os.environ.get("MODELS_PATH", None)

weight_uvr5_root = "uvr5_weights" if _models_dir is None else f"{_models_dir}/voice-swap/rvc_uvr5_weights"
pretrained_pth_root = "pretrained" if _models_dir is None else f"{_models_dir}/voice-swap/rvc_pretrained"