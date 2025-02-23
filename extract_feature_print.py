import os, sys, traceback
from pathlib import Path
from ltxcloudapi import get_logger
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from fairseq import checkpoint_utils

log = get_logger(__name__, log_level="INFO")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def extract_feature_print(device, n_part, i_part, i_gpu, exp_dir, version):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)

    assert torch.cuda.is_available()
    device = "cuda"

    base_dir = Path(__file__).parent.parent.resolve()
    default_path = base_dir / "models"
    models_path = Path(os.environ.get("MODELS_PATH", str(default_path))).resolve()
    models_path = models_path / "voice-swap"
    model_path = str(models_path / "hubert_base.pt") 

    wavPath = "%s/1_16k_wavs" % exp_dir
    outPath = (
        "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
    )
    os.makedirs(outPath, exist_ok=True)

    # wave must be 16k, hop_size=320
    def readwave(wav_path, normalize=False):
        wav, sr = sf.read(wav_path)
        assert sr == 16000
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        feats = feats.view(1, -1)
        return feats


    # HuBERT model
    # if hubert model doesn't exist
    if os.access(model_path, os.F_OK) == False:
        raise Exception(f"Extracting failed because `{model_path}` does not exist.")

    log.info("load model(s) from {}".format(model_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    log.info("move model to %s" % device)
    if device not in ["mps", "cpu"]:
        model = model.half()
    model.eval()

    todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        log.info("no-feature-todo")
    else:
        log.info(f"extract_feature ({len(todo)})")
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.half().to(device)
                        if device not in ["mps", "cpu"]
                        else feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        log.info("%s-contains nan" % file)
                    if idx % n == 0:
                        log.info("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
            except:
                raise Exception(traceback.format_exc())
        log.info("extract_feature done")
