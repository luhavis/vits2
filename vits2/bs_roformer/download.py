"""
Download Pre-trained models

https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/main?tab=readme-ov-file#pre-trained-models
"""
import os
import requests
from typing import Literal

from loguru import logger

VOCAL_MODELS = Literal["BS Roformer"]
MODEL_FILE_TYPE = Literal["config", "checkpoint"]

VOCAL_MODEL_DOWNLOAD_URL = {
    "BS Roformer": {
        "config": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "checkpoint": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    }
}

def download_pre_trainted_model(model_name: VOCAL_MODELS, model_file_type: MODEL_FILE_TYPE):

    logger.info(f"Downloading...")
    r = requests.get(
        VOCAL_MODEL_DOWNLOAD_URL[model_name][model_file_type],
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"},
    )
    r.raise_for_status()

    file_extension = os.path.splitext(VOCAL_MODEL_DOWNLOAD_URL[model_name][model_file_type])[-1]
    with open(f"configs/{model_name}{file_extension}", "wb") as f:
        f.write(r.content)
    logger.info(f"Download success. path: configs/{model_name}{file_extension}")
    

