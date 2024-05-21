import librosa
import numpy as np
import soundfile as sf
import yaml
from ml_collections import ConfigDict
import torch
from torch.nn import functional as F
from vits2.bs_roformer.bs_roformer import BSRoformer

    

def load_model(config_path: str, checkpoint_path: str):
    with open(config_path, encoding="utf-8") as f:
        config = ConfigDict(yaml.load(f, yaml.FullLoader))
    
    model = BSRoformer(**dict(config.model))
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda:0")
    model = model.to(device)

    return model, config

def separate_vocal(audio_path: str, config_path: str, checkpoint_path: str):
    
    model, config = load_model(config_path, checkpoint_path)

    model.eval()
    device = torch.device("cuda:0")
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=-1)

    mix = torch.tensor(audio.T, dtype=torch.float32)

    C = config.audio.chunk_size
    N = config.inference.num_overlap

    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size

    length_init = mix.shape[-1]

    if length_init > 2 * border and (border > 0):
        mix = F.pad(mix, (border, border), mode="reflect")

    window_size = C
    fade_in = torch.linspace(0, 1, fade_size)
    fade_out = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)

    window_start[-fade_size:] *= fade_out
    window_finish[:fade_size] *= fade_in
    window_middle[-fade_size:] *= fade_out
    window_middle[:fade_size] *= fade_in


    with torch.cuda.amp.autocast():

        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = F.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = window_middle
                    if i - step == 0:  # First audio chunk, no fadein
                        window = window_start
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}
