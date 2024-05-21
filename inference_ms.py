import torch
from scipy.io.wavfile import write

from vits2 import commons, utils
from vits2.models import SynthesizerTrn
from vits2.text import text_to_sequence
from vits2.text.symbols import symbols


def get_text(text: str, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


CONFIG_PATH = "./configs/config.json"
MODEL_PATH = "./logs/model/G_0.pth"
TEXT = "VITS-2 is Awesome!"
SPK_ID = 4
OUTPUT_WAV_PATH = "sample.wav"

hps = utils.get_hparams_from_file(CONFIG_PATH)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # for vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    # posterior_channels,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    mas_noise_scale_initial=0.01,
    noise_scale_delta=2e-6,
    **hps.model,
).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(MODEL_PATH, net_g, None)

stn_tst = get_text(TEXT, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([SPK_ID]).cuda()
    audio = (
        net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
            max_len=1000,
            sdp_radio=1.0,
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )

write(data=audio, rate=hps.data.sampling_rate, filename=OUTPUT_WAV_PATH)
