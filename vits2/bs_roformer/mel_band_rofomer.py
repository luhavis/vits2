from functools import partial
from typing import Callable, Optional, Tuple

import torch
from beartype import beartype
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from librosa import filters
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.nn import functional as F
from vits2.bs_roformer.bs_roformer import (Attend, Attention, BandSplit,
                                           MaskEstimator, RMSNorm)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t: torch.Tensor, pad: int, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t: torch.Tensor):
    return F.normalize(t, dim=-1, p=2)


# Attention


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout

        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class LinearAttention(nn.Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_head=32,
        heads=8,
        scale=8,
        flash=False,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.scale = scale
        self.flash = flash
        self.dropout = dropout

        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.zeros(heads, 1, 1))

        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash,
        )

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(self, x: torch.Tensor):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        attn,
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.Tanh,
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in * ((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MelBandRofomer(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=1,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        linear_flash_attn=None,
        dim_freqs_in=1025,
        sample_rate=44100,  # needed for mel filter bank from librosa
        stft_n_fft=2048,
        stft_hop_length=512,  # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.stereo = stereo
        self.num_stems = num_stems
        self.time_transformer_depth = time_transformer_depth
        self.freq_transformer_depth = freq_transformer_depth
        self.linear_transformer_depth = linear_transformer_depth
        self.num_bands = num_bands
        self.dim_head = dim_head
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.flash_attn = flash_attn
        self.linear_flash_attn = linear_flash_attn
        self.dim_freqs_in = dim_freqs_in
        self.sample_rate = sample_rate
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn
        self.mask_estimator_depth = mask_estimator_depth
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_hop_size = multi_stft_hop_size
        self.multi_stft_normalized = multi_stft_normalized
        self.multi_stft_window_fn = multi_stft_window_fn
        self.match_input_audio_length = match_input_audio_length

        self.multi_stft_n_fft = stft_n_fft
        self.audio_channels = 2 if stereo else 1
        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        linear_flash_attn = default(linear_flash_attn, flash_attn)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        (
                            Transformer(
                                depth=linear_transformer_depth,
                                linear_attn=True,
                                flash_attn=linear_flash_attn,
                                **transformer_kwargs,
                            )
                            if linear_transformer_depth > 0
                            else None
                        ),
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            flash_attn=flash_attn,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            flash_attn=flash_attn,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.stft_window_fn = partial(
            default(stft_window_fn, torch.hann_window), stft_win_length
        )

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(
            torch.randn(1, 4096), **self.stft_kwargs, return_complex=True
        ).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(
            sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands
        )
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(
            dim=0
        ).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )

        self.match_input_audio_length = match_input_audio_length

    def forward(
        self,
        raw_audio: torch.Tensor,
        target: torch.Tensor = None,
        return_loss_breakdown: bool = False,
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)."

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(
            raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(
            stft_repr, "b s f t c -> b (f s) t c"
        )  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        # index out all frequencies for all frequency ranges across bands ascending in one go

        batch_arange = torch.arange(batch, device=device)[..., None]

        # account for stereo

        x = stft_repr[batch_arange, self.freq_indices]

        # fold the complex (real and imag) into the frequencies dimension

        x = rearrange(x, "b f t c -> b t (f c)")

        x = self.band_split(x)

        # axial / hierarchical attention

        for linear_transformer, time_transformer, freq_transformer in self.layers:
            if exists(linear_transformer):
                x, ft_ps = pack([x], "b * d")
                x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = repeat(
            self.freq_indices,
            "f -> b n f t",
            b=batch,
            n=num_stems,
            t=stft_repr.shape[-1],
        )

        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(
            2, scatter_indices, masks
        )

        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask

        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )
        recon_audio = rearrange(
            recon_audio,
            "(b n s) t -> b n s t",
            b=batch,
            s=self.audio_channels,
            n=num_stems,
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[
            ..., : recon_audio.shape[-1]
        ]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(
                    window_size, self.multi_stft_n_fft
                ),  # not sure what n_fft is across multi resolution stft.
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_y = torch.stft(
                rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs
            )
            target_y = torch.stft(
                rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs
            )

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(
                recon_y, target_y
            )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
