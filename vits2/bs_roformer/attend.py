from collections import namedtuple
from functools import wraps

import torch
from packaging import version
from torch import einsum
from torch.nn import functional as F
from torch import nn

FLASH_ATTENTION_CONFIG = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        flash=False,
        scale=None,
    ):
        super().__init__()
        self.dropout = dropout
        self.flash = flash
        self.scale = scale

        self.attn_dropout = nn.Dropout(dropout)

        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        self.cpu_config = FLASH_ATTENTION_CONFIG(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = FLASH_ATTENTION_CONFIG(True, False, False)
        else:
            print_once(
                "Non-A1000 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = FLASH_ATTENTION_CONFIG(False, True, True)

    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
