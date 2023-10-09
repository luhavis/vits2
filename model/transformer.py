import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.model import convert_pad_shape
from model.normalization import LayerNorm

class RelativePositionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        hidden_channels_ffn: int,
        n_heads: int,
        n_layers: int,
        kernel_size=1,
        dropout=0.0,
        window_size=4,
        gin_channels=0,
        lang_channels=0,
        speaker_cond_layer=0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.speaker_cond_layer = speaker_cond_layer

        self.drop = nn.Dropout(dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layer_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttenstion(hidden_channels if i != 0 else in_channels, hidden_channels, n_heads, p_dropout=dropout, window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, hidden_channels_ffn, kernel_size, p_dropout=dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))
        if gin_channels != 0:
            self.cond = nn.Linear(gin_channels, hidden_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor = None,
        lang: torch.Tensor = None
    ):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):

            if i == self.speaker_cond_layer - 1 and g is not None:
                x = x + self.cond(g.mT).mT
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layer_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(x)
            x = self.norm_layers_2[i](x + y)

        x = x * x_mask
        return x
    
class MultiHeadAttension(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=None,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Linear(channels, channels)
        self.conv_k = nn.Linear(channels, channels)
        self.conv_v = nn.Linear(channels, channels)
        self.conv_o = nn.Linear(channels, out_channels)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.wieght.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)
        
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x.mT).mT
        k = self.conv_k(c.mT).mT
        v = self.conv_v(c.mT).mT


