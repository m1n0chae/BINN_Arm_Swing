# models_cnn.py
from typing import Sequence
import torch
from torch import nn

class _BTF_to_BFT(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T)
        return x.transpose(1, 2).contiguous()

class _BFT_to_BTF(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, T) -> (B, T, F)
        return x.transpose(1, 2).contiguous()

class CNNRegressor(nn.Module):
    """
    Multi-layer, multi-kernel 1D CNN.
    - input  : (B, T, F)
    - output : (B, T, C)
    - channels : output channels of each block (e.g. [256,128,128,512,512,256,64])
    - kernels  : kernel size of each block (e.g. [3,5,9,7,11,7,11]) — must match the number of layers
    - dropout  : dropout rate after each block
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        channels: Sequence[int] = (128, 128, 64),
        kernels: Sequence[int]  = (9, 5, 3),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernels) and len(channels) > 0, \
            "channels and kernels must have the same, non-zero length."

        self.in_perm  = _BTF_to_BFT()
        self.out_perm = _BFT_to_BTF()

        blocks = []
        c_in = in_dim
        for c, k in zip(channels, kernels):
            k = int(k)
            # length-preserving padding (exact for odd kernels, approximate for even)
            pad = (k - 1) // 2
            blocks += [
                nn.Conv1d(c_in, c, kernel_size=k, padding=pad, bias=True),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            ]
            c_in = c

        # final 1x1 convolution maps to the output channels
        blocks += [nn.Conv1d(c_in, out_dim, kernel_size=1, padding=0, bias=True)]
        self.backbone = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.in_perm(x)          # (B, F, T)
        y = self.backbone(x)         # (B, C, T)
        y = self.out_perm(y)         # (B, T, C)
        return y
