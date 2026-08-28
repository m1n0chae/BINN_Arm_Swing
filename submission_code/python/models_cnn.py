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
    멀티 레이어 · 멀티 커널 1D CNN
    - 입력  : (B, T, F)
    - 출력  : (B, T, C)
    - channels : 각 블록의 출력 채널 수 (예: [256,128,128,512,512,256,64])
    - kernels  : 각 블록의 커널 크기 (예: [3,5,9,7,11,7,11])  ← 레이어 수와 반드시 동일 길이
    - dropout  : 각 블록 뒤의 dropout 비율
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
            "channels와 kernels 길이는 같아야 하며 1 이상이어야 합니다."

        self.in_perm  = _BTF_to_BFT()
        self.out_perm = _BFT_to_BTF()

        blocks = []
        c_in = in_dim
        for c, k in zip(channels, kernels):
            k = int(k)
            # 길이 보존 패딩 (홀수 커널에 대해 정확히 보존, 짝수도 근사 보존)
            pad = (k - 1) // 2
            blocks += [
                nn.Conv1d(c_in, c, kernel_size=k, padding=pad, bias=True),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            ]
            c_in = c

        # 최종 1x1 컨볼루션으로 출력 채널 매핑
        blocks += [nn.Conv1d(c_in, out_dim, kernel_size=1, padding=0, bias=True)]
        self.backbone = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.in_perm(x)          # (B, F, T)
        y = self.backbone(x)         # (B, C, T)
        y = self.out_perm(y)         # (B, T, C)
        return y
