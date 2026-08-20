# models_physres_cnn_attn.py
import torch
from torch import nn


class PhysResidualCNNAttnRegressor(nn.Module):
    """
    physics residual + 1D CNN + sacrum-guided cross-attention(1 layer)

    Input : x (B, T, F)
      - x[:, :, :wrist_dim]                         -> wrist features
      - x[:, :, wrist_dim:wrist_dim+sacrum_dim]     -> sacrum features
      - remaining features                          -> extra (optional), by default appended to wrist

    Output: y_hat (B, T, C)

    y_hat = y_phys(sacrum)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        wrist_dim: int,
        sacrum_dim: int,
        *,
        # encoders
        wrist_channels=(64, 64),
        wrist_kernels=(9, 5),
        sacrum_channels=(32, 32),
        sacrum_kernels=(9, 5),
        embed_dim: int = 64,
        attn_heads: int = 1,
        # decoder
        decoder_channels=(128, 64),
        decoder_kernels=(5, 3),
        # physics block
        phys_smooth_kernel: int = 5,
        phys_scale_init: float = 1.0,
        # misc
        dropout: float = 0.1,
        include_phys_in_residual: bool = False,
        extra_to_wrist: bool = True,
    ) -> None:
        super().__init__()

        if wrist_dim <= 0 or sacrum_dim <= 0:
            raise ValueError("wrist_dim and sacrum_dim must be positive integers.")
        if wrist_dim + sacrum_dim > in_dim:
            raise ValueError("wrist_dim + sacrum_dim must be <= in_dim.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.wrist_dim = int(wrist_dim)
        self.sacrum_dim = int(sacrum_dim)
        self.extra_dim = int(in_dim - wrist_dim - sacrum_dim)
        self.extra_to_wrist = bool(extra_to_wrist)
        self.include_phys_in_residual = bool(include_phys_in_residual)

        # -------------------------
        # (1) physics baseline block
        # -------------------------
        k = int(phys_smooth_kernel)
        if k <= 1:
            self.phys_smooth = nn.Identity()
        else:
            pad = (k - 1) // 2
            self.phys_smooth = nn.Conv1d(
                self.sacrum_dim,
                self.sacrum_dim,
                kernel_size=k,
                padding=pad,
                groups=self.sacrum_dim,
                bias=False,
            )
            # init as simple average smoothing
            with torch.no_grad():
                w = torch.zeros_like(self.phys_smooth.weight)  # (S,1,k)
                w[:] = 1.0 / float(k)
                self.phys_smooth.weight.copy_(w)

        self.phys_map = nn.Conv1d(self.sacrum_dim, self.out_dim, kernel_size=1, padding=0, bias=True)
        # per-output-channel scale (broadcasted)
        self.phys_scale = nn.Parameter(torch.full((1, 1, self.out_dim), float(phys_scale_init)))

        # -------------------------
        # (2) CNN encoders
        # -------------------------
        wrist_in = self.wrist_dim + (self.extra_dim if self.extra_to_wrist else 0)

        # wrist encoder: (B,T,Fw) -> (B,T,E)
        w_blocks = []
        c_in = wrist_in
        if len(wrist_channels) != len(wrist_kernels) or len(wrist_channels) == 0:
            raise ValueError("wrist_channels and wrist_kernels must have the same non-zero length.")
        for c, kk in zip(wrist_channels, wrist_kernels):
            kk = int(kk)
            pad = (kk - 1) // 2
            w_blocks += [
                nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad, bias=True),
                nn.BatchNorm1d(int(c)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            c_in = int(c)
        w_blocks += [
            nn.Conv1d(c_in, int(embed_dim), kernel_size=1, padding=0, bias=True),
            nn.BatchNorm1d(int(embed_dim)),
            nn.ReLU(inplace=True),
        ]
        self.wrist_enc = nn.Sequential(*w_blocks)

        # sacrum encoder: (B,T,Fs) -> (B,T,E)
        s_blocks = []
        c_in = self.sacrum_dim
        if len(sacrum_channels) != len(sacrum_kernels) or len(sacrum_channels) == 0:
            raise ValueError("sacrum_channels and sacrum_kernels must have the same non-zero length.")
        for c, kk in zip(sacrum_channels, sacrum_kernels):
            kk = int(kk)
            pad = (kk - 1) // 2
            s_blocks += [
                nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad, bias=True),
                nn.BatchNorm1d(int(c)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            c_in = int(c)
        s_blocks += [
            nn.Conv1d(c_in, int(embed_dim), kernel_size=1, padding=0, bias=True),
            nn.BatchNorm1d(int(embed_dim)),
            nn.ReLU(inplace=True),
        ]
        self.sacrum_enc = nn.Sequential(*s_blocks)

        # -------------------------
        # (3) cross-attention (1 layer)
        # -------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(attn_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(int(embed_dim))

        # -------------------------
        # (4) residual decoder (CNN)
        # -------------------------
        fused_in = int(embed_dim) + int(embed_dim)  # wrist_feat + attn_out
        if self.include_phys_in_residual:
            fused_in += self.out_dim
        if self.extra_dim > 0 and (not self.extra_to_wrist):
            fused_in += self.extra_dimrmfja

        if len(decoder_channels) != len(decoder_kernels) or len(decoder_channels) == 0:
            raise ValueError("decoder_channels and decoder_kernels must have the same non-zero length.")

        d_blocks = []
        c_in = fused_in
        for c, kk in zip(decoder_channels, decoder_kernels):
            kk = int(kk)
            pad = (kk - 1) // 2
            d_blocks += [
                nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad, bias=True),
                nn.BatchNorm1d(int(c)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            c_in = int(c)
        d_blocks += [nn.Conv1d(c_in, self.out_dim, kernel_size=1, padding=0, bias=True)]
        self.decoder = nn.Sequential(*d_blocks)

        # residual head init to 0 -> start from physics-only prediction
        # last = self.decoder[-1]
        # if isinstance(last, nn.Conv1d):
        #     nn.init.zeros_(last.weight)
        #     if last.bias is not None:
        #         nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        if F != self.in_dim:
            raise ValueError(f"Input feature dim mismatch: got {F}, expected {self.in_dim}")

        wrist = x[:, :, : self.wrist_dim]
        sacrum = x[:, :, self.wrist_dim : self.wrist_dim + self.sacrum_dim]
        extra = x[:, :, self.wrist_dim + self.sacrum_dim :] if self.extra_dim > 0 else None

        if self.extra_dim > 0 and self.extra_to_wrist:
            wrist_in = torch.cat([wrist, extra], dim=-1)
        else:
            wrist_in = wrist

        # ---- physics baseline ----
        s = sacrum.transpose(1, 2).contiguous()  # (B, S, T)
        s_sm = self.phys_smooth(s)
        y_phys = self.phys_map(s_sm).transpose(1, 2).contiguous()  # (B, T, C)
        y_phys = y_phys * self.phys_scale

        # ---- encoders ----
        w_feat = self.wrist_enc(wrist_in.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # (B, T, E)
        s_feat = self.sacrum_enc(sacrum.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()   # (B, T, E)

        # ---- sacrum-guided cross-attention (1 layer) ----
        attn_out, _ = self.cross_attn(w_feat, s_feat, s_feat, need_weights=False)
        attn_out = self.attn_norm(w_feat + attn_out)  # residual + norm

        # ---- fuse & residual decode ----
        fused = [w_feat, attn_out]
        if self.include_phys_in_residual:
            fused.append(y_phys)
        if self.extra_dim > 0 and (not self.extra_to_wrist):
            fused.append(extra)
        fused = torch.cat(fused, dim=-1)  # (B, T, F_fused)

        y_res = self.decoder(fused.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # (B, T, C)

        return y_res
