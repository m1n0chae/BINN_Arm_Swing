import torch
from torch import nn


class PhysResidualCNNAttnRegressorAblation(nn.Module):
    """
    Biomechanics-Informed Neural Network (BINN) and its ablation variants.

    Mode names as used in the paper:
      - 'proposed'  : Proposed BINN (full architecture).
                      Cross-Attention + CoM-to-GRF Mapper output passed directly to the
                      GRF Regression Head.
      - 'no_attn'   : w/o Cross-Attention.  The CoM pathway and the CoM-to-GRF Mapper are
                      retained, but attention-based fusion is removed.
      - 'attn_only' : w/o CoM-to-GRF Mapper.  CoM information reaches the head only
                      through Cross-Attention.

    'w/o Biomechanics-informed Architecture' is not a mode of this class; it is a plain
    CNN with the CoM channels concatenated at the input (model type 'cnn', see
    configs/ablation_wo_biomech_arch.yaml).

    Legacy aliases, kept so that older configs and saved checkpoints still load:
      'binn' == 'no_physics' == 'proposed'
      'no_converter', 'phys_to_attn', 'no_physics_enc', 'no_attn_enc' are additional
      exploratory variants that are not reported in the paper.

    Regression head input width:
      - proposed:          w_feat(d) + h_attn(d) + y_CoM(C)     = 82  (d=40, C=2)
      - no_attn:           w_feat(d) + y_CoM(C)                 = 42
      - attn_only:         w_feat(d) + h_attn(d)                = 80
      - no_converter:      w_feat(d) + h_attn(d) + sacrum_raw(S)= 82  (S=sacrum_dim=2)
      - phys_to_attn:      w_feat(d) + h_attn(d) + y_CoM(C)     = 82
      - no_physics_enc:    w_feat(d) + h_attn(d) + phys_feat(d) = 120
      - no_attn_enc:       w_feat(d) + phys_feat(d)             = 80

    Regressor 입력 정리:
      - proposed:       w_feat(d) + h_attn(d) + y_phys(C)      = 82  (d=40, C=2)
      - no_attn:        w_feat(d) + y_phys(C)                  = 42
      - no_converter:   w_feat(d) + h_attn(d) + sacrum_raw(S)  = 82  (S=sacrum_dim=2)
      - attn_only:      w_feat(d) + h_attn(d)                  = 80
      - phys_to_attn:   w_feat(d) + h_attn(d) + y_phys(C)      = 82
      - no_physics_enc: w_feat(d) + h_attn(d) + phys_feat(d)   = 120
      - no_attn_enc:    w_feat(d) + phys_feat(d)                = 80
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            wrist_dim: int,
            sacrum_dim: int,
            ablation_mode: str = 'proposed',
            *,
            # feature extractors
            wrist_channels=(64, 64),
            wrist_kernels=(9, 5),
            sacrum_channels=(32, 32),
            sacrum_kernels=(9, 5),
            embed_dim: int = 64,
            attn_heads: int = 1,
            # physics encoder (for *_enc modes)
            phys_enc_channels=(32, 32),
            phys_enc_kernels=(5, 3),
            # GRF regression head
            decoder_channels=(128, 64),
            decoder_kernels=(5, 3),
            # CoM-to-GRF converter
            phys_smooth_kernel: int = 5,
            phys_scale_init: float = 1.0,
            # misc
            dropout: float = 0.1,
            include_phys_in_residual: bool = True,
            extra_to_wrist: bool = True,
    ) -> None:
        super().__init__()

        # 설정 저장
        # 'proposed' is the canonical name of the full architecture (the proposed BINN).
        # 'binn' and the legacy internal name 'no_physics' are accepted as aliases so that
        # older configs and previously saved checkpoints still load unchanged.
        MODE_ALIASES = {'binn': 'proposed', 'no_physics': 'proposed'}
        self.ablation_mode = MODE_ALIASES.get(ablation_mode.lower(), ablation_mode.lower())
        valid_modes = ['proposed', 'no_attn', 'no_converter', 'attn_only', 'phys_to_attn', 'no_physics_enc', 'no_attn_enc']
        if self.ablation_mode not in valid_modes:
            raise ValueError(
                f"ablation_mode must be one of {sorted(set(valid_modes) | set(MODE_ALIASES))}")

        # 기존 설정
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.wrist_dim = int(wrist_dim)
        self.sacrum_dim = int(sacrum_dim)
        self.extra_dim = int(in_dim - wrist_dim - sacrum_dim)
        self.extra_to_wrist = bool(extra_to_wrist)

        # *_enc 모드 플래그
        self.use_phys_encoder = self.ablation_mode in ('no_physics_enc', 'no_attn_enc')

        # y_phys를 regressor 입력에 포함할지 (non-enc 모드에서만)
        if self.use_phys_encoder:
            self.include_phys_in_residual = False
        else:
            self.include_phys_in_residual = bool(include_phys_in_residual)

        # -------------------------
        # (1) CoM-to-GRF Converter
        # -------------------------
        k = int(phys_smooth_kernel)
        if k <= 1:
            self.phys_smooth = nn.Identity()
        else:
            pad = (k - 1) // 2
            self.phys_smooth = nn.Conv1d(
                self.sacrum_dim, self.sacrum_dim, kernel_size=k,
                padding=pad, groups=self.sacrum_dim, bias=False,
            )
            with torch.no_grad():
                w = torch.zeros_like(self.phys_smooth.weight)
                w[:] = 1.0 / float(k)
                self.phys_smooth.weight.copy_(w)

        self.phys_map = nn.Conv1d(self.sacrum_dim, self.out_dim, kernel_size=1, padding=0, bias=True)
        self.phys_scale = nn.Parameter(torch.full((1, 1, self.out_dim), float(phys_scale_init)))

        # -------------------------
        # (1-b) Physics Encoder (for *_enc modes)
        #   raw sacrum(CoM) input → embed_dim 차원 특징으로 변환
        # -------------------------
        if self.use_phys_encoder:
            p_blocks = []
            c_in = self.sacrum_dim
            for c, kk in zip(phys_enc_channels, phys_enc_kernels):
                kk, pad = int(kk), (int(kk) - 1) // 2
                p_blocks += [
                    nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad),
                    nn.BatchNorm1d(int(c)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
                c_in = int(c)
            p_blocks += [
                nn.Conv1d(c_in, int(embed_dim), 1),
                nn.BatchNorm1d(int(embed_dim)),
                nn.ReLU(inplace=True),
            ]
            self.phys_enc = nn.Sequential(*p_blocks)
        else:
            self.phys_enc = None

        # -------------------------
        # (2) Feature Extractors
        # -------------------------
        wrist_in = self.wrist_dim + (self.extra_dim if self.extra_to_wrist else 0)

        # Wrist Feature Extractor
        w_blocks = []
        c_in = wrist_in
        for c, kk in zip(wrist_channels, wrist_kernels):
            kk, pad = int(kk), (int(kk) - 1) // 2
            w_blocks += [
                nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad),
                nn.BatchNorm1d(int(c)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c_in = int(c)
        w_blocks += [
            nn.Conv1d(c_in, int(embed_dim), 1),
            nn.BatchNorm1d(int(embed_dim)),
            nn.ReLU(inplace=True),
        ]
        self.wrist_enc = nn.Sequential(*w_blocks)

        # CoM Feature Extractor (cross-attention 사용하는 모드에서만 필요)
        self.use_sacrum_enc = self.ablation_mode in ('proposed', 'no_converter', 'attn_only', 'phys_to_attn', 'no_physics_enc')
        if self.use_sacrum_enc:
            s_blocks = []
            c_in = self.sacrum_dim
            for c, kk in zip(sacrum_channels, sacrum_kernels):
                kk, pad = int(kk), (int(kk) - 1) // 2
                s_blocks += [
                    nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad),
                    nn.BatchNorm1d(int(c)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
                c_in = int(c)
            s_blocks += [
                nn.Conv1d(c_in, int(embed_dim), 1),
                nn.BatchNorm1d(int(embed_dim)),
                nn.ReLU(inplace=True),
            ]
            self.sacrum_enc = nn.Sequential(*s_blocks)
        else:
            self.sacrum_enc = None

        # -------------------------
        # (3) Cross-Attention (cross-attention 사용하는 모드에서만)
        # -------------------------
        if self.use_sacrum_enc:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=int(embed_dim), num_heads=int(attn_heads),
                dropout=float(dropout), batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(int(embed_dim))
        else:
            self.cross_attn = None
            self.attn_norm = None

        # -------------------------
        # (4) GRF Regression Head
        # -------------------------
        if self.ablation_mode == 'proposed':
            # w_feat(d) + h_attn(d) + y_phys(out_dim)
            fused_in = int(embed_dim) + int(embed_dim) + self.out_dim
        elif self.ablation_mode == 'no_attn':
            # w_feat(d) + y_phys(out_dim)
            fused_in = int(embed_dim) + self.out_dim
        elif self.ablation_mode == 'no_converter':
            # w_feat(d) + h_attn(d) + sacrum_raw(sacrum_dim)
            fused_in = int(embed_dim) + int(embed_dim) + self.sacrum_dim
        elif self.ablation_mode == 'attn_only':
            # w_feat(d) + h_attn(d)
            fused_in = int(embed_dim) + int(embed_dim)
        elif self.ablation_mode == 'phys_to_attn':
            # w_feat(d) + h_attn(d) + y_phys(out_dim)
            fused_in = int(embed_dim) + int(embed_dim) + self.out_dim
        elif self.ablation_mode == 'no_physics_enc':
            # w_feat(d) + h_attn(d) + phys_feat(d)
            fused_in = int(embed_dim) + int(embed_dim) + int(embed_dim)
        elif self.ablation_mode == 'no_attn_enc':
            # w_feat(d) + phys_feat(d)
            fused_in = int(embed_dim) + int(embed_dim)

        if self.extra_dim > 0 and (not self.extra_to_wrist):
            fused_in += self.extra_dim

        d_blocks = []
        c_in = fused_in
        for c, kk in zip(decoder_channels, decoder_kernels):
            kk, pad = int(kk), (int(kk) - 1) // 2
            d_blocks += [
                nn.Conv1d(c_in, int(c), kernel_size=kk, padding=pad),
                nn.BatchNorm1d(int(c)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c_in = int(c)
        d_blocks += [nn.Conv1d(c_in, self.out_dim, 1)]
        self.decoder = nn.Sequential(*d_blocks)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Split
        wrist = x[:, :, :self.wrist_dim]
        sacrum = x[:, :, self.wrist_dim:self.wrist_dim + self.sacrum_dim]
        extra = x[:, :, self.wrist_dim + self.sacrum_dim:] if self.extra_dim > 0 else None

        if self.extra_dim > 0 and self.extra_to_wrist:
            wrist_in = torch.cat([wrist, extra], dim=-1)
        else:
            wrist_in = wrist

        # 1. CoM-to-GRF Converter
        s = sacrum.transpose(1, 2).contiguous()
        s_sm = self.phys_smooth(s)
        y_phys = self.phys_map(s_sm).transpose(1, 2).contiguous()
        y_phys = y_phys * self.phys_scale  # (B, T, out_dim)

        # 1-b. Physics Encoder (*_enc modes): raw sacrum → embed_dim
        if self.use_phys_encoder:
            phys_feat = self.phys_enc(
                sacrum.transpose(1, 2).contiguous()
            ).transpose(1, 2).contiguous()  # (B, T, embed_dim)

        # 2. Wrist Feature Extractor
        w_feat = self.wrist_enc(wrist_in.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # 3. CoM Feature Extractor + Cross-Attention (해당 모드에서만)
        if self.use_sacrum_enc:
            # phys_to_attn: CoM FE에 raw sacrum 대신 y_phys를 입력
            if self.ablation_mode == 'phys_to_attn':
                s_feat = self.sacrum_enc(y_phys.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                s_feat = self.sacrum_enc(sacrum.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            attn_out, _ = self.cross_attn(w_feat, s_feat, s_feat, need_weights=False)
            h_attn = self.attn_norm(w_feat + attn_out)

        # 4. Feature Fusion
        if self.ablation_mode == 'proposed':
            # w_feat + h_attn + y_phys
            fused = torch.cat([w_feat, h_attn, y_phys], dim=-1)
        elif self.ablation_mode == 'no_attn':
            # w_feat + y_phys
            fused = torch.cat([w_feat, y_phys], dim=-1)
        elif self.ablation_mode == 'no_converter':
            # w_feat + h_attn + raw sacrum
            fused = torch.cat([w_feat, h_attn, sacrum], dim=-1)
        elif self.ablation_mode == 'attn_only':
            # w_feat + h_attn
            fused = torch.cat([w_feat, h_attn], dim=-1)
        elif self.ablation_mode == 'phys_to_attn':
            # w_feat + h_attn + y_phys
            fused = torch.cat([w_feat, h_attn, y_phys], dim=-1)
        elif self.ablation_mode == 'no_physics_enc':
            # w_feat + h_attn + phys_feat
            fused = torch.cat([w_feat, h_attn, phys_feat], dim=-1)
        elif self.ablation_mode == 'no_attn_enc':
            # w_feat + phys_feat
            fused = torch.cat([w_feat, phys_feat], dim=-1)

        if self.extra_dim > 0 and (not self.extra_to_wrist):
            fused = torch.cat([fused, extra], dim=-1)

        # 5. GRF Regression Head
        y_res = self.decoder(fused.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # 6. Final Output
        return y_res

    def get_physics_output(self, x: torch.Tensor) -> torch.Tensor:
        """Physics block(CoM-to-GRF Converter)의 출력 y_phys만 반환"""
        sacrum = x[:, :, self.wrist_dim:self.wrist_dim + self.sacrum_dim]
        s = sacrum.transpose(1, 2).contiguous()
        s_sm = self.phys_smooth(s)
        y_phys = self.phys_map(s_sm).transpose(1, 2).contiguous()
        y_phys = y_phys * self.phys_scale
        return y_phys