#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tune_transformer.py

import os
import argparse
import random
import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from matlab_data_prep import load_all_mat_data
from utils_and_modules import (
    set_seed, build_dataset_from_mat, make_fold_loaders, train_collect_nrmse
)


# -----------------------------------------------------------
# [모델 정의] Transformer Regressor
# -----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, nhead, num_layers, dim_ff, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_proj(x)

    # -----------------------------------------------------------


# [메인 스크립트]
# -----------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default="config.yaml")
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--max_folds", type=int, default=0)
ap.add_argument("--seed", type=int, default=98)
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--out_csv", type=str, default="tune_transformer_results_400000.csv")
# max_params 인자는 이제 코드 내부 로직(40~60만)이 우선하므로 참고용으로만 쓰입니다.
ap.add_argument("--max_params", type=int, default=400000)
args = ap.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dset = cfg.get("dataset", {})
data = cfg.get("data", {})
run = cfg.get("run", {})

# 타겟 파라미터 범위 설정
MIN_PARAMS_TARGET = 300000
MAX_PARAMS_TARGET = args.max_params

dev = torch.device("cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

print("\n[INFO] loading .mat ...")
grf, kin, mdl = load_all_mat_data(
    grf_path=data.get("grf_mat", ""), grf_struct=data.get("grf_struct", "stride_grf"),
    kin_path=data.get("kin_mat", ""), kin_struct=data.get("kin_struct", "stride_kinematics_arm"),
    modeling_path=data.get("modeling_mat", ""), modeling_struct=data.get("modeling_struct", "stride_modeling"),
)

tv_sessions = dset.get("train_val_sessions", ["ss2"])
te_sessions = dset.get("test_sessions", ["ss2"])
all_sessions = sorted(list(set(tv_sessions) | set(te_sessions)))

print("\n[INFO] assembling X,y ...")
X, y, sample_info = build_dataset_from_mat(
    grf=grf,
    kin=kin,
    modeling=mdl,
    sessions=all_sessions,
    grf_sides=dset.get("grf_sides", ["Total"]),
    grf_axes=dset.get("grf_axes", ["Fx", "Fz"]),
    body_parts=dset.get("body_parts", ["L_Wrist"]),

    axes_pos=dset.get("axes_pos", []),
    axes_vel=dset.get("axes_vel", []),
    axes_acc=dset.get("axes_acc", []),
    axes_imu_acc_local=dset.get("axes_imu_acc_local", []),
    axes_imu_acc_global=dset.get("axes_imu_acc_global", []),
    axes_imu_gyro_local=dset.get("axes_imu_gyro_local", []),
    axes_imu_gyro_global=dset.get("axes_imu_gyro_global", []),

    add_stride_duration_scalar=bool(dset.get("add_stride_duration_scalar", True)),
    window_mode=dset.get("window_mode", "percent"),
    win_pct=tuple(dset.get("win_pct", [0.0, 100.0])),
    win_time=tuple(dset.get("win_time", [0.0, 0.5])),

    use_modeling_input=bool(dset.get("use_modeling_input", False)),
    modeling_points_to_use=dset.get("modeling_points_to_use", []),
    modeling_fields_to_use=dset.get("modeling_fields_to_use", []),
)

N, T, F = X.shape
C = y.shape[-1]
idx_tv = [i for i, it in enumerate(sample_info) if str(it.get("session")) in set(map(str, tv_sessions))]
X_tv, y_tv = X[np.array(idx_tv)], y[np.array(idx_tv)]
info_tv = [sample_info[i] for i in idx_tv]
subs = sorted(list({str(s.get("subject")) for s in info_tv}))
if args.max_folds > 0: subs = subs[:args.max_folds]

# ---- Transformer Search Space (40만~60만 개를 위해 스케일 상향) ----
# d_model_list = [128, 144, 160, 192, 256]  # 차원을 좀 더 키움
# nhead_list = [2, 4, 8]
# num_layers_list = [3, 4, 5, 6]  # 레이어를 좀 더 깊게
# dim_ff_mult_list = [2, 4]
#
# dropout_list = [0.1, 0.2]
# lr_list = [5e-5, 1e-4, 3e-4]
# batch_list = [32, 64]
# wd_list = [1e-4, 1e-3]


# ---- Transformer Search Space (30만~40만 개를 위해 스케일 상향) ----
d_model_list = [96, 112, 128]       # 제곱으로 파라미터가 늘어나므로 값을 낮춤
nhead_list = [4, 8]                 # d_model의 약수여야 함 (96, 112, 128 모두 8로 나누어짐)
num_layers_list = [3, 4, 5]         # d_model이 작아졌으므로 5층까지 고려 가능
dim_ff_mult_list = [2, 3, 4]        # FFN 크기 비율 (파라미터 수 미세 조정용)

# 2. 학습 파라미터 (기존과 유사)
dropout_list = [0.1, 0.2]
lr_list = [5e-5, 1e-4, 5e-4]        # 모델이 작아지면 LR을 약간 높여도 되는 경향이 있음
batch_list = [32, 64]
wd_list = [1e-4, 1e-3]


out_csv = args.out_csv
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
csv_file = open(out_csv, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
# [수정 코드] 뒤에 "dropout", "wd" 추가
writer.writerow([
    "trial", "mean_val_nrmse", "std_val_nrmse", "params",
    "d_model", "nhead", "layers", "dim_ff", "lr", "batch",
    "dropout", "wd"  # <--- 여기 추가됨
])
random.seed(args.seed)
np.random.seed(args.seed)

print(f"[TUNE] Start Transformer Tuning (Target Params: {MIN_PARAMS_TARGET} ~ {MAX_PARAMS_TARGET})")

for trial in range(1, args.trials + 1):
    trial_seed = int(args.seed + trial * 1000)
    set_seed(trial_seed)

    # 1. Sampling (조건 만족할 때까지 반복)
    attempt = 0
    while True:
        attempt += 1
        d_model = random.choice(d_model_list)
        nhead = random.choice(nhead_list)
        if d_model % nhead != 0: continue

        layers = random.choice(num_layers_list)
        dim_ff = d_model * random.choice(dim_ff_mult_list)
        dropout = random.choice(dropout_list)

        lr = random.choice(lr_list)
        bs = random.choice(batch_list)
        wd = random.choice(wd_list)

        tmp_model = TransformerRegressor(F, C, d_model, nhead, layers, dim_ff, dropout).cpu()
        params_count = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        del tmp_model

        # ★ 수정된 조건: 40만 ~ 60만 사이만 통과
        if MIN_PARAMS_TARGET <= params_count <= MAX_PARAMS_TARGET:
            break

        if attempt > 1000:  # 범위를 좁혔으므로 시도 횟수를 늘림
            print(f"[WARN] Failed to find config in range {MIN_PARAMS_TARGET}-{MAX_PARAMS_TARGET} (Trial {trial})")
            # 실패 시 범위 밖이라도 그냥 실행할지, 건너뛸지 결정 (여기선 건너뜀)
            params_count = -1
            break

    if params_count == -1:
        continue

    # 2. Training
    fold_scores = []
    print(
        f"\n[TRIAL {trial}] Params={params_count}, d_model={d_model}, Head={nhead}, Lay={layers}, FF={dim_ff}, LR={lr}")

    for fi, val_sub in enumerate(subs, 1):
        set_seed(trial_seed + fi)
        tr_loader, vl_loader, in_F, out_C, _, _ = make_fold_loaders(
            X_tv, y_tv, info_tv, val_subject=val_sub, batch_size=bs, normalize=True
        )

        model = TransformerRegressor(in_F, out_C, d_model, nhead, layers, dim_ff, dropout).to(dev)
        hist = train_collect_nrmse(model, tr_loader, vl_loader, epochs=args.epochs, lr=lr, weight_decay=wd, device=dev,
                                   add_graph_once=False)

        best = hist.get("best_nrmse", np.nan)
        fold_scores.append(best)
        print(f" - Fold {fi} ({val_sub}): {best:.4f}")
        del model

    mean_val = np.mean(fold_scores)
    std_val = np.std(fold_scores)

    # [수정 코드] 뒤에 dropout, wd 변수 추가
    writer.writerow([
        trial, mean_val, std_val, params_count,
        d_model, nhead, layers, dim_ff, lr, bs,
        dropout, wd  # <--- 여기 추가됨
    ])
    csv_file.flush()

csv_file.close()
print("[DONE] Transformer Tuning Finished.")