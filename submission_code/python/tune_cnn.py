#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tune_cnn.py

import os
import argparse
import random
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from matlab_data_prep import load_all_mat_data
from utils_and_modules import (
    set_seed,
    build_dataset_from_mat,
    make_fold_loaders,
    train_collect_nrmse,
)


# -----------------------------------------------------------
# [모델 정의] 1D CNN Regressor
# -----------------------------------------------------------
class CNNRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, channels, kernels, dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_dim

        if len(channels) != len(kernels):
            raise ValueError("channels and kernels must have same length")

        for c, k in zip(channels, kernels):
            pad = (k - 1) // 2
            layers.append(nn.Conv1d(c_in, c, kernel_size=k, padding=pad))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c_in = c

        # Final prediction layer
        layers.append(nn.Conv1d(c_in, out_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        # y: (B, C, T) -> (B, T, C)
        return y.transpose(1, 2)


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
ap.add_argument("--out_csv", type=str, default="tune_cnn_results.csv")
# max_params 인자는 코드 내부 로직(25~35만)이 우선하므로 참고용
ap.add_argument("--max_params", type=int, default=350000)
args = ap.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dset = cfg.get("dataset", {})
data = cfg.get("data", {})
run = cfg.get("run", {})

# 타겟 파라미터 범위 설정
MIN_PARAMS_TARGET = 250000
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
# 에러 수정된 호출부
X, y, sample_info = build_dataset_from_mat(
    grf=grf,
    kin=kin,
    modeling=mdl,
    sessions=all_sessions,
    grf_sides=dset.get("grf_sides", ["Total"]),
    grf_axes=dset.get("grf_axes", ["Fx", "Fz"]),
    body_parts=dset.get("body_parts", ["L_Wrist"]),

    # --- [Missing args Fixed] ---
    axes_pos=dset.get("axes_pos", []),
    axes_vel=dset.get("axes_vel", []),
    axes_acc=dset.get("axes_acc", []),
    axes_imu_acc_local=dset.get("axes_imu_acc_local", []),
    axes_imu_acc_global=dset.get("axes_imu_acc_global", []),
    axes_imu_gyro_local=dset.get("axes_imu_gyro_local", []),
    axes_imu_gyro_global=dset.get("axes_imu_gyro_global", []),
    # ----------------------------

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
print(f"[DATA] X={X.shape} y={y.shape} subjects={len(set([s['subject'] for s in sample_info]))}")

idx_tv = [i for i, it in enumerate(sample_info) if str(it.get("session")) in set(map(str, tv_sessions))]
X_tv, y_tv = X[np.array(idx_tv)], y[np.array(idx_tv)]
info_tv = [sample_info[i] for i in idx_tv]
subs = sorted(list({str(s.get("subject")) for s in info_tv}))
if args.max_folds > 0: subs = subs[:args.max_folds]

# ---- CNN Search Space (25만~35만 개를 위해 채널 상향) ----
channel_configs = [
    [128, 128, 64],
    [256, 128, 64],
    [128, 256, 128],
    [64, 128, 256, 128],
    [128, 256, 256, 128],  # 더 큰 모델 추가
]
kernel_configs = [
    [7, 5, 3],
    [5, 5, 3],
    [9, 5, 3],
    [7, 5, 3, 3],
]
dropout_list = [0.1, 0.2, 0.3]
lr_list = [1e-4, 5e-4, 1e-3]
batch_list = [16, 32, 64]
wd_list = [1e-5, 1e-4]

out_csv = args.out_csv
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
csv_file = open(out_csv, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow([
    "trial", "mean_val_nrmse", "std_val_nrmse", "params",
    "channels", "kernels", "dropout", "lr", "batch", "wd"
])
random.seed(args.seed)
np.random.seed(args.seed)

print(f"[TUNE] Start CNN Tuning (Target Params: {MIN_PARAMS_TARGET} ~ {MAX_PARAMS_TARGET})")

for trial in range(1, args.trials + 1):
    trial_seed = int(args.seed + trial * 1000)
    set_seed(trial_seed)

    # 1. Random Sampling (조건 만족할 때까지 반복)
    attempt = 0
    while True:
        attempt += 1
        channels = random.choice(channel_configs)
        compatible_kernels = [k for k in kernel_configs if len(k) == len(channels)]
        if not compatible_kernels: continue

        kernels = random.choice(compatible_kernels)
        dropout = random.choice(dropout_list)
        lr = random.choice(lr_list)
        bs = random.choice(batch_list)
        wd = random.choice(wd_list)

        tmp_model = CNNRegressor(F, C, channels, kernels, dropout).cpu()
        params_count = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        del tmp_model

        # ★ 수정된 조건: 25만 ~ 35만 사이만 통과
        if MIN_PARAMS_TARGET <= params_count <= MAX_PARAMS_TARGET:
            break

        if attempt > 1000:
            print(f"[WARN] Failed to find config in range {MIN_PARAMS_TARGET}-{MAX_PARAMS_TARGET} (Trial {trial})")
            params_count = -1
            break

    if params_count == -1:
        continue

    # 2. Cross-Validation
    fold_scores = []
    print(f"\n[TRIAL {trial}] Params={params_count}, Ch={channels}, K={kernels}, Drop={dropout}, LR={lr}")

    for fi, val_sub in enumerate(subs, 1):
        set_seed(trial_seed + fi)
        tr_loader, vl_loader, in_F, out_C, _, _ = make_fold_loaders(
            X_tv, y_tv, info_tv, val_subject=val_sub, batch_size=bs, normalize=True
        )

        model = CNNRegressor(in_F, out_C, channels, kernels, dropout).to(dev)
        hist = train_collect_nrmse(model, tr_loader, vl_loader, epochs=args.epochs, lr=lr, weight_decay=wd, device=dev,
                                   add_graph_once=False)

        best = hist.get("best_nrmse", np.nan)
        fold_scores.append(best)
        print(f" - Fold {fi} ({val_sub}): {best:.4f}")
        del model

    mean_val = np.mean(fold_scores)
    std_val = np.std(fold_scores)

    writer.writerow([
        trial, mean_val, std_val, params_count,
        str(channels), str(kernels), dropout, lr, bs, wd
    ])
    csv_file.flush()

csv_file.close()
print("[DONE] CNN Tuning Finished.")