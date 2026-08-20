#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import csv
import time

import numpy as np
import torch
import yaml

from matlab_data_prep import load_all_mat_data
from utils_and_modules import (
    set_seed,
    build_dataset_from_mat,
    make_fold_loaders,
    train_collect_nrmse,
)

from models_physres_cnn_attn import PhysResidualCNNAttnRegressor


ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default="config.yaml")
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--max_folds", type=int, default=0)       # 0이면 전부
ap.add_argument("--seed", type=int, default=98)
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--out_csv", type=str, default="tune_physres_results.csv")
ap.add_argument("--max_params", type=int, default=150000) # ★ 파라미터 상한
args = ap.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

run = cfg.get("run", {}) or {}
dset = cfg.get("dataset", {}) or {}
data = cfg.get("data", {}) or {}

MAX_PARAMS = int(args.max_params)

if not bool(dset.get("use_modeling_input", False)):
    raise RuntimeError("config.yaml에서 dataset.use_modeling_input: true 로 바꾸셔야 sacrum(modeling)이 X에 들어갑니다.")

mp = dset.get("modeling_points_to_use", [])
mf = dset.get("modeling_fields_to_use", [])
if not (isinstance(mp, list) and len(mp) > 0 and isinstance(mf, list) and len(mf) > 0):
    raise RuntimeError("dataset.modeling_points_to_use / modeling_fields_to_use 가 비어있습니다. (sacrum_dim 설정이 불가능)")

sacrum_dim_cfg = len(mf)

wrist_dim_cfg = 0
wrist_dim_cfg += len(dset.get("axes_imu_acc_local", []) or [])
wrist_dim_cfg += len(dset.get("axes_imu_gyro_local", []) or [])
if wrist_dim_cfg <= 0:
    raise RuntimeError("wrist_dim을 계산할 수 없습니다. dataset.axes_imu_acc_local / axes_imu_gyro_local 확인 필요.")

dev = torch.device("cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

print("\n[INFO] loading .mat ...")
grf, kin, mdl = load_all_mat_data(
    grf_path=data.get("grf_mat", ""),
    grf_struct=data.get("grf_struct", "stride_grf"),
    kin_path=data.get("kin_mat", ""),
    kin_struct=data.get("kin_struct", "stride_kinematics_arm"),
    modeling_path=data.get("modeling_mat", ""),
    modeling_struct=data.get("modeling_struct", "stride_modeling"),
)

tv_sessions = dset.get("train_val_sessions", dset.get("sessions", ["ss2"])) or ["ss2"]
te_sessions = dset.get("test_sessions", dset.get("sessions", ["ss2"])) or ["ss2"]
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
    use_modeling_input=True,
    modeling_points_to_use=mp,
    modeling_fields_to_use=mf,
)

N, T, F = X.shape
C = y.shape[-1]
print(f"[DATA] X={X.shape} y={y.shape} subjects={len(set([s['subject'] for s in sample_info]))}")

idx_tv = []
for i, it in enumerate(sample_info):
    if str(it.get("session", "")) in set(map(str, tv_sessions)):
        idx_tv.append(i)

X_tv = X[np.array(idx_tv)]
y_tv = y[np.array(idx_tv)]
info_tv = [sample_info[i] for i in idx_tv]

subs = sorted(list({str(s.get("subject", "")) for s in info_tv}))
if args.max_folds > 0:
    subs = subs[:args.max_folds]

print(f"[TUNE] folds used: {len(subs)}  (max_folds={args.max_folds})")
print(f"[TUNE] wrist_dim={wrist_dim_cfg}, sacrum_dim={sacrum_dim_cfg}, total_in_dim(F)={F}, out_dim(C)={C}")
print(f"[TUNE] max_params <= {MAX_PARAMS}")

# ---- Search space ----
# MAX_PARAMS 크기에 따라 search space를 보수적으로 줄여서
# params_count <= MAX_PARAMS 조건을 만족하는 조합이 충분히 나오도록 설정
if MAX_PARAMS <= 100000:
    # 약 10만 개 이하 파라미터를 목표로 하는 작은 모델 공간
    embed_dim_list = [32, 40, 48]
    attn_heads_list = [1, 2, 4]  # embed_dim % head 조건은 아래에서 체크
    dropout_list = [0.05, 0.10]

    wrist_channels_list = [
        (32, 32),
        (40, 32),
    ]
    wrist_kernels_list = [
        (9, 5),
    ]

    sacrum_channels_list = [
        (16, 16),
        (24, 16),
    ]
    sacrum_kernels_list = [
        (9, 5),
    ]

    decoder_channels_list = [
        (64, 32),
        (80, 40),
        (96, 48),
    ]
    decoder_kernels_list = [
        (5, 3),
    ]

    phys_smooth_kernel_list = [3, 5]

elif MAX_PARAMS <= 150000:
    # 약 15만 개 이하 파라미터를 목표로 하는 중간 규모 모델 공간
    embed_dim_list = [40, 48, 64]
    attn_heads_list = [1, 2, 4]
    dropout_list = [0.05, 0.10, 0.15]

    wrist_channels_list = [
        (32, 32),
        (48, 32),
        (48, 48),
    ]
    wrist_kernels_list = [
        (9, 5),
        (11, 5),
    ]

    sacrum_channels_list = [
        (24, 24),
        (32, 24),
    ]
    sacrum_kernels_list = [
        (9, 5),
    ]

    decoder_channels_list = [
        (96, 48),
        (112, 56),
        (128, 64),
    ]
    decoder_kernels_list = [
        (5, 3),
        (7, 3),
    ]

    phys_smooth_kernel_list = [3, 5, 7]

else:
    # 그 이상일 때는 원래 쓰던 큰 search space를 사용해도 무방
    embed_dim_list = [80, 96, 112, 128]
    attn_heads_list = [1, 2, 4]  # embed_dim이 head로 나누어 떨어져야 함
    dropout_list = [0.05, 0.10, 0.15]

    wrist_channels_list = [(64, 64), (96, 64), (96, 96)]
    wrist_kernels_list  = [(9, 5), (11, 5)]
    sacrum_channels_list = [(32, 32), (48, 32)]
    sacrum_kernels_list  = [(9, 5)]

    decoder_channels_list = [(128, 64), (144, 72), (160, 96), (192, 128)]
    decoder_kernels_list  = [(5, 3), (7, 3)]

    phys_smooth_kernel_list = [3, 5, 7]

# 학습 관련 하이퍼파라미터는 그대로 유지
lr_list = [3e-5, 5e-5, 1e-4]
wd_list = [0.0, 1e-5, 1e-4]
batch_list = [16, 32, 64]


out_csv = args.out_csv
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
csv_file = open(out_csv, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow([
    "trial", "mean_val_nrmse", "std_val_nrmse",
    "params",
    "epochs", "batch", "lr", "weight_decay",
    "embed_dim", "attn_heads", "dropout",
    "wrist_channels", "wrist_kernels",
    "sacrum_channels", "sacrum_kernels",
    "decoder_channels", "decoder_kernels",
    "phys_smooth_kernel",
])

all_results = []

random.seed(args.seed)
np.random.seed(args.seed)

t0 = time.time()

for trial in range(1, args.trials + 1):
    trial_seed = int(args.seed + trial * 1000)
    set_seed(trial_seed)

    # === params<=MAX_PARAMS 만족할 때까지 재샘플링 ===
    attempt = 0
    while True:
        attempt += 1

        embed_dim = random.choice(embed_dim_list)
        attn_heads = random.choice(attn_heads_list)
        tries = 0
        while (embed_dim % attn_heads) != 0:
            attn_heads = random.choice(attn_heads_list)
            tries += 1
            if tries > 50:
                attn_heads = 1
                break

        dropout = random.choice(dropout_list)

        wrist_channels = random.choice(wrist_channels_list)
        wrist_kernels  = random.choice(wrist_kernels_list)
        sacrum_channels = random.choice(sacrum_channels_list)
        sacrum_kernels  = random.choice(sacrum_kernels_list)

        decoder_channels = random.choice(decoder_channels_list)
        decoder_kernels  = random.choice(decoder_kernels_list)

        phys_smooth_kernel = random.choice(phys_smooth_kernel_list)

        lr = float(random.choice(lr_list))
        wd = float(random.choice(wd_list))
        batch_size = int(random.choice(batch_list))

        # CPU에서 모델 1회 생성 후 params 체크
        tmp = PhysResidualCNNAttnRegressor(
            in_dim=F, out_dim=C,
            wrist_dim=wrist_dim_cfg,
            sacrum_dim=sacrum_dim_cfg,

            wrist_channels=wrist_channels,
            wrist_kernels=wrist_kernels,
            sacrum_channels=sacrum_channels,
            sacrum_kernels=sacrum_kernels,

            embed_dim=int(embed_dim),
            attn_heads=int(attn_heads),

            decoder_channels=decoder_channels,
            decoder_kernels=decoder_kernels,

            phys_smooth_kernel=int(phys_smooth_kernel),
            phys_scale_init=1.0,

            dropout=float(dropout),
            include_phys_in_residual=True,
            extra_to_wrist=True,
        ).cpu()

        params_count = 0
        for p in tmp.parameters():
            if p.requires_grad:
                params_count += p.numel()
        del tmp

        if params_count <= MAX_PARAMS:
            break

        if attempt >= 300:
            raise RuntimeError(
                f"[TRIAL {trial}] 300번 재샘플링했는데도 params<={MAX_PARAMS} 조건을 만족하는 조합을 못 찾았습니다. "
                f"search space를 줄이거나 max_params를 올려야 합니다."
            )

    fold_scores = []

    print(
        f"\n[TRIAL {trial:03d}/{args.trials}] params={params_count} embed={embed_dim} heads={attn_heads} "
        f"dec={decoder_channels} drop={dropout} lr={lr} wd={wd} bs={batch_size} (attempt={attempt})"
    )

    for fi, val_sub in enumerate(subs, 1):
        set_seed(trial_seed + fi)

        tr_loader, vl_loader, in_F, out_C, seq_len_T, scaler = make_fold_loaders(
            X_tv, y_tv, info_tv,
            val_subject=val_sub,
            batch_size=batch_size,
            n_workers=int(run.get("num_workers", 2)),
            normalize=True
        )

        model = PhysResidualCNNAttnRegressor(
            in_dim=in_F, out_dim=out_C,
            wrist_dim=wrist_dim_cfg,
            sacrum_dim=sacrum_dim_cfg,

            wrist_channels=wrist_channels,
            wrist_kernels=wrist_kernels,
            sacrum_channels=sacrum_channels,
            sacrum_kernels=sacrum_kernels,

            embed_dim=int(embed_dim),
            attn_heads=int(attn_heads),

            decoder_channels=decoder_channels,
            decoder_kernels=decoder_kernels,

            phys_smooth_kernel=int(phys_smooth_kernel),
            phys_scale_init=1.0,

            dropout=float(dropout),
            include_phys_in_residual=True,
            extra_to_wrist=True,
        ).to(dev)

        hist = train_collect_nrmse(
            model, tr_loader, vl_loader,
            epochs=int(args.epochs),
            lr=float(lr),
            weight_decay=float(wd),
            device=dev,
            log_dir=None,        # 튜닝은 TB trace 끔
            add_graph_once=False
        )

        best = float(hist.get("best_nrmse", np.nan))
        fold_scores.append(best)
        print(f"  - fold {fi}/{len(subs)} val_sub={val_sub} best_nrmse={best:.6f}")

        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    mean_val = float(np.mean(fold_scores)) if fold_scores else float("nan")
    std_val  = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else float("nan")

    row = [
        trial, mean_val, std_val,
        params_count,
        int(args.epochs), batch_size, lr, wd,
        embed_dim, attn_heads, dropout,
        str(wrist_channels), str(wrist_kernels),
        str(sacrum_channels), str(sacrum_kernels),
        str(decoder_channels), str(decoder_kernels),
        phys_smooth_kernel,
    ]
    writer.writerow(row)
    csv_file.flush()

    all_results.append({
        "trial": trial,
        "mean_val_nrmse": mean_val,
        "std_val_nrmse": std_val,
        "params": params_count,
        "epochs": int(args.epochs),
        "batch": batch_size,
        "lr": lr,
        "wd": wd,
        "embed_dim": embed_dim,
        "attn_heads": attn_heads,
        "dropout": dropout,
        "wrist_channels": wrist_channels,
        "wrist_kernels": wrist_kernels,
        "sacrum_channels": sacrum_channels,
        "sacrum_kernels": sacrum_kernels,
        "decoder_channels": decoder_channels,
        "decoder_kernels": decoder_kernels,
        "phys_smooth_kernel": phys_smooth_kernel,
    })

csv_file.close()

all_results = sorted(all_results, key=lambda d: d["mean_val_nrmse"])
print("\n========== TOP 5 (by mean VAL NRMSE) ==========")
for i in range(min(5, len(all_results))):
    r = all_results[i]
    print(f"[{i+1}] trial={r['trial']} mean={r['mean_val_nrmse']:.6f} std={r['std_val_nrmse']:.6f} params={r['params']}")
    print(f"    lr={r['lr']} wd={r['wd']} bs={r['batch']}  embed={r['embed_dim']} heads={r['attn_heads']} drop={r['dropout']}")
    print(f"    wrist={r['wrist_channels']} k={r['wrist_kernels']}  sacrum={r['sacrum_channels']}  "
          f"dec={r['decoder_channels']} k={r['decoder_kernels']}  smooth={r['phys_smooth_kernel']}")

print(f"\n[OK] saved results -> {out_csv}")
print(f"[TIME] {time.time() - t0:.1f} sec")
