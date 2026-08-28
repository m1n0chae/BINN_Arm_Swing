#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — LOSO / Fixed 학습·평가 엔트리포인트 (NRMSE(range) 전용)
- train/val 세션과 test 세션을 분리해 사용
- 각 fold/FIXED 학습 후 동일 스케일러로 TEST 평가 + 3종 플롯 저장
- 채널별 nRMSE(range)도 VAL/TEST에서 콘솔 출력 및 JSON 저장
- VAL/TEST 평가는 last.pth(최종 epoch) 기준으로만 수행
  (LOSO 에서는 val 폴드가 곧 test 피험자이므로 검증 기준 체크포인트 선택은
   test 셋 선택이 된다. 따라서 best 체크포인트는 저장도 로드도 하지 않는다)
- per-sample NRMSE(range) mean/std(샘플 표준편차, ddof=1) 기록
- per-subject(피험자) 평균의 mean/std(샘플 표준편차, ddof=1) 기록
- TEST 전량 예측을 CSV로 저장(각 fold 폴더와 run 통합본). CSV에는 split/seed 컬럼 없음.
- 화면 출력과 최종 summary는 시드별 항목 없이, VAL/TEST 각각의
  (per-sample mean/std, per-subject mean/std)만 보여줌.
"""

from __future__ import annotations
import argparse
import os
import json
import shutil
from typing import Dict, List, Any, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from matlab_data_prep import load_all_mat_data
from utils_and_modules import (
    print_box, set_seed, ensure_dir, make_run_paths, save_json,
    train_collect_nrmse, make_fold_loaders,
    build_dataset_from_mat,
    nrmse_from_loader, nrmse_by_channel,
)

# 모델 파일
from models_cnn import CNNRegressor
from models_transformer import TransformerRegressor
from models_binn import BINN


# ---------------- session tag helper ----------------
def _format_session_tag(sessions: List[str]) -> str:
    """세션 리스트를 축약 태그로 변환. ['ss1','ss2','ss3'] → 'ss123', ['ss3'] → 'ss3'"""
    nums = sorted([s.replace("ss", "") for s in sessions])
    return "ss" + "".join(nums)


# ---------------- argparse / config ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="경로 to YAML config (기본: ./config.yaml)")
    return p.parse_args()


def load_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- pretty print config ----------------
def print_config_summary(cfg: Dict) -> None:
    split = cfg.get("split", {})
    dset  = cfg.get("dataset", {})
    model = cfg.get("model", {})

    tv_sessions = dset.get("train_val_sessions", dset.get("sessions", []))
    te_sessions = dset.get("test_sessions", dset.get("sessions", []))

    print_box("Config Summary")
    # Split
    mode = split.get("mode", "loso").upper()
    print(f"- Split mode       : {mode}")
    if mode == "FIXED":
        print(f"  • fixed val subs : {split.get('fixed_val_subjects', [])}")
        if "fixed_test_subjects" in split:
            print(f"  • fixed test subs: {split.get('fixed_test_subjects', [])}")

    # Model
    mtype = (model.get("type") or "cnn").lower()
    print(f"- Model type       : {mtype}")
    if mtype == "cnn":
        print(f"  • cnn params     : {model.get('cnn', {})}")
    elif mtype == "transformer":
        print(f"  • transformer    : {model.get('transformer', {})}")
    elif mtype == "binn":
        print(f"  • binn params    : {model.get('binn', {})}")

    # Dataset (요약)
    print(f"- Train/Val sess   : {tv_sessions}")
    print(f"- Test sessions    : {te_sessions}")
    print(f"  • GRF            : sides={dset.get('grf_sides', [])}, axes={dset.get('grf_axes', [])}")
    print(f"  • Body parts     : {dset.get('body_parts', [])}")
    print(f"  • pos/vel/acc    : {dset.get('axes_pos', [])} / {dset.get('axes_vel', [])} / {dset.get('axes_acc', [])}")
    print(f"  • IMU acc (L/G)  : {dset.get('axes_imu_acc_local', [])} / {dset.get('axes_imu_acc_global', [])}")
    print(f"  • IMU gyro (L/G) : {dset.get('axes_imu_gyro_local', [])} / {dset.get('axes_imu_gyro_global', [])}")
    print(f"  • modeling use   : {bool(dset.get('use_modeling_input', False))} "
          f"(points={dset.get('modeling_points_to_use', [])}, fields={dset.get('modeling_fields_to_use', [])})")
    print(f"  • window         : mode={dset.get('window_mode','percent')}, "
          f"pct={dset.get('win_pct',[0.0,100.0])}, time={dset.get('win_time',[0.0,0.5])}")


def config_summary_dict(cfg: Dict) -> Dict:
    split = cfg.get("split", {})
    dset  = cfg.get("dataset", {})
    model = cfg.get("model", {}) or {}
    mtype = (model.get("type") or "cnn").lower()

    return {
        "mode": split.get("mode", "loso"),
        "fixed_val_subjects":  split.get("fixed_val_subjects", []),
        "fixed_test_subjects": split.get("fixed_test_subjects", []),

        "model_type": mtype,
        "model_params": model.get(mtype, {}),

        "train sessions": dset.get("train_val_sessions"),
        "test sessions" : dset.get("test_sessions"),
        "grf": {
            "sides": dset.get("grf_sides", []),
            "axes":  dset.get("grf_axes", []),
        },
        "body_parts": dset.get("body_parts", []),
        "axes": {
            "pos": dset.get("axes_pos", []),
            "vel": dset.get("axes_vel", []),
            "acc": dset.get("axes_acc", []),
            "imu_acc_local":  dset.get("axes_imu_acc_local", []),
            "imu_acc_global": dset.get("axes_imu_acc_global", []),
            "imu_gyro_local":  dset.get("axes_imu_gyro_local", []),
            "imu_gyro_global": dset.get("axes_imu_gyro_global", []),
        },
        "modeling": {
            "use_modeling_input": bool(dset.get("use_modeling_input", False)),
            "points": dset.get("modeling_points_to_use", []),
            "fields": dset.get("modeling_fields_to_use", []),
        },
        "window": {
            "mode": dset.get("window_mode", "percent"),
            "win_pct": dset.get("win_pct", [0.0, 100.0]),
            "win_time": dset.get("win_time", [0.0, 0.5]),
        },
    }


def build_model_from_cfg(model_cfg: Dict, in_ch: int, out_ch: int) -> torch.nn.Module:
    mtype = (model_cfg.get("type") or "cnn").lower()

    if mtype == "cnn":
        hp = model_cfg.get("cnn", {}) or {}
        channels = tuple(hp.get("channels", [128, 128, 64]))
        kernels  = tuple(hp.get("kernels",  [9, 5, 3]))
        dropout  = float(hp.get("dropout",  0.0))
        return CNNRegressor(in_dim=in_ch, out_dim=out_ch,
                            channels=channels, kernels=kernels, dropout=dropout)

    if mtype == "transformer":
        hp = model_cfg.get("transformer", {}) or {}
        return TransformerRegressor(
            in_dim=in_ch, out_dim=out_ch,
            d_model=hp.get("d_model", 128),
            nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 3),
            dim_ff=hp.get("dim_ff", 256),
            drop=hp.get("dropout", 0.1),
        )

    if mtype == "binn":
        # config.yaml에서 파라미터는 편의상 기존 'binn' 항목을 재사용합니다.
        hp = model_cfg.get("binn", {}) or {}

        wrist_dim = int(hp.get("wrist_dim", 0))
        sacrum_dim = int(hp.get("sacrum_dim", 0))
        if wrist_dim <= 0 or sacrum_dim <= 0:
            raise ValueError("[binn] Please set wrist_dim and sacrum_dim")

        # Ablation 모드 읽기 (없으면 기본값 = 제안 모델)
        # 'proposed' 가 표준 이름이고, 'binn' 과 구버전 내부 이름 'no_physics' 도 허용한다.
        # decoder_channels_by_mode 조회와 출력 폴더명이 이 값을 그대로 쓰므로
        # 모델 생성 전에 여기서 한 번 정규화해야 한다.
        _MODE_ALIASES = {"binn": "proposed", "no_physics": "proposed"}
        ablation_mode = hp.get("ablation_mode", "proposed")
        ablation_mode = _MODE_ALIASES.get(str(ablation_mode).lower(), str(ablation_mode).lower())

        # 모드별 decoder channels (파라미터 수 맞춤)
        # 구버전 config 는 이 표의 키를 'no_physics' 로 쓴다. 키를 놓치면 기본값으로
        # 떨어져 파라미터 수가 달라지므로, 별칭 키까지 함께 조회한다.
        decoder_channels_by_mode = hp.get("decoder_channels_by_mode", {})
        _dec_keys = [ablation_mode]
        if ablation_mode == "proposed":
            _dec_keys += ["no_physics", "binn"]
        _hit = next((k for k in _dec_keys if k in decoder_channels_by_mode), None)
        if _hit is not None:
            dec_ch = tuple(decoder_channels_by_mode[_hit])
        else:
            dec_ch = tuple(hp.get("decoder_channels", [128, 64]))

        return BINN(
            in_dim=in_ch,
            out_dim=out_ch,
            wrist_dim=wrist_dim,
            sacrum_dim=sacrum_dim,
            ablation_mode=ablation_mode,

            # 나머지 파라미터 전달
            wrist_channels=tuple(hp.get("wrist_channels", [64, 64])),
            wrist_kernels=tuple(hp.get("wrist_kernels", [9, 5])),
            sacrum_channels=tuple(hp.get("sacrum_channels", [32, 32])),
            sacrum_kernels=tuple(hp.get("sacrum_kernels", [9, 5])),
            embed_dim=int(hp.get("embed_dim", 64)),
            attn_heads=int(hp.get("attn_heads", 1)),
            decoder_channels=dec_ch,
            decoder_kernels=tuple(hp.get("decoder_kernels", [7, 3])),
            phys_smooth_kernel=int(hp.get("phys_smooth_kernel", 5)),
            phys_scale_init=float(hp.get("phys_scale_init", 1.0)),
            dropout=float(hp.get("dropout", 0.1)),
            include_phys_in_residual=bool(hp.get("include_phys_in_residual", True)),
            extra_to_wrist=bool(hp.get("extra_to_wrist", True)),
        )

    raise ValueError(f"Unknown model type: {mtype}")

# ---------------- helpers ----------------
def _select_indices(info: List[Dict[str,Any]], *, subjects=None, sessions=None) -> List[int]:
    """subjects(집합/리스트)와 sessions(집합/리스트)로 필터."""
    sub_set = set(map(str, subjects)) if subjects is not None else None
    ses_set = set(map(str, sessions)) if sessions is not None else None
    idx = []
    for i, it in enumerate(info):
        ok = True
        if sub_set is not None:
            ok = ok and (str(it.get("subject","")) in sub_set)
        if ses_set is not None:
            ok = ok and (str(it.get("session","")) in ses_set)
        if ok:
            idx.append(i)
    return idx


def _mean_std_from_sums_ddof1(_sum: float, _sqsum: float, _cnt: int):
    """샘플 평균/표준편차(ddof=1) 계산. cnt<=1이면 (mean, nan)"""
    if _cnt <= 0:
        return float("nan"), float("nan")
    m = _sum / _cnt
    if _cnt <= 1:
        return float(m), float("nan")
    # 분산 = (Σx^2 - (Σx)^2 / n) / (n-1)
    var = max(0.0, (_sqsum - (_sum * _sum) / _cnt) / (_cnt - 1))
    return float(m), float(np.sqrt(var))


def _mean_std_from_list_ddof1(values: List[float]):
    """리스트에 대해 ddof=1 표준편차."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


# ---------------- main ----------------
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # run 설정
    run = cfg.get("run", {})
    seeds: List[int] = list(run.get("seeds", [42]))
    device = torch.device("cuda" if run.get("device", "cuda").startswith("cuda") and torch.cuda.is_available() else "cpu")
    epochs = int(run.get("epochs", 50))
    batch  = int(run.get("batch_size", 64))
    lr     = float(run.get("lr", 3e-4))
    wd     = float(run.get("weight_decay", 1e-4))
    out_root = run.get("output_dir", "./runs")
    exp_name = run.get("name", "grf_loso_demo")
    n_workers = int(run.get("num_workers", 2))
    save_ckpt = bool(run.get("save_checkpoints", True))

    print_box(f"Experiment: {exp_name} | device={device} | seeds={seeds}")
    print_config_summary(cfg)

    # .mat 로드
    data = cfg.get("data", {})
    grf, kin, mdl = load_all_mat_data(
        grf_path=data.get("grf_mat", ""),
        grf_struct=data.get("grf_struct", "stride_grf"),
        kin_path=data.get("kin_mat", ""),
        kin_struct=data.get("kin_struct", "stride_kinematics_arm"),
        modeling_path=data.get("modeling_mat", ""),
        modeling_struct=data.get("modeling_struct", "stride_modeling"),
    )

    # 세션 집합
    dset = cfg.get("dataset", {})
    tv_sessions = dset.get("train_val_sessions", dset.get("sessions", ["ss2"]))
    te_sessions = dset.get("test_sessions", dset.get("sessions", ["ss2"]))
    all_sessions = sorted(list(set(tv_sessions) | set(te_sessions)))

    # CSV 파일명 결정: 세션이 같으면 ss3.csv, 다르면 train_ss1.csv
    if sorted(tv_sessions) == sorted(te_sessions):
        test_csv_name = _format_session_tag(te_sessions) + ".csv"
    else:
        test_csv_name = "train_" + _format_session_tag(tv_sessions) + ".csv"

    # X,y 구성 (all_sessions로 조립)
    print_box("Assemble dataset X(N,T,F), y(N,T,C)")
    X, y, sample_info = build_dataset_from_mat(
        grf=grf, kin=kin, modeling=mdl if dset.get("use_modeling_input") else {},
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
    print(f"[data] X: {X.shape}, y: {y.shape}, samples={len(sample_info)}")

    # ------------------------------------------------------------------
    # BINN: wrist_dim / sacrum_dim 을 dataset 설정에서 자동 유도
    #
    # 모델이 x[:, :, :wrist_dim] / x[:, :, wrist_dim:wrist_dim+sacrum_dim] 로
    # "위치 기반" 슬라이싱을 하므로, config의 dim 값이 실제 채널 구성과 어긋나면
    # 예외 없이 엉뚱한 채널이 sacrum(CoM) 브랜치로 들어간다.
    #   예) modeling_fields_to_use를 2개->1개로 줄이고 sacrum_dim=2를 그대로 두면
    #       stride duration 스칼라가 CoM-to-GRF converter 입력이 되어버린다.
    # 그래서 config 값을 신뢰하지 않고 dataset 설정에서 다시 계산해 덮어쓴다.
    # ------------------------------------------------------------------
    if cfg.get("model", {}).get("type", "") in ("binn",):
        n_kin_axes = sum(len(dset.get(k, []) or []) for k in (
            "axes_pos", "axes_vel", "axes_acc",
            "axes_imu_acc_local", "axes_imu_acc_global",
            "axes_imu_gyro_local", "axes_imu_gyro_global",
        ))
        wrist_auto = len(dset.get("body_parts", []) or []) * n_kin_axes
        sacrum_auto = (
            len(dset.get("modeling_points_to_use", []) or [])
            * len(dset.get("modeling_fields_to_use", []) or [])
        ) if dset.get("use_modeling_input", False) else 0
        extra_auto = 1 if dset.get("add_stride_duration_scalar", True) else 0
        expected_F = wrist_auto + sacrum_auto + extra_auto

        if expected_F != F:
            raise ValueError(
                f"[dim-check] dataset 설정으로 계산한 feature 수 {expected_F} "
                f"(= wrist {wrist_auto} + sacrum {sacrum_auto} + extra {extra_auto}) 가 "
                f"실제 X 채널 수 {F} 와 다릅니다. "
                f".mat 파일에 없는 필드를 요청했을 가능성이 큽니다 "
                f"(build_dataset_from_mat 은 없는 필드를 조용히 건너뜁니다)."
            )

        hp_auto = cfg.setdefault("model", {}).setdefault("binn", {})
        for key, val in (("wrist_dim", wrist_auto), ("sacrum_dim", sacrum_auto)):
            old = hp_auto.get(key, None)
            if old is not None and int(old) != int(val):
                print(f"[dim-check] model.binn.{key}: "
                      f"config={old} -> 자동 유도값 {val} 로 덮어씀")
            hp_auto[key] = int(val)
        print(f"[dim-check] in_dim={F} = wrist {wrist_auto} "
              f"+ sacrum {sacrum_auto} + extra {extra_auto}")

    # CSV 메타 플래그
    modeling_used = bool(dset.get("use_modeling_input", False))

    # Physics output 저장 플래그
    model_cfg_hp = cfg.get("model", {}).get("binn", {}) or {}
    save_phys_output = bool(model_cfg_hp.get("save_physics_output", False))
    mtype_check = (cfg.get("model", {}).get("type", "cnn")).lower()
    if save_phys_output and mtype_check != "binn":
        save_phys_output = False  # physics output은 BINN 에서만 가능

    # train/val에 쓸 서브셋만 따로 분리
    idx_tv = _select_indices(sample_info, sessions=tv_sessions)
    X_tv = X[idx_tv]; y_tv = y[idx_tv]
    info_tv = [sample_info[i] for i in idx_tv]

    split = cfg.get("split", {})
    mode = split.get("mode", "loso").lower()
    if mode not in ("loso", "fixed"):
        raise ValueError("split.mode must be 'loso' or 'fixed'")

    # -------- 전체(모든 시드) 통합 누적용 --------
    grand_val_ps_sum = 0.0
    grand_val_ps_sqsum = 0.0
    grand_val_ps_cnt = 0

    grand_test_ps_sum = 0.0
    grand_test_ps_sqsum = 0.0
    grand_test_ps_cnt = 0

    grand_val_subject_map: DefaultDict[str, List[float]] = defaultdict(list)
    grand_test_subject_map: DefaultDict[str, List[float]] = defaultdict(list)

    for seed in seeds:
        set_seed(int(seed))
        # 모델 폴더명 결정: ablation 모드는 하위 폴더로 분리
        mtype = cfg.get("model", {}).get("type", "cnn")
        if mtype == "binn":
            ablation_mode = cfg.get("model", {}).get("binn", {}).get("ablation_mode", "proposed")
            ablation_mode = {"binn": "proposed", "no_physics": "proposed"}.get(
                str(ablation_mode).lower(), str(ablation_mode).lower())
            model_folder = os.path.join(mtype, ablation_mode)
        else:
            model_folder = mtype
        paths = make_run_paths(out_root, model_folder, seed, tag=f"{mode}_{exp_name}")
        tb_dir = paths["tb_dir"]
        ckpt_dir = paths["ckpt_dir"]
        run_dir = paths["run_dir"]

        # run_dir에만 config 저장
        with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # ★ 체크포인트 완전 비활성화: ckpt 폴더 제거 + 이후 사용 금지
        if not save_ckpt:
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            # ckpt_dir = None

        # 시드 통합 TEST CSV를 위해 부분 DF 모으기
        test_csv_parts: List[pd.DataFrame] = []

        print_box(f"Training | seed={seed} | mode={mode}")

        # 시드 단위 per-sample 누적
        val_sample_sum = 0.0
        val_sample_sqsum = 0.0
        val_sample_cnt = 0

        test_sample_sum = 0.0
        test_sample_sqsum = 0.0
        test_sample_cnt = 0

        # 시드 단위 per-subject 누적
        val_subject_map: DefaultDict[str, List[float]] = defaultdict(list)
        test_subject_map: DefaultDict[str, List[float]] = defaultdict(list)

        axes_names = dset.get("grf_axes", [])
        ch_names = [axes_names[i] if i < C else f"ch{i}" for i in range(C)]

        if mode == "loso":
            # LOSO: train/val은 info_tv에서만
            subs = sorted({s["subject"] for s in info_tv})
            for i, val_sub in enumerate(subs, 1):
                print(f"[Fold {i}/{len(subs)}] val_subject={val_sub}")

                # DataLoaders
                tr_loader, vl_loader, in_ch, out_ch, seq_len, scaler = make_fold_loaders(
                    X_tv, y_tv, info_tv,
                    val_subject=val_sub, batch_size=batch, n_workers=n_workers, normalize=True
                )

                model = build_model_from_cfg(cfg.get("model", {}), in_ch, out_ch).to(device)

                # fold 디렉터리 및 log_dir
                fold_dir = os.path.join(ckpt_dir, f"fold_{val_sub}")
                ensure_dir(fold_dir)
                log_dir  = os.path.join(fold_dir, "tb")

                # 학습
                hist = train_collect_nrmse(
                    model, tr_loader, vl_loader,
                    epochs=epochs, lr=lr, weight_decay=wd,
                    device=device, log_dir=log_dir, add_graph_once=False
                )

                # learning curve CSV 저장
                lc_df = pd.DataFrame({
                    "epoch": list(range(1, epochs + 1)),
                    "train_nrmse": hist["train_nrmse_curve"],
                    "val_nrmse": hist["val_nrmse_curve"],
                })
                lc_csv_path = os.path.join(fold_dir, "learning_curve.csv")
                lc_df.to_csv(lc_csv_path, index=False)
                print(f"[csv] learning curve saved → {lc_csv_path}")

                # 표준화 통계 & config 저장
                if scaler is not None:
                    mu = scaler.mean_.reshape(1, 1, -1).astype(np.float32)
                    sd = scaler.scale_.reshape(1, 1, -1).astype(np.float32)
                    np.savez_compressed(os.path.join(fold_dir, "norm_stats.npz"), mu=mu, sd=sd)
                with open(os.path.join(fold_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

                # 최종 epoch 가중치 로드 (논문 프로토콜)
                last_pth = os.path.join(fold_dir, "last.pth")
                if os.path.isfile(last_pth):
                    state = torch.load(last_pth, map_location=device)
                    model.load_state_dict(state)
                    model.eval()
                    print(f"[eval] loaded LAST checkpoint for VAL/TEST → {last_pth}")
                else:
                    raise RuntimeError(
                        f"last.pth not found in {fold_dir}. The paper protocol evaluates "
                        "the final-epoch weights; there is no best-checkpoint fallback "
                        "because under LOSO the validation fold is the test subject.")

                # ----- VAL per-channel NRMSE (표시/저장만) -----
                val_per_ch = nrmse_by_channel(vl_loader, model, device)  # (C,)
                print("[VAL per-channel NRMSE(range)]")
                for ci, v in enumerate(val_per_ch):
                    nm = ch_names[ci]
                    print(f"  - {nm}: {v:.6f}")

                # ----- VAL per-sample NRMSE(range) 수집 → 사람맵 누적 -----
                val_ps_list = []
                with torch.no_grad():
                    for xb, yb in vl_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        yhat = model(xb)
                        err = yhat - yb
                        mse_t = (err ** 2).mean(dim=1)
                        rmse = torch.sqrt(mse_t + 1e-12)
                        rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)
                        nrmse_b = (rmse / rng).mean(dim=1)  # (B,)
                        val_ps_list.append(nrmse_b.cpu().numpy())

                val_ps = np.concatenate(val_ps_list, axis=0)
                val_subject_map[str(val_sub)].extend(val_ps.tolist())
                val_sample_sum   += float(val_ps.sum())
                val_sample_sqsum += float((val_ps ** 2).sum())
                val_sample_cnt   += int(val_ps.size)

                # ===== TEST 평가(세션=te_sessions & subject=val_sub) =====
                idx_te = _select_indices(sample_info, subjects=[val_sub], sessions=te_sessions)
                if len(idx_te) == 0:
                    print(f"[TEST] hold_{val_sub}: test_sessions 안에 해당 피험자 샘플이 없습니다. (skip)")
                else:
                    X_te = X[idx_te]; Y_te = y[idx_te]
                    Nt = X_te.shape[0]
                    Xt_sc = scaler.transform(X_te.reshape(Nt*T, F)).reshape(Nt, T, F).astype(np.float32)
                    te_ds = TensorDataset(torch.from_numpy(Xt_sc), torch.from_numpy(Y_te.astype(np.float32)))
                    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=n_workers)

                    # per-channel
                    test_per_ch = nrmse_by_channel(te_loader, model, device)  # (C,)
                    print("[TEST per-channel NRMSE(range)]")
                    for ci, v in enumerate(test_per_ch):
                        nm = ch_names[ci]
                        print(f"  - {nm}: {v:.6f}")

                    # per-sample + 예측 수집(CSV)
                    te_ps_list = []
                    yt_list, yp_list, yphys_list = [], [], []
                    with torch.no_grad():
                        for xb, yb in te_loader:
                            xb = xb.to(device); yb = yb.to(device)
                            yhat = model(xb)
                            err = yhat - yb
                            mse_t = (err ** 2).mean(dim=1)
                            rmse = torch.sqrt(mse_t + 1e-12)
                            rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)
                            nrmse_b = (rmse / rng).mean(dim=1)  # (B,)
                            te_ps_list.append(nrmse_b.cpu().numpy())

                            yt_list.append(yb.cpu().numpy())
                            yp_list.append(yhat.cpu().numpy())

                            if save_phys_output and hasattr(model, 'get_physics_output'):
                                yphys_list.append(model.get_physics_output(xb).cpu().numpy())

                    ytrue_te = np.concatenate(yt_list, axis=0)
                    yhat_te  = np.concatenate(yp_list, axis=0)
                    yphys_te = np.concatenate(yphys_list, axis=0) if yphys_list else None
                    te_ps    = np.concatenate(te_ps_list, axis=0)

                    # 사람맵 누적(LOSO는 단일 피험자)
                    test_subject_map[str(val_sub)].extend(te_ps.tolist())
                    test_sample_sum   += float(te_ps.sum())
                    test_sample_sqsum += float((te_ps ** 2).sum())
                    test_sample_cnt   += int(te_ps.size)

                    # TEST CSV
                    info_te  = [sample_info[i] for i in idx_te]

                    # CSV (split/seed 없음)
                    rows = []
                    for n_idx in range(ytrue_te.shape[0]):
                        meta = info_te[n_idx]
                        subj = str(meta.get("subject",""))
                        sess = str(meta.get("session",""))
                        day  = str(meta.get("day",""))
                        sidx = int(meta.get("stride_idx", -1))
                        for t_idx in range(ytrue_te.shape[1]):
                            r = {
                                "fold": f"fold_{val_sub}",
                                "subject": subj,
                                "test session": sess,
                                "day": day,
                                "modeling_used": bool(modeling_used),
                                "stride_idx": sidx,
                                "t": int(t_idx),
                            }
                            for ci, nm in enumerate(ch_names):
                                r[f"y_true_{nm}"] = float(ytrue_te[n_idx, t_idx, ci])
                                r[f"y_pred_{nm}"] = float(yhat_te[n_idx, t_idx, ci])
                                if yphys_te is not None:
                                    r[f"y_phys_{nm}"] = float(yphys_te[n_idx, t_idx, ci])
                            rows.append(r)
                    df_fold = pd.DataFrame(rows)
                    csv_fold_path = os.path.join(fold_dir, test_csv_name)
                    df_fold.to_csv(csv_fold_path, index=False)
                    test_csv_parts.append(df_fold)

        else:
            # fixed: train/val은 tv_sessions에서 subject 분리
            val_subs  = split.get("fixed_val_subjects", [])
            test_subs = split.get("fixed_test_subjects", val_subs)
            if not val_subs:
                raise ValueError("[fixed] split.fixed_val_subjects가 비었습니다.")

            # 테스트 피험자도 학습에서 제외
            train_exclude = set(map(str, val_subs)) | set(map(str, test_subs))
            idx_tr_tv = [i for i, s in enumerate(info_tv) if str(s["subject"]) not in train_exclude]
            idx_vl_tv = [i for i, s in enumerate(info_tv) if str(s["subject"]) in set(map(str, val_subs))]
            if not idx_tr_tv or not idx_vl_tv:
                raise ValueError("[fixed] invalid split on train_val_sessions]")

            Xtr, ytr = X_tv[idx_tr_tv], y_tv[idx_tr_tv]
            Xvl, yvl = X_tv[idx_vl_tv], y_tv[idx_vl_tv]

            scaler = StandardScaler()
            Ntr = Xtr.shape[0]
            Xtr_sc = scaler.fit_transform(Xtr.reshape(Ntr * T, F)).reshape(Ntr, T, F).astype(np.float32)
            Xvl_sc = scaler.transform(Xvl.reshape(len(Xvl) * T, F)).reshape(len(Xvl), T, F).astype(np.float32)

            tr_ds = TensorDataset(torch.from_numpy(Xtr_sc), torch.from_numpy(ytr.astype(np.float32)))
            vl_ds = TensorDataset(torch.from_numpy(Xvl_sc), torch.from_numpy(yvl.astype(np.float32)))
            tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=n_workers)
            vl_loader = DataLoader(vl_ds, batch_size=batch, shuffle=False, num_workers=n_workers)

            model = build_model_from_cfg(cfg.get("model", {}), in_ch=F, out_ch=C).to(device)

            fold_dir = os.path.join(ckpt_dir, "fixed")
            ensure_dir(fold_dir)
            log_dir  = os.path.join(fold_dir, "tb")

            hist = train_collect_nrmse(
                model, tr_loader, vl_loader,
                epochs=epochs, lr=lr, weight_decay=wd,
                device=device, log_dir=log_dir, add_graph_once=False
            )

            # learning curve CSV 저장
            lc_df = pd.DataFrame({
                "epoch": list(range(1, epochs + 1)),
                "train_nrmse": hist["train_nrmse_curve"],
                "val_nrmse": hist["val_nrmse_curve"],
            })
            lc_csv_path = os.path.join(fold_dir, "learning_curve.csv")
            lc_df.to_csv(lc_csv_path, index=False)
            print(f"[csv] learning curve saved → {lc_csv_path}")

            # 표준화 통계 & config 저장
            mu = scaler.mean_.reshape(1, 1, -1).astype(np.float32)
            sd = scaler.scale_.reshape(1, 1, -1).astype(np.float32)
            np.savez_compressed(os.path.join(fold_dir, "norm_stats.npz"), mu=mu, sd=sd)
            with open(os.path.join(fold_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

            # LAST-EPOCH 로드 (LOSO 경로와 동일 프로토콜)
            last_pth = os.path.join(fold_dir, "last.pth")
            if os.path.isfile(last_pth):
                state = torch.load(last_pth, map_location=device)
                model.load_state_dict(state)
                model.eval()
                print(f"[eval] loaded LAST checkpoint for VAL/TEST → {last_pth}")
            else:
                raise RuntimeError(
                    f"last.pth not found in {fold_dir}. The paper protocol evaluates "
                    "the final-epoch weights; there is no best-checkpoint fallback.")

            # ----- FIXED VAL per-channel (표시만) -----
            val_per_ch = nrmse_by_channel(vl_loader, model, device)  # (C,)
            print("[FIXED VAL per-channel NRMSE(range)]")
            for ci, v in enumerate(val_per_ch):
                nm = ch_names[ci]
                print(f"  - {nm}: {v:.6f}")

            # ----- FIXED VAL per-sample + 사람맵 누적 -----
            val_ps_all = []
            val_subj_seq = [str(info_tv[i]["subject"]) for i in idx_vl_tv]  # DataLoader 순서와 동일
            offset = 0
            with torch.no_grad():
                for xb, yb in vl_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    yhat = model(xb)
                    B = yb.shape[0]
                    err = yhat - yb
                    mse_t = (err ** 2).mean(dim=1)
                    rmse = torch.sqrt(mse_t + 1e-12)
                    rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)
                    nrmse_b = (rmse / rng).mean(dim=1).cpu().numpy()  # (B,)
                    # 사람별 누적
                    for j in range(B):
                        val_subject_map[val_subj_seq[offset + j]].append(float(nrmse_b[j]))
                    val_ps_all.append(nrmse_b)
                    offset += B
            val_ps = np.concatenate(val_ps_all, axis=0)
            val_sample_sum   += float(val_ps.sum())
            val_sample_sqsum += float((val_ps ** 2).sum())
            val_sample_cnt   += int(val_ps.size)

            # ===== TEST 평가 =====
            idx_te = _select_indices(sample_info, subjects=test_subs, sessions=te_sessions)
            if len(idx_te) == 0:
                print("[TEST] fixed: test_sessions 안에 테스트 피험자 샘플이 없습니다. (skip)")
            else:
                X_te = X[idx_te]; Y_te = y[idx_te]
                Nt = X_te.shape[0]
                Xt_sc = scaler.transform(X_te.reshape(Nt*T, F)).reshape(Nt, T, F).astype(np.float32)
                te_ds = TensorDataset(torch.from_numpy(Xt_sc), torch.from_numpy(Y_te.astype(np.float32)))
                te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=n_workers)

                # per-channel
                test_per_ch = nrmse_by_channel(te_loader, model, device)
                print("[FIXED TEST per-channel NRMSE(range)]")
                for ci, v in enumerate(test_per_ch):
                    nm = ch_names[ci]
                    print(f"  - {nm}: {v:.6f}")

                # per-sample + 사람맵 누적 + 예측 수집
                te_ps_all = []
                test_subj_seq = [str(sample_info[i]["subject"]) for i in idx_te]
                offset = 0
                yt_list, yp_list, yphys_list = [], [], []
                with torch.no_grad():
                    for xb, yb in te_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        yhat = model(xb)
                        B = yb.shape[0]
                        err = yhat - yb
                        mse_t = (err ** 2).mean(dim=1)
                        rmse = torch.sqrt(mse_t + 1e-12)
                        rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)
                        nrmse_b = (rmse / rng).mean(dim=1).cpu().numpy()
                        for j in range(B):
                            test_subject_map[test_subj_seq[offset + j]].append(float(nrmse_b[j]))
                        te_ps_all.append(nrmse_b)
                        offset += B

                        yt_list.append(yb.cpu().numpy())
                        yp_list.append(yhat.cpu().numpy())

                        if save_phys_output and hasattr(model, 'get_physics_output'):
                            yphys_list.append(model.get_physics_output(xb).cpu().numpy())

                te_ps = np.concatenate(te_ps_all, axis=0)
                test_sample_sum   += float(te_ps.sum())
                test_sample_sqsum += float((te_ps ** 2).sum())
                test_sample_cnt   += int(te_ps.size)

                ytrue_te = np.concatenate(yt_list, axis=0)
                yhat_te  = np.concatenate(yp_list, axis=0)
                yphys_te = np.concatenate(yphys_list, axis=0) if yphys_list else None
                info_te  = [sample_info[i] for i in idx_te]

                # TEST CSV

                rows = []
                for n_idx in range(ytrue_te.shape[0]):
                    meta = info_te[n_idx]
                    subj = str(meta.get("subject",""))
                    sess = str(meta.get("session",""))
                    day  = str(meta.get("day",""))
                    sidx = int(meta.get("stride_idx", -1))
                    for t_idx in range(ytrue_te.shape[1]):
                        r = {
                            "fold": "fixed",
                            "subject": subj,
                            "test session": sess,
                            "day": day,
                            "modeling_used": bool(modeling_used),
                            "stride_idx": sidx,
                            "t": int(t_idx),
                        }
                        for ci, nm in enumerate(ch_names):
                            r[f"y_true_{nm}"] = float(ytrue_te[n_idx, t_idx, ci])
                            r[f"y_pred_{nm}"] = float(yhat_te[n_idx, t_idx, ci])
                            if yphys_te is not None:
                                r[f"y_phys_{nm}"] = float(yphys_te[n_idx, t_idx, ci])
                        rows.append(r)
                df_fold = pd.DataFrame(rows)
                csv_fold_path = os.path.join(fold_dir, test_csv_name)
                df_fold.to_csv(csv_fold_path, index=False)
                test_csv_parts.append(df_fold)

        # 시드별 검증 곡선 → CSV로 대체 (fig 저장 제거됨)

        # 시드 통합 TEST CSV 저장(그대로 유지)
        if test_csv_parts:
            df_all = pd.concat(test_csv_parts, ignore_index=True)
            csv_run_path = os.path.join(run_dir, test_csv_name)
            df_all.to_csv(csv_run_path, index=False)

        # -------- 시드 종료: Grand 누적(샘플/사람) --------
        grand_val_ps_sum   += val_sample_sum
        grand_val_ps_sqsum += val_sample_sqsum
        grand_val_ps_cnt   += val_sample_cnt

        grand_test_ps_sum   += test_sample_sum
        grand_test_ps_sqsum += test_sample_sqsum
        grand_test_ps_cnt   += test_sample_cnt

        for k, v in val_subject_map.items():
            if v: grand_val_subject_map[k].extend(v)
        for k, v in test_subject_map.items():
            if v: grand_test_subject_map[k].extend(v)

        # (참고) per-seed 요약/출력은 요청에 따라 모두 생략합니다.

    # ======================
    # 최종(모든 시드 통합) 요약
    # ======================

    # per-sample 통계
    val_sample_mean,  val_sample_std  = _mean_std_from_sums_ddof1(grand_val_ps_sum,  grand_val_ps_sqsum,  grand_val_ps_cnt)
    test_sample_mean, test_sample_std = _mean_std_from_sums_ddof1(grand_test_ps_sum, grand_test_ps_sqsum, grand_test_ps_cnt)

    # per-subject 통계: 피험자별 평균 → 그 분포의 mean/std(ddof=1)
    val_subject_means = []
    for subj, vals in grand_val_subject_map.items():
        if len(vals) > 0:
            val_subject_means.append(float(np.mean(vals)))
    test_subject_means = []
    for subj, vals in grand_test_subject_map.items():
        if len(vals) > 0:
            test_subject_means.append(float(np.mean(vals)))

    val_subject_mean,  val_subject_std  = _mean_std_from_list_ddof1(val_subject_means)
    test_subject_mean, test_subject_std = _mean_std_from_list_ddof1(test_subject_means)

    # 최종 요약 JSON (시드/폴드 평균 키 제거, 요청한 항목만)
    final_summary = {
        "experiment": exp_name,
        "mode": mode,
        "seeds": seeds,  # 어떤 시드들이 사용되었는지 레퍼런스로만 남김
        "VAL_per_sample": {
            "mean": val_sample_mean,
            "std":  val_sample_std,
            "n_samples": int(grand_val_ps_cnt),
        },
        "VAL_per_subject": {
            "mean_of_subject_means": val_subject_mean,
            "std_of_subject_means":  val_subject_std,
            "n_subjects": len(val_subject_means),
        },
        "TEST_per_sample": {
            "mean": test_sample_mean,
            "std":  test_sample_std,
            "n_samples": int(grand_test_ps_cnt),
        },
        "TEST_per_subject": {
            "mean_of_subject_means": test_subject_mean,
            "std_of_subject_means":  test_subject_std,
            "n_subjects": len(test_subject_means),
        },
        "config_summary": config_summary_dict(cfg),
    }

    # --- save final summary (공통: LOSO/FIXED 모두) ---
    summary_path = os.path.join(run_dir, "summary.json")  # 마지막 시드의 run 디렉터리
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved summary → {summary_path}")

    print_box("Grand summary (no per-seed)")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))

    # 화면에 추가로 사람이 바로 보도록 간단 요약 라인
    print("\n[FINAL] VAL   per-sample : mean={:.6f} | std={:.6f} | N={}".format(
        val_sample_mean, val_sample_std, grand_val_ps_cnt))
    print("[FINAL] VAL   per-subject: mean={:.6f} | std={:.6f} | n_subjects={}".format(
        val_subject_mean, val_subject_std, len(val_subject_means)))
    print("[FINAL] TEST  per-sample : mean={:.6f} | std={:.6f} | N={}".format(
        test_sample_mean, test_sample_std, grand_test_ps_cnt))
    print("[FINAL] TEST  per-subject: mean={:.6f} | std={:.6f} | n_subjects={}".format(
        test_subject_mean, test_subject_std, len(test_subject_means)))


if __name__ == "__main__":
    main()