#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — LOSO / Fixed training and evaluation entry point (NRMSE(range) only)
- Train/val sessions and test sessions are configured separately.
- After each fold (or the FIXED split) is trained, TEST is evaluated with the same scaler.
- Per-channel NRMSE(range) is printed and saved to JSON for VAL/TEST.
- VAL/TEST are evaluated exclusively with last.pth (final-epoch weights).
  (Under LOSO the validation fold IS the test subject, so selecting a checkpoint by
   validation loss would amount to test-set selection. Therefore no best checkpoint
   is saved or loaded.)
- Per-sample NRMSE(range) mean/std (sample SD, ddof=1) is recorded.
- Mean/std of per-subject averages (sample SD, ddof=1) is recorded.
- All TEST predictions are written to CSV (per-fold and run-level merged files);
  the CSV carries no split/seed columns.
- Console output and the final summary report only the VAL/TEST
  (per-sample mean/std, per-subject mean/std), with no per-seed items.
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

# model definitions
from models_cnn import CNNRegressor
from models_transformer import TransformerRegressor
from models_binn import BINN


# ---------------- session tag helper ----------------
def _format_session_tag(sessions: List[str]) -> str:
    """Compress a session list into a tag: ['ss1','ss2','ss3'] → 'ss123', ['ss3'] → 'ss3'."""
    nums = sorted([s.replace("ss", "") for s in sessions])
    return "ss" + "".join(nums)


# ---------------- argparse / config ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="path to YAML config (default: ./config.yaml)")
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

    # Dataset (summary)
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
        # For convenience the hyperparameters are read from the existing 'binn' block.
        hp = model_cfg.get("binn", {}) or {}

        wrist_dim = int(hp.get("wrist_dim", 0))
        sacrum_dim = int(hp.get("sacrum_dim", 0))
        if wrist_dim <= 0 or sacrum_dim <= 0:
            raise ValueError("[binn] Please set wrist_dim and sacrum_dim")

        # Read the ablation mode (default = proposed model).
        # 'proposed' is the canonical name; 'binn' and the legacy internal name
        # 'no_physics' are accepted as aliases. The decoder_channels_by_mode lookup
        # and the output folder name use this value verbatim, so it must be
        # normalised here once before the model is built.
        _MODE_ALIASES = {"binn": "proposed", "no_physics": "proposed"}
        ablation_mode = hp.get("ablation_mode", "proposed")
        ablation_mode = _MODE_ALIASES.get(str(ablation_mode).lower(), str(ablation_mode).lower())

        # Per-mode decoder channels (keeps the parameter counts matched).
        # Older configs key this table with 'no_physics'. Missing the key would fall
        # back to the defaults and change the parameter count, so alias keys are
        # looked up as well.
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

            # remaining hyperparameters
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
    """Filter sample indices by subjects (set/list) and sessions (set/list)."""
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
    """Sample mean/std (ddof=1) from running sums. Returns (mean, nan) when cnt<=1."""
    if _cnt <= 0:
        return float("nan"), float("nan")
    m = _sum / _cnt
    if _cnt <= 1:
        return float(m), float("nan")
    # variance = (Σx^2 - (Σx)^2 / n) / (n-1)
    var = max(0.0, (_sqsum - (_sum * _sum) / _cnt) / (_cnt - 1))
    return float(m), float(np.sqrt(var))


def _mean_std_from_list_ddof1(values: List[float]):
    """Mean and ddof=1 standard deviation of a list."""
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

    # run settings
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

    # load .mat files
    data = cfg.get("data", {})
    grf, kin, mdl = load_all_mat_data(
        grf_path=data.get("grf_mat", ""),
        grf_struct=data.get("grf_struct", "stride_grf"),
        kin_path=data.get("kin_mat", ""),
        kin_struct=data.get("kin_struct", "stride_kinematics_arm"),
        modeling_path=data.get("modeling_mat", ""),
        modeling_struct=data.get("modeling_struct", "stride_modeling"),
    )

    # session sets
    dset = cfg.get("dataset", {})
    tv_sessions = dset.get("train_val_sessions", dset.get("sessions", ["ss2"]))
    te_sessions = dset.get("test_sessions", dset.get("sessions", ["ss2"]))
    all_sessions = sorted(list(set(tv_sessions) | set(te_sessions)))

    # Test-CSV file name: same sessions → e.g. ss3.csv; different → e.g. train_ss1.csv
    if sorted(tv_sessions) == sorted(te_sessions):
        test_csv_name = _format_session_tag(te_sessions) + ".csv"
    else:
        test_csv_name = "train_" + _format_session_tag(tv_sessions) + ".csv"

    # assemble X, y over all_sessions
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
    # BINN: derive wrist_dim / sacrum_dim automatically from the dataset settings.
    #
    # The model slices the input positionally as x[:, :, :wrist_dim] and
    # x[:, :, wrist_dim:wrist_dim+sacrum_dim], so if the dims in the config
    # disagree with the actual channel layout, the wrong channels silently end
    # up in the CoM branch.
    #   e.g. reducing modeling_fields_to_use from 2 to 1 while keeping
    #        sacrum_dim=2 would feed the stride-duration scalar into the
    #        CoM-to-GRF converter.
    # The config values are therefore not trusted; they are recomputed from the
    # dataset settings and overwritten.
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
                f"[dim-check] feature count computed from the dataset settings, {expected_F} "
                f"(= wrist {wrist_auto} + sacrum {sacrum_auto} + extra {extra_auto}), "
                f"differs from the actual number of X channels, {F}. "
                f"Most likely a field was requested that does not exist in the .mat file "
                f"(build_dataset_from_mat silently skips missing fields)."
            )

        hp_auto = cfg.setdefault("model", {}).setdefault("binn", {})
        for key, val in (("wrist_dim", wrist_auto), ("sacrum_dim", sacrum_auto)):
            old = hp_auto.get(key, None)
            if old is not None and int(old) != int(val):
                print(f"[dim-check] model.binn.{key}: "
                      f"config={old} -> overwritten with auto-derived value {val}")
            hp_auto[key] = int(val)
        print(f"[dim-check] in_dim={F} = wrist {wrist_auto} "
              f"+ sacrum {sacrum_auto} + extra {extra_auto}")

    # CSV metadata flag
    modeling_used = bool(dset.get("use_modeling_input", False))

    # flag: save the CoM-to-GRF Converter output alongside the predictions
    model_cfg_hp = cfg.get("model", {}).get("binn", {}) or {}
    save_phys_output = bool(model_cfg_hp.get("save_physics_output", False))
    mtype_check = (cfg.get("model", {}).get("type", "cnn")).lower()
    if save_phys_output and mtype_check != "binn":
        save_phys_output = False  # physics output exists only for BINN

    # subset used for train/val
    idx_tv = _select_indices(sample_info, sessions=tv_sessions)
    X_tv = X[idx_tv]; y_tv = y[idx_tv]
    info_tv = [sample_info[i] for i in idx_tv]

    split = cfg.get("split", {})
    mode = split.get("mode", "loso").lower()
    if mode not in ("loso", "fixed"):
        raise ValueError("split.mode must be 'loso' or 'fixed'")

    # -------- accumulators pooled over all seeds --------
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
        # Output folder name: ablation modes get their own subfolder
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

        # save the config into run_dir
        with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # checkpoints fully disabled: remove the ckpt folder and never use it
        if not save_ckpt:
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            # ckpt_dir = None

        # partial DataFrames for the seed-level merged TEST CSV
        test_csv_parts: List[pd.DataFrame] = []

        print_box(f"Training | seed={seed} | mode={mode}")

        # per-sample accumulators for this seed
        val_sample_sum = 0.0
        val_sample_sqsum = 0.0
        val_sample_cnt = 0

        test_sample_sum = 0.0
        test_sample_sqsum = 0.0
        test_sample_cnt = 0

        # per-subject accumulators for this seed
        val_subject_map: DefaultDict[str, List[float]] = defaultdict(list)
        test_subject_map: DefaultDict[str, List[float]] = defaultdict(list)

        axes_names = dset.get("grf_axes", [])
        ch_names = [axes_names[i] if i < C else f"ch{i}" for i in range(C)]

        if mode == "loso":
            # LOSO: train/val come from info_tv only
            subs = sorted({s["subject"] for s in info_tv})
            for i, val_sub in enumerate(subs, 1):
                print(f"[Fold {i}/{len(subs)}] val_subject={val_sub}")

                # DataLoaders
                tr_loader, vl_loader, in_ch, out_ch, seq_len, scaler = make_fold_loaders(
                    X_tv, y_tv, info_tv,
                    val_subject=val_sub, batch_size=batch, n_workers=n_workers, normalize=True
                )

                model = build_model_from_cfg(cfg.get("model", {}), in_ch, out_ch).to(device)

                # fold directory and log_dir
                fold_dir = os.path.join(ckpt_dir, f"fold_{val_sub}")
                ensure_dir(fold_dir)
                log_dir  = os.path.join(fold_dir, "tb")

                # training
                hist = train_collect_nrmse(
                    model, tr_loader, vl_loader,
                    epochs=epochs, lr=lr, weight_decay=wd,
                    device=device, log_dir=log_dir, add_graph_once=False
                )

                # save learning curve CSV
                lc_df = pd.DataFrame({
                    "epoch": list(range(1, epochs + 1)),
                    "train_nrmse": hist["train_nrmse_curve"],
                    "val_nrmse": hist["val_nrmse_curve"],
                })
                lc_csv_path = os.path.join(fold_dir, "learning_curve.csv")
                lc_df.to_csv(lc_csv_path, index=False)
                print(f"[csv] learning curve saved → {lc_csv_path}")

                # save normalisation statistics & config
                if scaler is not None:
                    mu = scaler.mean_.reshape(1, 1, -1).astype(np.float32)
                    sd = scaler.scale_.reshape(1, 1, -1).astype(np.float32)
                    np.savez_compressed(os.path.join(fold_dir, "norm_stats.npz"), mu=mu, sd=sd)
                with open(os.path.join(fold_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

                # load final-epoch weights (paper protocol)
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

                # ----- VAL per-channel NRMSE (display/logging only) -----
                val_per_ch = nrmse_by_channel(vl_loader, model, device)  # (C,)
                print("[VAL per-channel NRMSE(range)]")
                for ci, v in enumerate(val_per_ch):
                    nm = ch_names[ci]
                    print(f"  - {nm}: {v:.6f}")

                # ----- collect VAL per-sample NRMSE(range) → accumulate into the subject map -----
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

                # ===== TEST evaluation (sessions=te_sessions & subject=val_sub) =====
                idx_te = _select_indices(sample_info, subjects=[val_sub], sessions=te_sessions)
                if len(idx_te) == 0:
                    print(f"[TEST] hold_{val_sub}: no samples for this subject in test_sessions. (skip)")
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

                    # per-sample metrics + prediction collection (CSV)
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

                    # accumulate into the subject map (a single subject under LOSO)
                    test_subject_map[str(val_sub)].extend(te_ps.tolist())
                    test_sample_sum   += float(te_ps.sum())
                    test_sample_sqsum += float((te_ps ** 2).sum())
                    test_sample_cnt   += int(te_ps.size)

                    # TEST CSV
                    info_te  = [sample_info[i] for i in idx_te]

                    # CSV rows (no split/seed columns)
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
            # fixed: split train/val by subject within tv_sessions
            val_subs  = split.get("fixed_val_subjects", [])
            test_subs = split.get("fixed_test_subjects", val_subs)
            if not val_subs:
                raise ValueError("[fixed] split.fixed_val_subjects is empty.")

            # test subjects are also excluded from training
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

            # save learning curve CSV
            lc_df = pd.DataFrame({
                "epoch": list(range(1, epochs + 1)),
                "train_nrmse": hist["train_nrmse_curve"],
                "val_nrmse": hist["val_nrmse_curve"],
            })
            lc_csv_path = os.path.join(fold_dir, "learning_curve.csv")
            lc_df.to_csv(lc_csv_path, index=False)
            print(f"[csv] learning curve saved → {lc_csv_path}")

            # save normalisation statistics & config
            mu = scaler.mean_.reshape(1, 1, -1).astype(np.float32)
            sd = scaler.scale_.reshape(1, 1, -1).astype(np.float32)
            np.savez_compressed(os.path.join(fold_dir, "norm_stats.npz"), mu=mu, sd=sd)
            with open(os.path.join(fold_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

            # load LAST-epoch weights (same protocol as the LOSO path)
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

            # ----- FIXED VAL per-channel (display only) -----
            val_per_ch = nrmse_by_channel(vl_loader, model, device)  # (C,)
            print("[FIXED VAL per-channel NRMSE(range)]")
            for ci, v in enumerate(val_per_ch):
                nm = ch_names[ci]
                print(f"  - {nm}: {v:.6f}")

            # ----- FIXED VAL per-sample + subject-map accumulation -----
            val_ps_all = []
            val_subj_seq = [str(info_tv[i]["subject"]) for i in idx_vl_tv]  # same order as the DataLoader
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
                    # accumulate per subject
                    for j in range(B):
                        val_subject_map[val_subj_seq[offset + j]].append(float(nrmse_b[j]))
                    val_ps_all.append(nrmse_b)
                    offset += B
            val_ps = np.concatenate(val_ps_all, axis=0)
            val_sample_sum   += float(val_ps.sum())
            val_sample_sqsum += float((val_ps ** 2).sum())
            val_sample_cnt   += int(val_ps.size)

            # ===== TEST evaluation =====
            idx_te = _select_indices(sample_info, subjects=test_subs, sessions=te_sessions)
            if len(idx_te) == 0:
                print("[TEST] fixed: no test-subject samples in test_sessions. (skip)")
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

                # per-sample metrics + subject-map accumulation + prediction collection
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

        # per-seed validation curves are saved as CSV (figure export removed)

        # save the seed-level merged TEST CSV
        if test_csv_parts:
            df_all = pd.concat(test_csv_parts, ignore_index=True)
            csv_run_path = os.path.join(run_dir, test_csv_name)
            df_all.to_csv(csv_run_path, index=False)

        # -------- end of seed: pool into the grand accumulators (samples/subjects) --------
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

        # (note) per-seed summaries/printouts are intentionally omitted.

    # ======================
    # Final summary (pooled over all seeds)
    # ======================

    # per-sample statistics
    val_sample_mean,  val_sample_std  = _mean_std_from_sums_ddof1(grand_val_ps_sum,  grand_val_ps_sqsum,  grand_val_ps_cnt)
    test_sample_mean, test_sample_std = _mean_std_from_sums_ddof1(grand_test_ps_sum, grand_test_ps_sqsum, grand_test_ps_cnt)

    # per-subject statistics: subject means → mean/std (ddof=1) of that distribution
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

    # final summary JSON (no per-seed/per-fold keys)
    final_summary = {
        "experiment": exp_name,
        "mode": mode,
        "seeds": seeds,  # kept only as a reference to which seeds were used
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

    # --- save final summary (both LOSO and FIXED) ---
    summary_path = os.path.join(run_dir, "summary.json")  # run directory of the last seed
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved summary → {summary_path}")

    print_box("Grand summary (no per-seed)")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))

    # short human-readable summary lines
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