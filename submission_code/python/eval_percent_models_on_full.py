"""
Re-evaluate the percent_xxx-trained models (5/10/20/50%) on the FULL 100% data.

Target models:
- runs/cnn/
- runs/transformer/
- runs/binn/

For each (model, data_variant, session_variant) the latest run folder is picked →
each fold's last.pth + norm_stats.npz is loaded (final-epoch weights, matching the
paper protocol) →
the dataset is reassembled from the FULL .mat files →
inference is run on the held-out subject's test data →
eval CSVs are written in the same format as ss*.csv.

Output:
- per-run combined prediction CSV under OUTPUT_ROOT
- per-run summary (per-fold NRMSE)
"""
from __future__ import annotations
import os, glob, re, json, time
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from matlab_data_prep import load_all_mat_data
from utils_and_modules import build_dataset_from_mat
from models_cnn import CNNRegressor
from models_transformer import TransformerRegressor
from models_binn import BINN


# Root paths: repository-relative defaults, overridable via environment variables
ROOT = os.environ.get("RUNS_ROOT", "../runs")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "../eval_outputs")

MODELS_TO_EVAL = ["cnn", "transformer"]
# 35 cases = full + percent005/010/020/050 × 7 session splits
DATA_VARIANTS_TO_EVAL = ["full", "percent005", "percent010", "percent020", "percent050"]

# Ablation modes (for the 35-case evaluation)
# The three reported in the paper: proposed / w/o Cross-Attention / w/o CoM-to-GRF Mapper
ABLATION_MODES = ["proposed", "no_attn", "attn_only"]
DATA_VARIANTS_FULL_SET = ["full", "percent005", "percent010", "percent020", "percent050"]

# Output-folder mapping — branched on use_modeling_input (umi)
# umi=False (no modeling input) / umi=True (modeling input used → _with_model)
MODEL_OUTDIR_BY_UMI = {
    ("cnn",              False): "CNN",
    ("cnn",              True):  "CNN_with_model",
    ("transformer",      False): "Transformer",
    ("transformer",      True):  "Transformer_with_model",
    ("binn", False): "Physres_proposed",        # in practice umi is almost always True; mapped defensively
    ("binn", True):  "Physres_proposed",
}
# Ablation: the folder name is the mode name itself (no Ablation_ prefix)
def ablation_outdir(mode):
    return mode

# data_variant -> percent folder name
DATA_PCT_DIR = {
    "percent005": "5percent",
    "percent010": "10percent",
    "percent020": "20percent",
    "percent050": "50percent",
}

# session_variant -> (Figure, filename)
SESSION_MAP = {
    "tv_te_ss1":         ("Figure5_datasize",         "ss1.csv"),
    "tv_te_ss2":         ("Figure5_datasize",         "ss2.csv"),
    "tv_te_ss3":         ("Figure5_datasize",         "ss3.csv"),
    "tv_te_ss123":       ("Figure5_datasize",         "ss123.csv"),
    "tv_ss1__te_ss2ss3": ("Figure6_datasize_sploso",  "train_ss1.csv"),
    "tv_ss2__te_ss1ss3": ("Figure6_datasize_sploso",  "train_ss2.csv"),
    "tv_ss3__te_ss1ss2": ("Figure6_datasize_sploso",  "train_ss3.csv"),
}


def output_csv_path(model: str, dvar: str, svar: str,
                    ablation_mode: str = None, umi: bool = False) -> str:
    if svar not in SESSION_MAP:
        return None
    # full is supported too: uses the 100percent folder
    if dvar == "full":
        pct_dir = "100percent"
    elif dvar in DATA_PCT_DIR:
        pct_dir = DATA_PCT_DIR[dvar]
    else:
        return None
    fig, fname = SESSION_MAP[svar]
    if ablation_mode is not None:
        outdir = ablation_outdir(ablation_mode)
    else:
        outdir = MODEL_OUTDIR_BY_UMI.get((model, bool(umi)))
        if outdir is None:
            return None
    return os.path.join(OUTPUT_ROOT, outdir, fig, pct_dir, fname)

# FULL (100%) dataset path
DATA_DIR_CANDIDATES = [os.environ.get("DATA_DIR", "../data")]


def find_full_mat_paths():
    """Locate the 100% full .mat files (the base files without a suffix)."""
    for base in DATA_DIR_CANDIDATES:
        base_abs = os.path.abspath(os.path.join(SCRIPT_DIR, base)) if not os.path.isabs(base) else base
        cand = {
            "grf_mat":      os.path.join(base_abs, "merged_stride_grf_gyro.mat"),
            "kin_mat":      os.path.join(base_abs, "merged_stride_kinematics_arm_gyro.mat"),
            "modeling_mat": os.path.join(base_abs, "merged_stride_modeling_gyro.mat"),
        }
        if all(os.path.isfile(p) for p in cand.values()):
            print(f"[DATA] full mat dir: {base_abs}")
            return cand
    raise FileNotFoundError("Cannot find full 100% .mat files. Edit DATA_DIR_CANDIDATES in the script.")


def parse_variant(name):
    m = re.search(r"loso_speed_loso(__.+?)?(__.+?)?$", name)
    if m and m.group(1) and m.group(2):
        return m.group(1).strip("_"), m.group(2).strip("_")
    return None, None


def _read_umi(run_dir: str):
    """Read use_modeling_input from the run's config_used.yaml. None if missing."""
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return bool((cfg.get("dataset", {}) or {}).get("use_modeling_input", False))
    except Exception:
        return None


def find_latest_runs(model_root: str, allow_full: bool = True) -> Dict[Tuple[str, str, bool], str]:
    """Latest run dir per (data_variant, session_variant, use_modeling_input).
    use_modeling_input (umi) is part of the key so that both variants of the same
    dvar/svar are evaluated. Only runs with per-fold last.pth + norm_stats.npz."""
    by_var = {}
    if not os.path.isdir(model_root):
        return by_var
    valid_dvars = set(DATA_VARIANTS_FULL_SET) if allow_full else set(DATA_VARIANTS_TO_EVAL)
    for d in sorted(os.listdir(model_root)):
        full = os.path.join(model_root, d)
        if not (os.path.isdir(full) and re.match(r"^\d{8}-", d)):
            continue
        dvar, svar = parse_variant(d)
        if dvar is None or dvar not in valid_dvars:
            continue
        # check that last.pth + norm_stats.npz exist
        ckpt_dir = os.path.join(full, "ckpt")
        if not os.path.isdir(ckpt_dir):
            continue
        valid = False
        for fd in os.listdir(ckpt_dir):
            fp = os.path.join(ckpt_dir, fd)
            if (os.path.isdir(fp)
                and os.path.isfile(os.path.join(fp, "last.pth"))
                and os.path.isfile(os.path.join(fp, "norm_stats.npz"))):
                valid = True
                break
        if not valid:
            continue
        umi = _read_umi(full)
        if umi is None:
            continue
        ts = d.split("_")[0]
        key = (dvar, svar, umi)
        if key not in by_var or ts > os.path.basename(by_var[key]).split("_")[0]:
            by_var[key] = full
    return by_var


def build_model(cfg: Dict, in_ch: int, out_ch: int, model_name: str) -> torch.nn.Module:
    """Build the model from config_used.yaml. Partial reimplementation of main.py:build_model_from_cfg."""
    m = cfg.get("model", {}) or {}
    mtype = (m.get("type") or model_name).lower()
    if mtype == "cnn":
        hp = m.get("cnn", {}) or {}
        return CNNRegressor(
            in_dim=in_ch, out_dim=out_ch,
            channels=tuple(hp.get("channels", [128, 128, 64])),
            kernels=tuple(hp.get("kernels", [9, 5, 3])),
            dropout=float(hp.get("dropout", 0.0)),
        )
    if mtype == "transformer":
        hp = m.get("transformer", {}) or {}
        return TransformerRegressor(
            in_dim=in_ch, out_dim=out_ch,
            d_model=hp.get("d_model", 128),
            nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 3),
            dim_ff=hp.get("dim_ff", 256),
            drop=hp.get("dropout", 0.1),
        )
    if mtype == "binn":
        hp = m.get("binn", {}) or {}
        ablation_mode = hp.get("ablation_mode", "proposed")
        ablation_mode = {"binn": "proposed", "no_physics": "proposed"}.get(
            str(ablation_mode).lower(), str(ablation_mode).lower())
        wrist_dim  = int(hp.get("wrist_dim", 0))
        sacrum_dim = int(hp.get("sacrum_dim", 0))
        # per-mode decoder channels
        dec_by_mode = hp.get("decoder_channels_by_mode", {})
        if ablation_mode in dec_by_mode:
            dec_ch = tuple(dec_by_mode[ablation_mode])
        else:
            dec_ch = tuple(hp.get("decoder_channels", [128, 64]))
        return BINN(
            in_dim=in_ch, out_dim=out_ch,
            wrist_dim=wrist_dim, sacrum_dim=sacrum_dim,
            ablation_mode=ablation_mode,
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
    raise ValueError(f"Unknown model: {mtype}")


def nrmse_range_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """per-sample NRMSE(range), shape (N,)"""
    err = y_pred - y_true
    rmse = np.sqrt(((err ** 2).mean(axis=1) + 1e-12))  # (N, C) -- mean over time
    rng  = (y_true.max(axis=1) - y_true.min(axis=1)).clip(min=1e-8)  # (N, C)
    return (rmse / rng).mean(axis=1)  # (N,)


def evaluate_run(run_dir: str, full_data: Tuple, model_name: str, device,
                 output_csv: str) -> Dict:
    """Evaluate every fold of one run. full_data = (X, y, sample_info, F, C, T).
    Results are combined into a single output_csv (same format as ss*.csv)."""
    X_full, y_full, info_full, F, C, T = full_data

    # load the config
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dset = cfg.get("dataset", {}) or {}
    te_sessions = dset.get("test_sessions", dset.get("sessions", []))
    grf_axes = dset.get("grf_axes", ["Fx", "Fz"])
    modeling_used = bool(dset.get("use_modeling_input", False))

    fold_dirs = sorted(d for d in os.listdir(os.path.join(run_dir, "ckpt"))
                       if d.startswith("fold_"))

    all_rows = []
    summary = {"folds": {}, "model": model_name, "run_dir": os.path.basename(run_dir)}
    for fold in fold_dirs:
        subj = fold.replace("fold_", "")
        fd = os.path.join(run_dir, "ckpt", fold)
        ckpt = os.path.join(fd, "last.pth")   # final-epoch weights (paper protocol)
        ns   = os.path.join(fd, "norm_stats.npz")
        if not (os.path.isfile(ckpt) and os.path.isfile(ns)):
            continue

        # test data of the held-out subject (full data)
        idx_te = [i for i, s in enumerate(info_full)
                  if str(s.get("subject", "")) == subj
                  and str(s.get("session", "")) in set(map(str, te_sessions))]
        if not idx_te:
            print(f"  [{fold}] no test samples in full data, skip")
            continue
        X_te = X_full[idx_te]; Y_te = y_full[idx_te]
        info_te = [info_full[i] for i in idx_te]

        # apply the scaler fitted at training time
        ns_data = np.load(ns)
        mu = ns_data["mu"].reshape(1, 1, -1)
        sd = ns_data["sd"].reshape(1, 1, -1)
        Xt_sc = ((X_te - mu) / sd).astype(np.float32)

        # build the model + load the weights
        try:
            model = build_model(cfg, in_ch=F, out_ch=C, model_name=model_name).to(device)
        except Exception as e:
            print(f"  [{fold}] build_model error: {e}")
            continue
        try:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"  [{fold}] load_state_dict error: {e}")
            continue
        model.eval()

        # inference
        ds = TensorDataset(torch.from_numpy(Xt_sc), torch.from_numpy(Y_te.astype(np.float32)))
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yhat = model(xb).cpu().numpy()
                preds.append(yhat)
        ypred = np.concatenate(preds, axis=0)  # (N, T, C)

        # per-sample NRMSE
        nrmse_per = nrmse_range_per_sample(Y_te, ypred)
        summary["folds"][subj] = {
            "n_samples": int(len(idx_te)),
            "nrmse_mean": float(np.nanmean(nrmse_per)),
            "nrmse_std":  float(np.nanstd(nrmse_per, ddof=1)) if len(nrmse_per) > 1 else float("nan"),
            "nrmse_median": float(np.nanmedian(nrmse_per)),
        }

        # CSV rows
        rows = []
        for n in range(Y_te.shape[0]):
            meta = info_te[n]
            for t in range(Y_te.shape[1]):
                r = {
                    "fold": fold, "subject": subj,
                    "test session": str(meta.get("session", "")),
                    "day":          str(meta.get("day", "")),
                    "modeling_used": modeling_used,
                    "stride_idx":  int(meta.get("stride_idx", -1)),
                    "t": int(t),
                }
                for ci, ax in enumerate(grf_axes):
                    if ci < Y_te.shape[2]:
                        r[f"y_true_{ax}"] = float(Y_te[n, t, ci])
                        r[f"y_pred_{ax}"] = float(ypred[n, t, ci])
                rows.append(r)
        df_fold = pd.DataFrame(rows)
        all_rows.append(df_fold)

        # cleanup
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # combined CSV → merged output folder
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_all.to_csv(output_csv, index=False)
        summary["output_csv"] = output_csv

    # summary
    if summary["folds"]:
        per_fold = list(summary["folds"].values())
        summary["overall"] = {
            "n_folds": len(per_fold),
            "mean_nrmse": float(np.nanmean([f["nrmse_mean"] for f in per_fold])),
            "std_nrmse":  float(np.nanstd([f["nrmse_mean"] for f in per_fold], ddof=1)) if len(per_fold) > 1 else float("nan"),
        }

    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to run; all if omitted (cnn transformer binn)")
    ap.add_argument("--data_variants", nargs="+", default=None,
                    help="data variants to run (e.g. percent005 percent010 full)")
    ap.add_argument("--ablation_modes", nargs="+", default=None,
                    help="ablation modes (e.g. proposed no_attn); enables ablation evaluation")
    ap.add_argument("--session_variants", nargs="+", default=None,
                    help="session variants to run (e.g. tv_te_ss1 tv_te_ss2)")
    args = ap.parse_args()
    session_variants_filter = set(args.session_variants) if args.session_variants else None

    is_ablation = args.ablation_modes is not None and len(args.ablation_modes) > 0
    if args.models is not None:
        models_to_run = args.models
    elif is_ablation:
        models_to_run = []  # ablation only → baselines are not run
    else:
        models_to_run = MODELS_TO_EVAL
    # all 35 cases (full included) is the default
    data_variants_filter = set(args.data_variants) if args.data_variants else set(DATA_VARIANTS_FULL_SET)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[models] {models_to_run}")
    full_paths = find_full_mat_paths()

    # load the FULL data (once)
    print("[load] full mat ...")
    grf, kin, mdl = load_all_mat_data(
        grf_path=full_paths["grf_mat"], grf_struct="stride_grf",
        kin_path=full_paths["kin_mat"], kin_struct="stride_kinematics_arm",
        modeling_path=full_paths["modeling_mat"], modeling_struct="stride_modeling",
    )

    # ====== dataset reassembly per model + variant (following the config's dataset section) ======
    # datasets are identical within a dataset_variant, so cache them
    dataset_cache: Dict[str, Tuple] = {}

    # merged summary
    grand_summary = {}

    # ===== ablation-mode evaluation =====
    if is_ablation:
        ablation_root_base = os.path.join(ROOT, "binn")
        for amode in args.ablation_modes:
            print(f"\n========== ABLATION MODE: {amode} ==========")
            # find the run folders of this mode (umi key ignored — ablation is always umi=True)
            mode_dir = os.path.join(ablation_root_base, amode)
            runs_by_key = find_latest_runs(mode_dir) if os.path.isdir(mode_dir) else {}
            # reduce (dvar, svar, umi) -> run_dir to (dvar, svar) -> run_dir (latest first)
            runs = {}
            for (dv, sv, umi), rd in runs_by_key.items():
                k = (dv, sv)
                ts = os.path.basename(rd).split("_")[0]
                if k not in runs or ts > os.path.basename(runs[k]).split("_")[0]:
                    runs[k] = rd
            # from the parent level, include only runs whose actual ablation_mode matches amode
            for d in sorted(os.listdir(ablation_root_base)):
                full = os.path.join(ablation_root_base, d)
                if not (os.path.isdir(full) and re.match(r"^\d{8}-", d)):
                    continue
                dvar, svar = parse_variant(d)
                if dvar is None or dvar not in data_variants_filter:
                    continue
                ckpt_dir = os.path.join(full, "ckpt")
                if not os.path.isdir(ckpt_dir):
                    continue
                # check the actual ablation_mode in the config
                cfg_p = os.path.join(full, "config_used.yaml")
                if not os.path.isfile(cfg_p):
                    continue
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        cfg_chk = yaml.safe_load(f)
                    actual_mode = (cfg_chk.get("model", {}) or {}).get(
                        "binn", {}).get("ablation_mode", None)
                except Exception:
                    actual_mode = None
                if actual_mode != amode:
                    continue
                has_ckpt = any(
                    os.path.isfile(os.path.join(ckpt_dir, fd, "last.pth"))
                    and os.path.isfile(os.path.join(ckpt_dir, fd, "norm_stats.npz"))
                    for fd in os.listdir(ckpt_dir)
                )
                if not has_ckpt:
                    continue
                ts = d.split("_")[0]
                key = (dvar, svar)
                if key not in runs or ts > os.path.basename(runs[key]).split("_")[0]:
                    runs[key] = full
            print(f"  found {len(runs)} sweep runs (target: 35)")

            for (dvar, svar), run_dir in sorted(runs.items()):
                if dvar not in data_variants_filter:
                    continue
                print(f"\n  [ablation/{amode} | {dvar} | {svar}]  {os.path.basename(run_dir)}")
                t0 = time.time()
                cfg_path = os.path.join(run_dir, "config_used.yaml")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                dset = cfg.get("dataset", {}) or {}
                cache_key = json.dumps(dset, sort_keys=True, default=str)
                if cache_key in dataset_cache:
                    full_data = dataset_cache[cache_key]
                else:
                    tv_sessions = dset.get("train_val_sessions", dset.get("sessions", ["ss2"]))
                    te_sessions = dset.get("test_sessions",      dset.get("sessions", ["ss2"]))
                    all_sessions = sorted(list(set(tv_sessions) | set(te_sessions)))
                    X, y, info = build_dataset_from_mat(
                        grf=grf, kin=kin,
                        modeling=mdl if dset.get("use_modeling_input") else {},
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
                    full_data = (X, y, info, X.shape[-1], y.shape[-1], X.shape[1])
                    dataset_cache[cache_key] = full_data

                out_csv = output_csv_path("binn", dvar, svar, ablation_mode=amode)
                if out_csv is None:
                    continue
                try:
                    summary = evaluate_run(run_dir, full_data, "binn",
                                           device, out_csv)
                except Exception as e:
                    print(f"    [error] {e}")
                    continue
                n_folds = len(summary.get("folds", {}))
                ovr = summary.get("overall", {})
                mn = ovr.get("mean_nrmse", float("nan"))
                sd = ovr.get("std_nrmse", float("nan"))
                print(f"    -> {n_folds} folds, NRMSE = {mn:.5f} ± {sd:.5f}  ({time.time()-t0:.1f}s)")
                grand_summary[(f"ablation/{amode}", dvar, svar)] = {
                    "n_folds": n_folds, "mean_nrmse": mn, "std_nrmse": sd,
                    "run_dir": os.path.basename(run_dir), "output_csv": out_csv,
                }

    # ===== baseline-model evaluation =====
    allow_full_for_models = "full" in data_variants_filter
    for model in models_to_run:
        model_root = os.path.join(ROOT, model)
        runs = find_latest_runs(model_root, allow_full=allow_full_for_models)
        print(f"\n[{model}] {len(runs)} sweep runs to re-evaluate (including umi variants)")

        for (dvar, svar, umi), run_dir in sorted(runs.items()):
            if dvar not in data_variants_filter:
                continue
            if session_variants_filter is not None and svar not in session_variants_filter:
                continue
            tag = "with_model" if umi else "no_model"
            print(f"\n  [{model}/{tag} | {dvar} | {svar}]  {os.path.basename(run_dir)}")
            t0 = time.time()

            # load the config → build the full dataset (the percent run's dataset config is reused
            # verbatim; the subsampling lived in the .mat files, and here the full .mat is used instead)
            cfg_path = os.path.join(run_dir, "config_used.yaml")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            dset = cfg.get("dataset", {}) or {}

            cache_key = json.dumps(dset, sort_keys=True, default=str)
            if cache_key in dataset_cache:
                full_data = dataset_cache[cache_key]
            else:
                tv_sessions = dset.get("train_val_sessions", dset.get("sessions", ["ss2"]))
                te_sessions = dset.get("test_sessions",      dset.get("sessions", ["ss2"]))
                all_sessions = sorted(list(set(tv_sessions) | set(te_sessions)))
                X, y, info = build_dataset_from_mat(
                    grf=grf, kin=kin,
                    modeling=mdl if dset.get("use_modeling_input") else {},
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
                full_data = (X, y, info, X.shape[-1], y.shape[-1], X.shape[1])
                dataset_cache[cache_key] = full_data
                print(f"    [data] X={X.shape}, y={y.shape}, samples={len(info)}")

            out_csv = output_csv_path(model, dvar, svar, umi=umi)
            if out_csv is None:
                print(f"    [skip] no output mapping for ({model}, {dvar}, {svar}, umi={umi})")
                continue
            try:
                summary = evaluate_run(run_dir, full_data, model, device, out_csv)
            except Exception as e:
                print(f"    [error] {e}")
                continue
            n_folds = len(summary.get("folds", {}))
            ovr = summary.get("overall", {})
            mn = ovr.get("mean_nrmse", float("nan"))
            sd = ovr.get("std_nrmse", float("nan"))
            print(f"    -> {n_folds} folds, NRMSE = {mn:.5f} ± {sd:.5f}  ({time.time()-t0:.1f}s)")
            print(f"       saved -> {out_csv}")

            model_label = MODEL_OUTDIR_BY_UMI.get((model, bool(umi)), model)
            grand_summary[(model_label, dvar, svar)] = {
                "n_folds": n_folds,
                "mean_nrmse": mn,
                "std_nrmse": sd,
                "run_dir": os.path.basename(run_dir),
                "output_csv": out_csv,
            }

    # merged table
    rows = []
    for (model, dvar, svar), v in grand_summary.items():
        rows.append({"model": model, "data_variant": dvar, "session_variant": svar, **v})
    grand_df = pd.DataFrame(rows).sort_values(["model", "data_variant", "session_variant"])
    out_csv = os.path.join(ROOT, "binn", "_paper_figures", "eval_full_grand_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    grand_df.to_csv(out_csv, index=False)
    print(f"\n[OK] grand summary -> {out_csv}")
    print(grand_df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
