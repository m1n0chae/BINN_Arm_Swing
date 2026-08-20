"""
percent_xxx (5/10/20/50) 학습된 모델들을 FULL 100% 데이터에서 재검증.

대상 모델:
- runs/cnn/
- runs/transformer/
- runs/physres_cnn_attn/

각 (model, data_variant, session_variant)의 최신 run 폴더 선택 →
각 fold의 last.pth + norm_stats.npz를 로드  [revision] 논문 프로토콜(final epoch)에 맞춤 →
FULL .mat 데이터로 dataset 재조립 →
held-out subject의 test 데이터에서 inference →
ss*.csv와 동일 포맷으로 eval_full.csv 저장.

출력:
- 각 fold/eval_full.csv (per-stride predictions)
- 각 run/eval_full.csv (combined)
- 각 run/eval_full_summary.json (per-fold NRMSE)
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
from models_physres_cnn_attn import PhysResidualCNNAttnRegressor
from models_physres_cnn_attn_ablation import PhysResidualCNNAttnRegressorAblation


# Plat 별 root 경로 자동 설정 (Windows 로컬 vs SSH 원격)
ROOT = os.environ.get("RUNS_ROOT", "../runs")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "../eval_outputs")

MODELS_TO_EVAL = ["cnn", "transformer"]
# 35 cases = full + percent005/010/020/050 × 7 session splits
DATA_VARIANTS_TO_EVAL = ["full", "percent005", "percent010", "percent020", "percent050"]

# Ablation 모드 (35 cases 평가용)
# 논문에 보고된 세 가지: 제안 / w-o Cross-Attention / w-o CoM-to-GRF Mapper
ABLATION_MODES = ["proposed", "no_attn", "attn_only"]
DATA_VARIANTS_FULL_SET = ["full", "percent005", "percent010", "percent020", "percent050"]

# 출력 폴더 매핑 — use_modeling_input(umi)로 분기
# umi=False (modeling 미사용) / umi=True (modeling 사용 → _with_model)
MODEL_OUTDIR_BY_UMI = {
    ("cnn",              False): "CNN",
    ("cnn",              True):  "CNN_with_model",
    ("transformer",      False): "Transformer",
    ("transformer",      True):  "Transformer_with_model",
    ("physres_cnn_attn", False): "Physres_proposed",        # 실제로는 거의 umi=True지만 안전하게 매핑
    ("physres_cnn_attn", True):  "Physres_proposed",
}
# Ablation: 폴더명은 mode 이름 그대로 (Ablation_ prefix 없음)
def ablation_outdir(mode):
    return mode

# data_variant -> percent 폴더명
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
    # full도 지원: 100percent 폴더 사용
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

# FULL 데이터셋 (100%) 경로 — Windows / Linux 자동 분기
DATA_DIR_CANDIDATES = [os.environ.get("DATA_DIR", "../data")]


def find_full_mat_paths():
    """100% full mat 파일 경로 찾기 (suffix 없는 기본 파일)"""
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
    """run의 config_used.yaml에서 use_modeling_input 읽기. 없으면 None."""
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
    """모델별 (data_variant, session_variant, use_modeling_input) → 최신 run dir.
    use_modeling_input(umi)을 키에 포함시켜 같은 dvar/svar의 두 변형을 모두 평가.
    last.pth + norm_stats.npz가 fold별로 있는 것만.  [revision]"""
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
        # last.pth + norm_stats.npz 존재 여부 확인
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
    """config_used.yaml 기반 모델 생성. main.py:build_model_from_cfg 일부 재구현."""
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
    if mtype == "physres_cnn_attn":
        hp = m.get("physres_cnn_attn", {}) or {}
        wrist_dim  = int(hp.get("wrist_dim", 0))
        sacrum_dim = int(hp.get("sacrum_dim", 0))
        return PhysResidualCNNAttnRegressor(
            in_dim=in_ch, out_dim=out_ch,
            wrist_dim=wrist_dim, sacrum_dim=sacrum_dim,
            wrist_channels=tuple(hp.get("wrist_channels", [64, 64])),
            wrist_kernels=tuple(hp.get("wrist_kernels", [9, 5])),
            sacrum_channels=tuple(hp.get("sacrum_channels", [32, 32])),
            sacrum_kernels=tuple(hp.get("sacrum_kernels", [9, 5])),
            embed_dim=int(hp.get("embed_dim", 64)),
            attn_heads=int(hp.get("attn_heads", 1)),
            decoder_channels=tuple(hp.get("decoder_channels", [128, 64])),
            decoder_kernels=tuple(hp.get("decoder_kernels", [5, 3])),
            phys_smooth_kernel=int(hp.get("phys_smooth_kernel", 5)),
            phys_scale_init=float(hp.get("phys_scale_init", 1.0)),
            dropout=float(hp.get("dropout", 0.1)),
            include_phys_in_residual=bool(hp.get("include_phys_in_residual", True)),
            extra_to_wrist=bool(hp.get("extra_to_wrist", True)),
        )
    if mtype == "physres_cnn_attn_ablation":
        hp = m.get("physres_cnn_attn", {}) or {}
        ablation_mode = hp.get("ablation_mode", "proposed")
        ablation_mode = {"binn": "proposed", "no_physics": "proposed"}.get(
            str(ablation_mode).lower(), str(ablation_mode).lower())
        wrist_dim  = int(hp.get("wrist_dim", 0))
        sacrum_dim = int(hp.get("sacrum_dim", 0))
        # 모드별 decoder channels
        dec_by_mode = hp.get("decoder_channels_by_mode", {})
        if ablation_mode in dec_by_mode:
            dec_ch = tuple(dec_by_mode[ablation_mode])
        else:
            dec_ch = tuple(hp.get("decoder_channels", [128, 64]))
        return PhysResidualCNNAttnRegressorAblation(
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
    rmse = np.sqrt(((err ** 2).mean(axis=1) + 1e-12))  # (N, C) -- 시간축 평균
    rng  = (y_true.max(axis=1) - y_true.min(axis=1)).clip(min=1e-8)  # (N, C)
    return (rmse / rng).mean(axis=1)  # (N,)


def evaluate_run(run_dir: str, full_data: Tuple, model_name: str, device,
                 output_csv: str) -> Dict:
    """한 run의 모든 fold를 평가. full_data = (X, y, sample_info, F, C, T) full 데이터셋.
    결과는 output_csv 1개 파일에 통합 저장 (기존 ss*.csv 형식)."""
    X_full, y_full, info_full, F, C, T = full_data

    # config 로드
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

        # held-out subject의 test 데이터 (full)
        idx_te = [i for i, s in enumerate(info_full)
                  if str(s.get("subject", "")) == subj
                  and str(s.get("session", "")) in set(map(str, te_sessions))]
        if not idx_te:
            print(f"  [{fold}] no test samples in full data, skip")
            continue
        X_te = X_full[idx_te]; Y_te = y_full[idx_te]
        info_te = [info_full[i] for i in idx_te]

        # 학습때 fit한 scaler 적용
        ns_data = np.load(ns)
        mu = ns_data["mu"].reshape(1, 1, -1)
        sd = ns_data["sd"].reshape(1, 1, -1)
        Xt_sc = ((X_te - mu) / sd).astype(np.float32)

        # 모델 생성 + 가중치 로드
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

    # combined CSV → 통합 출력 폴더로
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
                    help="실행할 모델 목록. 지정 안하면 전부 (cnn transformer physres_cnn_attn)")
    ap.add_argument("--data_variants", nargs="+", default=None,
                    help="실행할 data variant 목록 (예: percent005 percent010 full)")
    ap.add_argument("--ablation_modes", nargs="+", default=None,
                    help="ablation 모드 목록 (예: proposed no_attn). 지정하면 ablation 평가 모드.")
    ap.add_argument("--session_variants", nargs="+", default=None,
                    help="실행할 session variant 목록 (예: tv_te_ss1 tv_te_ss2)")
    args = ap.parse_args()
    session_variants_filter = set(args.session_variants) if args.session_variants else None

    is_ablation = args.ablation_modes is not None and len(args.ablation_modes) > 0
    if args.models is not None:
        models_to_run = args.models
    elif is_ablation:
        models_to_run = []  # ablation만 → baseline 미실행
    else:
        models_to_run = MODELS_TO_EVAL
    # 항상 35 cases (full 포함)을 default로
    data_variants_filter = set(args.data_variants) if args.data_variants else set(DATA_VARIANTS_FULL_SET)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[models] {models_to_run}")
    full_paths = find_full_mat_paths()

    # FULL 데이터 로드 (한 번만)
    print("[load] full mat ...")
    grf, kin, mdl = load_all_mat_data(
        grf_path=full_paths["grf_mat"], grf_struct="stride_grf",
        kin_path=full_paths["kin_mat"], kin_struct="stride_kinematics_arm",
        modeling_path=full_paths["modeling_mat"], modeling_struct="stride_modeling",
    )

    # ====== 모델별 + variant별로 데이터셋 재조립 (config의 dataset 섹션 따라) ======
    # 같은 dataset_variant 내에선 데이터셋 같으므로 캐시
    dataset_cache: Dict[str, Tuple] = {}

    # 통합 summary
    grand_summary = {}

    # ===== Ablation 모드 평가 =====
    if is_ablation:
        ablation_root_base = os.path.join(ROOT, "physres_cnn_attn_ablation")
        for amode in args.ablation_modes:
            print(f"\n========== ABLATION MODE: {amode} ==========")
            # 해당 모드의 run 폴더 찾기 (umi 키는 무시 — ablation은 모두 umi=True)
            mode_dir = os.path.join(ablation_root_base, amode)
            runs_by_key = find_latest_runs(mode_dir) if os.path.isdir(mode_dir) else {}
            # (dvar, svar, umi) -> run_dir 을 (dvar, svar) -> run_dir 로 reduce (latest umi=True 우선)
            runs = {}
            for (dv, sv, umi), rd in runs_by_key.items():
                k = (dv, sv)
                ts = os.path.basename(rd).split("_")[0]
                if k not in runs or ts > os.path.basename(runs[k]).split("_")[0]:
                    runs[k] = rd
            # parent-level 폴더에서 실제 ablation_mode가 amode와 일치하는 run만 포함
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
                # config의 실제 ablation_mode 확인
                cfg_p = os.path.join(full, "config_used.yaml")
                if not os.path.isfile(cfg_p):
                    continue
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        cfg_chk = yaml.safe_load(f)
                    actual_mode = (cfg_chk.get("model", {}) or {}).get(
                        "physres_cnn_attn", {}).get("ablation_mode", None)
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

                out_csv = output_csv_path("physres_cnn_attn_ablation", dvar, svar, ablation_mode=amode)
                if out_csv is None:
                    continue
                try:
                    summary = evaluate_run(run_dir, full_data, "physres_cnn_attn_ablation",
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

    # ===== 일반 baseline 모델 평가 =====
    allow_full_for_models = "full" in data_variants_filter
    for model in models_to_run:
        model_root = os.path.join(ROOT, model)
        runs = find_latest_runs(model_root, allow_full=allow_full_for_models)
        print(f"\n[{model}] {len(runs)} sweep runs to re-evaluate (umi 분기 포함)")

        for (dvar, svar, umi), run_dir in sorted(runs.items()):
            if dvar not in data_variants_filter:
                continue
            if session_variants_filter is not None and svar not in session_variants_filter:
                continue
            tag = "with_model" if umi else "no_model"
            print(f"\n  [{model}/{tag} | {dvar} | {svar}]  {os.path.basename(run_dir)}")
            t0 = time.time()

            # config 로드 → full 데이터셋 빌드 (dataset config는 percent run의 것을 그대로 사용,
            # 단 sample 축소가 mat 파일에서 됐으면 우리는 full mat을 쓰는 것으로 우회)
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

    # 통합 표
    rows = []
    for (model, dvar, svar), v in grand_summary.items():
        rows.append({"model": model, "data_variant": dvar, "session_variant": svar, **v})
    grand_df = pd.DataFrame(rows).sort_values(["model", "data_variant", "session_variant"])
    out_csv = os.path.join(ROOT, "physres_cnn_attn_ablation", "_paper_figures", "eval_full_grand_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    grand_df.to_csv(out_csv, index=False)
    print(f"\n[OK] grand summary -> {out_csv}")
    print(grand_df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
