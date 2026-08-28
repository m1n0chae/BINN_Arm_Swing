"""
eval_day1_on_full.py
====================
day1_output/ 안의 학습된 모델들을 100% Day1 데이터로 evaluation.
case1, case3 등 특정 case의 모델만 필터링 가능.

Usage:
  python eval_day1_on_full.py --case 1
  python eval_day1_on_full.py --case 3

출력: day1_eval_outputs/<model_outdir>/Figure5_datasize/{percent}/<session>.csv
"""
from __future__ import annotations
import os, sys, argparse, time, json, re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 기존 모듈 재사용
from matlab_data_prep import load_all_mat_data
from utils_and_modules import build_dataset_from_mat
from eval_percent_models_on_full import (
    build_model, evaluate_run, _read_umi,
    SESSION_MAP, MODEL_OUTDIR_BY_UMI, ablation_outdir,
)


_KNOWN_SVARS = [
    # cross-session (긴 매칭부터)
    "tv_ss1__te_ss2ss3", "tv_ss2__te_ss1ss3", "tv_ss3__te_ss1ss2",
    # LOSO
    "tv_te_ss123", "tv_te_ss1", "tv_te_ss2", "tv_te_ss3",
]


def parse_variant(name):
    """run.name 에서 알려진 session_variant 중 매칭되는 것을 찾고,
    그 직전의 __ 세그먼트를 dvar로 추출.
    suffix (cnn / cnn_with_model / transformer)가 있어도, dvar 위치가 svar 바로 앞이라 OK."""
    for svar in _KNOWN_SVARS:
        if name.endswith("__" + svar):
            prefix = name[: -len("__" + svar)]
            parts = prefix.split("__")
            if len(parts) >= 2:
                return parts[-1], svar
    return None, None


# 경로. 저장소 기준 상대경로가 기본값이고 환경변수로 덮어쓸 수 있다.
#   RUNS_ROOT    학습 산출물(runs) 최상위
#   OUTPUT_ROOT  예측 CSV 출력 위치 (paper_data_processing.m 의 입력)
#   DATA_DIR     build_dataset.m 이 만든 .mat 폴더
ROOT = os.environ.get("RUNS_ROOT", "../runs")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "../eval_outputs")
DATA_DIR = os.environ.get("DATA_DIR", "../data")


def find_day1_full_mat():
    cand = {
        "grf_mat":      os.path.join(DATA_DIR, "merged_stride_grf_gyro_day1.mat"),
        "kin_mat":      os.path.join(DATA_DIR, "merged_stride_kinematics_arm_gyro_day1.mat"),
        "modeling_mat": os.path.join(DATA_DIR, "manual_stride_modeling_gyro_day1.mat"),
    }
    for k, p in cand.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Day1 full mat missing: {p}")
    print(f"[DATA] day1 full: {DATA_DIR}")
    return cand


# data_variant -> percent dir
DATA_PCT_DIR = {
    "percent005": "5percent",
    "percent010": "10percent",
    "percent020": "20percent",
    "percent050": "50percent",
    "full":       "100percent",
}


def output_csv_path_day1(model: str, dvar: str, svar: str,
                         ablation_mode=None, umi=False):
    """
    구조:
    - 100% (full): <model>/Figure3_loso/<ssX>.csv 또는 Figure4_speedloso/<train_ssX>.csv
    - percent (5/10/20/50%): <model>/Figure5_datasize/<percent>/<ssX>/case<N>.csv
                             <model>/Figure6_datasize_sploso/<percent>/<train_ssX>/case<N>.csv
    """
    if svar not in SESSION_MAP:
        return None
    m = re.match(r"(percent\d{3}|full)(_day1)?(_case(\d+))?", dvar)
    if not m:
        return None
    pct = m.group(1)
    case_num = m.group(4)  # "1", "2", ... or None
    fig, fname = SESSION_MAP[svar]
    sess_basename = fname.replace(".csv", "")  # "ss1", "train_ss1" 등

    if ablation_mode is not None:
        outdir = ablation_outdir(ablation_mode)
    else:
        outdir = MODEL_OUTDIR_BY_UMI.get((model, bool(umi)))
        if outdir is None:
            return None

    if pct == "full":
        # 100% LOSO/cross-session: flat 파일
        base_fig = "Figure3_loso" if fig == "Figure5_datasize" else "Figure4_speedloso"
        return os.path.join(OUTPUT_ROOT, outdir, base_fig, fname)
    else:
        if pct not in DATA_PCT_DIR:
            return None
        pct_dir = DATA_PCT_DIR[pct]
        case_label = f"case{case_num}" if case_num else "case1"
        # <percent>/<ssX>/<caseN>.csv
        return os.path.join(OUTPUT_ROOT, outdir, fig, pct_dir, sess_basename, f"{case_label}.csv")


def find_day1_runs(model_root, case_filter=None):
    """day1_output/<model>/*/ 안에서 runs 찾기. case_filter=1이면 case1만 (full 포함)."""
    by_var = {}
    if not os.path.isdir(model_root):
        return by_var
    for d in sorted(os.listdir(model_root)):
        full = os.path.join(model_root, d)
        if not (os.path.isdir(full) and re.match(r"^\d{8}-", d)):
            continue
        dvar, svar = parse_variant(d)
        if dvar is None:
            continue
        # case filter
        if case_filter is not None:
            if case_filter == 1:
                # case1 → full_day1 또는 percent*_day1_case1
                if not (dvar == "full_day1" or dvar.endswith(f"_day1_case1")):
                    continue
            else:
                if not dvar.endswith(f"_day1_case{case_filter}"):
                    continue
        # ckpt check
        ckpt_dir = os.path.join(full, "ckpt")
        if not os.path.isdir(ckpt_dir):
            continue
        valid = any(
            os.path.isfile(os.path.join(ckpt_dir, fd, "last.pth"))
            and os.path.isfile(os.path.join(ckpt_dir, fd, "norm_stats.npz"))
            for fd in os.listdir(ckpt_dir)
        )
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1, help="평가할 case (1/2/3/4/5). full 포함시 1로 두면 full+case1")
    ap.add_argument("--include_full", action="store_true",
                    help="100% (full_day1)도 함께 평가")
    ap.add_argument("--only_full", action="store_true",
                    help="100% (full_day1)만 평가 (case 무시)")
    ap.add_argument("--models_filter", nargs="+", default=None,
                    help="특정 모델만 평가 (예: cnn transformer proposed)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="이미 존재하는 출력 csv는 skip")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[case] {args.case}")

    paths = find_day1_full_mat()
    print("[load] day1 full mat...")
    grf, kin, mdl = load_all_mat_data(
        grf_path=paths["grf_mat"], grf_struct="stride_grf",
        kin_path=paths["kin_mat"], kin_struct="stride_kinematics_arm",
        modeling_path=paths["modeling_mat"], modeling_struct="stride_modeling",
    )

    dataset_cache = {}
    grand = {}

    # 모델별 처리
    models = ["cnn", "transformer", "binn"]
    ablation_root = os.path.join(ROOT, "binn")

    # 모델 필터 적용
    models_filter_set = set(args.models_filter) if args.models_filter else None

    # 1) baseline (cnn, transformer, binn)
    for model in models:
        if models_filter_set is not None and model not in models_filter_set:
            continue
        model_root = os.path.join(ROOT, model)
        if not os.path.isdir(model_root):
            print(f"[skip] {model_root} not found")
            continue
        # DEBUG: list dirs
        all_dirs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
        print(f"  [DEBUG] {model_root}: {len(all_dirs)} subdirs, first 3: {all_dirs[:3]}")
        for d in all_dirs[:3]:
            dv, sv = parse_variant(d)
            print(f"    {d[-60:]} → dv={dv!r}, sv={sv!r}")
        if args.only_full:
            full_runs = find_day1_runs(model_root, case_filter=None)
            runs = {k: v for k, v in full_runs.items() if k[0] == "full_day1"}
        else:
            runs = find_day1_runs(model_root, case_filter=args.case)
            if args.include_full:
                full_runs = find_day1_runs(model_root, case_filter=None)
                for k, v in full_runs.items():
                    if k[0] == "full_day1" and k not in runs:
                        runs[k] = v
        print(f"\n[{model}] {len(runs)} runs to evaluate")

        for (dvar, svar, umi), run_dir in sorted(runs.items()):
            tag = "with_model" if umi else "no_model"
            print(f"  [{model}/{tag} | {dvar} | {svar}]  {os.path.basename(run_dir)}")
            t0 = time.time()
            cfg_path = os.path.join(run_dir, "config_used.yaml")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            dset = cfg.get("dataset", {}) or {}
            cache_key = json.dumps(dset, sort_keys=True, default=str)
            if cache_key in dataset_cache:
                full_data = dataset_cache[cache_key]
            else:
                tv_sessions = dset.get("train_val_sessions", ["ss2"])
                te_sessions = dset.get("test_sessions", ["ss2"])
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

            out_csv = output_csv_path_day1(model, dvar, svar, umi=umi)
            if out_csv is None:
                print(f"    [skip] no output mapping")
                continue
            if args.skip_existing and os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
                print(f"    [skip] existing: {out_csv}")
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

            outdir = MODEL_OUTDIR_BY_UMI.get((model, bool(umi)), model)
            grand[(outdir, dvar, svar)] = {
                "n_folds": n_folds, "mean_nrmse": mn, "std_nrmse": sd,
                "run_dir": os.path.basename(run_dir),
            }

    # 2) ablation modes
    if os.path.isdir(ablation_root):
        for amode in os.listdir(ablation_root):
            if models_filter_set is not None and amode not in models_filter_set:
                continue
            mode_dir = os.path.join(ablation_root, amode)
            if not os.path.isdir(mode_dir):
                continue
            if args.only_full:
                full_runs = find_day1_runs(mode_dir, case_filter=None)
                runs = {k: v for k, v in full_runs.items() if k[0] == "full_day1"}
            else:
                runs = find_day1_runs(mode_dir, case_filter=args.case)
                if args.include_full:
                    full_runs = find_day1_runs(mode_dir, case_filter=None)
                    for k, v in full_runs.items():
                        if k[0] == "full_day1" and k not in runs:
                            runs[k] = v
            print(f"\n[ablation/{amode}] {len(runs)} runs")

            for (dvar, svar, umi), run_dir in sorted(runs.items()):
                print(f"  [{amode} | {dvar} | {svar}]  {os.path.basename(run_dir)}")
                t0 = time.time()
                cfg_path = os.path.join(run_dir, "config_used.yaml")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                dset = cfg.get("dataset", {}) or {}
                cache_key = json.dumps(dset, sort_keys=True, default=str)
                if cache_key in dataset_cache:
                    full_data = dataset_cache[cache_key]
                else:
                    tv_sessions = dset.get("train_val_sessions", ["ss2"])
                    te_sessions = dset.get("test_sessions", ["ss2"])
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
                out_csv = output_csv_path_day1("binn", dvar, svar, ablation_mode=amode)
                if out_csv is None:
                    continue
                if args.skip_existing and os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
                    print(f"    [skip] existing: {out_csv}")
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
                grand[(amode, dvar, svar)] = {
                    "n_folds": n_folds, "mean_nrmse": mn, "std_nrmse": sd,
                    "run_dir": os.path.basename(run_dir),
                }

    # summary
    rows = [{"model_outdir": k[0], "data_variant": k[1], "session_variant": k[2], **v}
            for k, v in grand.items()]
    if not rows:
        print("\n[OK] no new evaluations (all skipped or no models matched).")
        return
    df = pd.DataFrame(rows).sort_values(["model_outdir", "data_variant", "session_variant"])
    summary_dir = os.path.join(OUTPUT_ROOT, "_summary")
    os.makedirs(summary_dir, exist_ok=True)
    out = os.path.join(summary_dir, f"eval_case{args.case}_summary.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] summary → {out}")
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
