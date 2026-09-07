#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kinematics-model parameter-level sensitivity — forearm amplitude A_WE mis-estimation.

Background
    The forearm swing amplitude is underestimated at the slowest walking speed
    (Section III-A). This script quantifies how such a kinematic bias propagates
    to the final GRF estimate.

    eval_noise_r_targeted.py corrupts the FINISHED CoM signal. This script goes
    one level up and mis-sets the KINEMATICS-MODEL PARAMETER itself:
    perturb_forearm_amplitude.m sets A_WE -> AMP_SCALE * A_WE, re-solves
    Eq. (1)-(3), and writes stride_modeling_weamp_<NNN>.mat in advance.
    (By Eq. (2), A_ES and A_SC scale by the same ratio; the angular offsets are
    constants and stay unchanged.)

    The scale factor passes through the cos/sin terms, so the waveform SHAPE and
    its correlation genuinely change — fundamentally different from multiplying
    the signal by a constant (which leaves r untouched).

Evaluation
    No retraining. The checkpoints trained on the unperturbed modeling data are
    reused as-is (final-epoch weights, last.pth); only the input CoM channels are
    swapped for the perturbed version at inference.

Logging
    Alongside the GRF accuracy (NRMSE, r), the change of the perturbed input CoM
    relative to the original (in_r_AP / in_r_AP_pre / in_r_AP_contra) is recorded
    so it can be compared against Table III.

Usage
    PERTURB_DIR=../data/perturb python eval_amp_perturbation.py \
        --runs_root <trained_runs> --out_csv amp_results.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from matlab_data_prep import load_all_mat_data
from utils_and_modules import build_dataset_from_mat
from main import build_model_from_cfg, _select_indices

TRAIN_SESS = {
    "tv_te_ss1": "ss1", "tv_te_ss2": "ss2", "tv_te_ss3": "ss3",
    "tv_ss1__te_ss2ss3": "ss1", "tv_ss2__te_ss1ss3": "ss2", "tv_ss3__te_ss1ss2": "ss3",
    "tv_te_ss123": "ss123",
}
OOD = {"tv_ss1__te_ss2ss3", "tv_ss2__te_ss1ss3", "tv_ss3__te_ss1ss2"}

# perturbed modeling files written by perturb_forearm_amplitude.m
PERTURB_DIR = os.environ.get("PERTURB_DIR", "../data/perturb")
AMPS = [100, 70, 80, 90, 110, 120, 130]      # 100 = unperturbed baseline
CONTRA_LO = 0.5                               # start fraction of the late (contralateral) window


def parse_variant(name: str) -> Tuple[str, str]:
    m = re.search(r"loso_speed_loso__(.+?)__(.+)$", name)
    return (m.group(1), m.group(2)) if m else ("?", "?")


def resolve_aux_channels(dset: Dict[str, Any]) -> Dict[str, int]:
    """Indices of the CoM auxiliary channels, following build_dataset_from_mat's stack order."""
    n_wrist = (len(dset.get("axes_pos", [])) + len(dset.get("axes_vel", [])) +
               len(dset.get("axes_acc", [])) + len(dset.get("axes_imu_acc_local", [])) +
               len(dset.get("axes_imu_acc_global", [])) +
               len(dset.get("axes_imu_gyro_local", [])) +
               len(dset.get("axes_imu_gyro_global", []))) * len(dset.get("body_parts", []))
    out: Dict[str, int] = {}
    for i, f in enumerate(dset.get("modeling_fields_to_use", [])):
        if f.endswith("_x"):
            out["AP"] = n_wrist + i
        elif f.endswith("_z"):
            out["VT"] = n_wrist + i
    return out


def corr_stats(x0: np.ndarray, x1: np.ndarray) -> Tuple[float, float, float]:
    """(overall r, early r, late/contralateral r). Computed per stride, then averaged."""
    T = x0.shape[1]
    i0 = int(T * CONTRA_LO)

    def _r(a: np.ndarray, b: np.ndarray) -> float:
        vals = []
        for i in range(a.shape[0]):
            u, v = a[i], b[i]
            if np.std(u) < 1e-12 or np.std(v) < 1e-12:
                continue
            vals.append(float(np.corrcoef(u, v)[0, 1]))
        return float(np.mean(vals)) if vals else np.nan

    return _r(x0, x1), _r(x0[:, :i0], x1[:, :i0]), _r(x0[:, i0:], x1[:, i0:])


def nrmse_and_r(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    nr, rr = [], []
    for i in range(y_true.shape[0]):
        t, p = y_true[i], y_pred[i]
        rng = t.max() - t.min()
        if rng > 0:
            nr.append(float(np.sqrt(np.mean((p - t) ** 2)) / rng))
        if np.std(t) > 1e-12 and np.std(p) > 1e-12:
            rr.append(float(np.corrcoef(t, p)[0, 1]))
    return (float(np.mean(nr)) if nr else np.nan,
            float(np.mean(rr)) if rr else np.nan)


def predict(model, X: np.ndarray, device, batch: int = 128) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, 0)


def find_ckpt(fold_dir: str) -> Optional[str]:
    """Only the final-epoch weights are used (paper protocol). No best-checkpoint fallback."""
    last = os.path.join(fold_dir, "last.pth")
    return last if os.path.isfile(last) else None


def build_for_amp(cfg, cfg_dir, amp, cache):
    """Build (grf, kin, modeling) with the modeling .mat of the given amplitude level."""
    dd = dict(cfg["data"])
    pert_path = os.path.join(PERTURB_DIR, f"stride_modeling_weamp_{amp:03d}.mat")
    if not os.path.isfile(pert_path):
        raise FileNotFoundError(pert_path)

    grf_p = os.path.abspath(os.path.join(cfg_dir, dd["grf_mat"]))
    kin_p = os.path.abspath(os.path.join(cfg_dir, dd["kin_mat"]))
    key = (grf_p, kin_p, pert_path)
    if key not in cache:
        print(f"    [load] {os.path.basename(pert_path)}", flush=True)
        cache[key] = load_all_mat_data(
            grf_p, dd.get("grf_struct", "stride_grf"),
            kin_p, dd.get("kin_struct", "stride_kinematics_arm"),
            pert_path, dd.get("modeling_struct", "stride_modeling"))
    return cache[key]


def run_one(run_dir, cache, device) -> List[Dict[str, Any]]:
    name = os.path.basename(run_dir)
    dvar, svar = parse_variant(name)
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return []
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    dset = cfg.get("dataset", {}) or {}
    aux = resolve_aux_channels(dset)
    if "AP" not in aux or "VT" not in aux:
        return []

    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    tv, te = dset.get("train_val_sessions", []), dset.get("test_sessions", [])
    axes = dset.get("grf_axes", ["Fx", "Fz"])
    svar_clean = re.sub(r"__aux\w+$", "", svar)
    regime = "OOD" if svar_clean in OOD else "ID"
    rows: List[Dict[str, Any]] = []

    # rebuild the dataset per amplitude level (the input CoM changes)
    built: Dict[int, Any] = {}
    for amp in AMPS:
        grf, kin, mdl = build_for_amp(cfg, cfg_dir, amp, cache)
        built[amp] = build_dataset_from_mat(
            grf=grf, kin=kin, modeling=mdl if dset.get("use_modeling_input") else {},
            sessions=sorted(set(tv) | set(te)),
            grf_sides=dset.get("grf_sides", ["Total"]), grf_axes=axes,
            body_parts=dset.get("body_parts", []), axes_pos=dset.get("axes_pos", []),
            axes_vel=dset.get("axes_vel", []), axes_acc=dset.get("axes_acc", []),
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
            modeling_fields_to_use=dset.get("modeling_fields_to_use", []))

    X0, y0, info0 = built[100]                       # baseline (unperturbed)
    idx_tv = _select_indices(info0, sessions=tv)
    info_tv = [info0[i] for i in idx_tv]
    subs = sorted({s["subject"] for s in info_tv})
    F = X0.shape[-1]

    for val_sub in subs:
        fold_dir = os.path.join(run_dir, "ckpt", f"fold_{val_sub}")
        ck = find_ckpt(fold_dir) if os.path.isdir(fold_dir) else None
        if ck is None:
            continue
        tr_mask = np.array([s.get("subject") != val_sub for s in info_tv])
        if tr_mask.sum() == 0:
            continue
        # the scaler is fitted on the UNPERTURBED training data, as in training
        scaler = StandardScaler().fit(X0[idx_tv][tr_mask].reshape(-1, F))

        idx_te = _select_indices(info0, subjects=[val_sub], sessions=te)
        if not idx_te:
            continue
        te_sess = np.array([info0[i]["session"] for i in idx_te])

        model = build_model_from_cfg(cfg.get("model", {}), F, y0.shape[-1]).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        for amp in AMPS:
            Xa, ya, _ = built[amp]
            Xte, yte = Xa[idx_te], ya[idx_te]
            # how much the input CoM changed relative to the baseline
            r_all, r_pre, r_con = corr_stats(X0[idx_te][:, :, aux["AP"]],
                                             Xte[:, :, aux["AP"]])
            rz_all, _, _ = corr_stats(X0[idx_te][:, :, aux["VT"]],
                                      Xte[:, :, aux["VT"]])
            Xs = scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape)
            pred = predict(model, Xs, device)
            for s in np.unique(te_sess):
                m = te_sess == s
                for ci, cn in enumerate(axes):
                    nr, rr = nrmse_and_r(yte[m, :, ci], pred[m, :, ci])
                    rows.append(dict(
                        data=dvar, session=svar, test_session=s, subject=val_sub,
                        regime=regime, tv=TRAIN_SESS.get(svar_clean, "?"),
                        ch=cn, amp=amp / 100.0,
                        in_r_AP=r_all, in_r_AP_pre=r_pre, in_r_AP_contra=r_con,
                        in_r_VT=rz_all, nrmse=nr, r=rr))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Proposed-model run folder. The canonical name is "proposed"; older runs were
    # saved under the internal name "no_physics", so both are searched.
    _base = os.path.join(args.runs_root, "binn")
    search = next((os.path.join(_base, m) for m in ("proposed", "no_physics")
                   if os.path.isdir(os.path.join(_base, m))),
                  os.path.join(_base, "proposed"))
    runs = [os.path.join(search, d) for d in sorted(os.listdir(search))
            if re.match(r"^\d{8}-", d) and d.endswith("__auxAPVT")
            and "full_day1" in d]
    print(f"[scan] {len(runs)} full_day1 auxAPVT runs | {len(AMPS)} amplitude levels | {args.device}")

    cache: Dict[Any, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for i, rd in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {os.path.basename(rd)}", flush=True)
        try:
            all_rows += run_one(rd, cache, torch.device(args.device))
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr, flush=True)

    if not all_rows:
        print("[warn] no results")
        sys.exit(0)
    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
    print(f"\n[OK] {len(all_rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
