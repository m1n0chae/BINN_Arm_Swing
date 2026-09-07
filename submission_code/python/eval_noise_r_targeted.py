#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoM auxiliary-input corruption sensitivity — correlation-targeted method.

Purpose
    Test whether GRF accuracy stays robust when the CoM auxiliary input is
    corrupted, in particular the AP component.

Why an r target rather than 'percent of SD'
    Noise of arbitrary magnitude cannot answer "is that amount severe?".
    Table III of the paper already reports the kinematics model's CoM estimation
    accuracy as r (AP: r = 0.00-0.36), so corrupting the signal to REPRODUCE that
    level directly answers "is GRF robust even at the worst estimation error
    actually observed?".

Corruption method (per stride)
    The original signal x is standardised per stride to z and mixed with
    equal-variance noise n:
        z' = r * z + sqrt(1 - r^2) * n
    If z and n are independent and both unit-variance, corr(z', z) = r exactly.
    The result is mapped back to the original mean/SD to preserve physical units,
    then the scaler is applied.
    The scaler is the one fitted on the clean training folds (deployment setting).

Two noise kinds
    white   : white Gaussian along time. A CNN can remove it easily by smoothing,
              which favours the model.
    lowfreq : white noise smoothed with a Gaussian (sigma = 10 frames), keeping
              the low-frequency component. Closer to real CoM estimation error
              (drift/bias) and hard to remove.
    Reporting both forestalls the objection "only easily removable noise was tested".

Evaluation
    Injected at inference only (no retraining). The final-epoch checkpoint
    (last.pth) is reused. Evaluation on the fold's test set keeps the ID / OOD
    distinction intact. Metrics: per-channel GRF NRMSE(range) and Pearson r.

Usage
    python eval_noise_r_targeted.py --runs_root <trained_runs> \
        --out_csv noise_results.csv
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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

R_LEVELS = [1.00, 0.90, 0.70, 0.50, 0.36, 0.00]   # 1.00 = clean
TARGETS = ["AP", "VT", "BOTH"]
NOISE_KINDS = ["white", "lowfreq"]
LOWFREQ_SIGMA = 10.0            # Gaussian smoothing width in frames
TRAIN_SESS = {
    "tv_te_ss1": "ss1", "tv_te_ss2": "ss2", "tv_te_ss3": "ss3",
    "tv_ss1__te_ss2ss3": "ss1", "tv_ss2__te_ss1ss3": "ss2", "tv_ss3__te_ss1ss2": "ss3",
    "tv_te_ss123": "ss123",
}


def parse_variant(name: str) -> Tuple[str, str]:
    m = re.search(r"loso_speed_loso(__.+?)?(__.+?)?$", name)
    if m and m.group(1) and m.group(2):
        return m.group(1).strip("_"), m.group(2).strip("_")
    return "unknown", "unknown"


def resolve_aux_channels(dset: Dict[str, Any]) -> Dict[str, int]:
    """Compute the CoM auxiliary-channel indices from the dataset settings. acc_x = AP, acc_z = VT."""
    n_ax = sum(len(dset.get(k, []) or []) for k in (
        "axes_pos", "axes_vel", "axes_acc", "axes_imu_acc_local",
        "axes_imu_acc_global", "axes_imu_gyro_local", "axes_imu_gyro_global"))
    ch = len(dset.get("body_parts", []) or []) * n_ax
    if not dset.get("use_modeling_input", False):
        return {}
    f2a = {"acc_x": "AP", "acc_z": "VT"}
    out: Dict[str, int] = {}
    for _pt in dset.get("modeling_points_to_use", []) or []:
        for f in dset.get("modeling_fields_to_use", []) or []:
            a = f2a.get(f)
            if a and a not in out:
                out[a] = ch
            ch += 1
    return out


def _seed_of(sd_: int, target: str, kind: str) -> int:
    """Deterministic seed reproducible across processes.

    Python's str hash() varies with PYTHONHASHSEED between processes, so blake2s
    is used to pin it.

    NOTE: r is intentionally NOT part of the seed. Including it would draw a
      different noise realisation per corruption level, so adjacent-level
      comparisons would mix "the effect of one more corruption step" with "the
      effect of a changed noise realisation". Without r, the same orthogonalised
      noise waveform is reused at every level for a given stride/channel and only
      the mixing coefficient changes — the increment is a pure dose increment.
    """
    key = f"{int(sd_)}|{target}|{kind}".encode()
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")


def _gauss_kernel(sigma: float) -> np.ndarray:
    rad = int(max(1, round(3 * sigma)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def make_noise(shape: Tuple[int, int], kind: str, rng: np.random.Generator) -> np.ndarray:
    """(N, T) zero-mean, unit-variance noise. lowfreq is re-normalised after smoothing.

    NOTE: centring is mandatory. Without it, per-stride mean(n) is nonzero, so the
      restoration mu + sd*z2 in corrupt_to_r does not preserve the channel mean and
      a DC offset of sqrt(1-r^2)*mean(n)*sd creeps in. Smoothed noise has few
      effective degrees of freedom, so |mean(n)| averages 0.68, and the offset grows
      as the corruption gets more severe (lower r). The dose-response would then mix
      the decorrelation effect with a DC-bias effect. Since correlation subtracts
      the mean, verifying the achieved r alone would never expose this flaw.
    """
    n = rng.standard_normal(shape)
    if kind == "lowfreq":
        k = _gauss_kernel(LOWFREQ_SIGMA)
        n = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, n)
    n = n - n.mean(axis=1, keepdims=True)
    sd = n.std(axis=1, keepdims=True)
    return n / np.maximum(sd, 1e-8)


def corrupt_to_r(x: np.ndarray, r: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Corrupt an (N, T) channel so that its correlation with the original is exactly r.
    The per-stride mean/SD are preserved.

    For z' = r*z + sqrt(1-r^2)*n to give corr(z',z)=r, <z,n>=0 must hold even in
    finite samples. Smoothing drops the effective degrees of freedom of low-frequency
    noise from T=101 to ~10, so within single strides |corr(z,n)| averages 0.42
    (white noise: 1/sqrt(101)=0.10). Left as-is, the achieved r scatters widely
    across strides (SD 0.43 at a target of 0.36).
    Therefore the z component is removed from n (Gram-Schmidt) and n re-normalised
    -> corr(z', z) = r holds exactly for every stride (SD 0.000).
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    z = (x - mu) / np.maximum(sd, 1e-8)
    n = make_noise(x.shape, kind, rng)
    # per-stride orthogonalisation
    zz = np.maximum((z * z).sum(axis=1, keepdims=True), 1e-8)
    n = n - ((n * z).sum(axis=1, keepdims=True) / zz) * z
    n = n / np.maximum(n.std(axis=1, keepdims=True), 1e-8)
    z2 = r * z + np.sqrt(max(0.0, 1.0 - r * r)) * n
    return mu + sd * z2


def nrmse_and_r(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """One (N, T) channel: mean of per-stride NRMSE(range), plus Pearson r over the concatenation."""
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean(axis=1))
    rng_ = np.maximum(y_true.max(axis=1) - y_true.min(axis=1), 1e-8)
    nr = float((rmse / rng_).mean())
    a, b = y_true.ravel(), y_pred.ravel()
    rr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return nr, rr


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 128) -> np.ndarray:
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def find_ckpt(fold_dir: str) -> Optional[str]:
    """Only the final-epoch weights are used. Under LOSO the validation fold IS the
    test subject, so a checkpoint chosen by validation loss would be test-set selection."""
    last = os.path.join(fold_dir, "last.pth")
    return last if os.path.isfile(last) else None


class MatCache:
    def __init__(self) -> None:
        self._c: Dict[Tuple[str, str, str], Any] = {}

    def get(self, cfg, cfg_dir):
        dd = cfg["data"]
        key = tuple(os.path.abspath(os.path.join(cfg_dir, dd[k]))
                    for k in ("grf_mat", "kin_mat", "modeling_mat"))
        if key not in self._c:
            print(f"  [load] {os.path.basename(key[1])}", flush=True)
            self._c[key] = load_all_mat_data(
                key[0], dd.get("grf_struct", "stride_grf"),
                key[1], dd.get("kin_struct", "stride_kinematics_arm"),
                key[2], dd.get("modeling_struct", "stride_modeling"))
        return self._c[key]


def run_one(run_dir: str, cache: MatCache, device, seeds: List[int]) -> List[Dict[str, Any]]:
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
    grf, kin, mdl = cache.get(cfg, cfg_dir)
    tv = dset.get("train_val_sessions", [])
    te = dset.get("test_sessions", [])
    X, y, info = build_dataset_from_mat(
        grf=grf, kin=kin, modeling=mdl if dset.get("use_modeling_input") else {},
        sessions=sorted(set(tv) | set(te)),
        grf_sides=dset.get("grf_sides", ["Total"]), grf_axes=dset.get("grf_axes", ["Fx", "Fz"]),
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

    idx_tv = _select_indices(info, sessions=tv)
    X_tv, y_tv = X[idx_tv], y[idx_tv]
    info_tv = [info[i] for i in idx_tv]
    axes = dset.get("grf_axes", ["Fx", "Fz"])
    train_sess = TRAIN_SESS.get(re.sub(r"__aux\w+$", "", svar), "?")
    rows: List[Dict[str, Any]] = []

    for val_sub in sorted({s["subject"] for s in info_tv}):
        fold_dir = os.path.join(run_dir, "ckpt", f"fold_{val_sub}")
        ck = find_ckpt(fold_dir) if os.path.isdir(fold_dir) else None
        if ck is None:
            continue
        # reproduce the scaler from the training folds only (same as main.py)
        tr_mask = np.array([s.get("subject") != val_sub for s in info_tv])
        if tr_mask.sum() == 0:
            continue
        F = X_tv.shape[-1]
        scaler = StandardScaler().fit(X_tv[tr_mask].reshape(-1, F))

        idx_te = _select_indices(info, subjects=[val_sub], sessions=te)
        if not idx_te:
            continue
        Xte_raw, yte = X[idx_te], y[idx_te]
        te_sess = np.array([info[i]["session"] for i in idx_te])

        model = build_model_from_cfg(cfg.get("model", {}), F, y.shape[-1]).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        for kind in NOISE_KINDS:
            for target in TARGETS:
                chans = [aux["AP"]] if target == "AP" else \
                        [aux["VT"]] if target == "VT" else [aux["AP"], aux["VT"]]
                for r in R_LEVELS:
                    use_seeds = [0] if r >= 1.0 else seeds
                    for sd_ in use_seeds:
                        Xp = Xte_raw.copy()
                        if r < 1.0:
                            rng = np.random.default_rng(_seed_of(sd_, target, kind))
                            for c in chans:
                                Xp[:, :, c] = corrupt_to_r(Xte_raw[:, :, c], r, kind, rng)
                        Xs = scaler.transform(Xp.reshape(-1, F)).reshape(Xp.shape)
                        pred = predict(model, Xs, device)
                        for s in np.unique(te_sess):
                            m = te_sess == s
                            for ci, cn in enumerate(axes):
                                nr, rr = nrmse_and_r(yte[m, :, ci], pred[m, :, ci])
                                rows.append(dict(
                                    data=dvar, session=svar, train_sess=train_sess,
                                    test_session=s, subject=val_sub, ch=cn,
                                    noise_kind=kind, target=target, r_level=r,
                                    noise_seed=sd_, nrmse=nr, r=rr))
                        if r >= 1.0:
                            break
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"  [done] {name}: {len(rows)} rows", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Proposed-model run folder. The canonical name is "proposed"; older runs were
    # saved under the internal name "no_physics", so both are searched.
    _base = os.path.join(args.runs_root, "binn")
    search = next((os.path.join(_base, m) for m in ("proposed", "no_physics")
                   if os.path.isdir(os.path.join(_base, m))),
                  os.path.join(_base, "proposed"))
    if not os.path.isdir(search):
        print(f"[error] path not found: {search}", file=sys.stderr)
        sys.exit(1)
    runs = [os.path.join(search, d) for d in sorted(os.listdir(search))
            if re.match(r"^\d{8}-", d) and d.endswith("__auxAPVT")]
    print(f"[scan] {len(runs)} auxAPVT runs | seeds={args.seeds} | device={args.device}")

    cache, all_rows = MatCache(), []
    for i, rd in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {os.path.basename(rd)}", flush=True)
        try:
            all_rows += run_one(rd, cache, torch.device(args.device), args.seeds)
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)

    if not all_rows:
        print("[warn] no results")
        sys.exit(0)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[OK] {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
