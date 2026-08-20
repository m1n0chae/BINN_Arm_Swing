#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[revision] CoM 보조입력 노이즈 민감도 — r 목표 방식

리뷰어 요구
    "AP-CoM 에 노이즈를 주입했을 때 GRF 정확도가 견고한지 민감도 분석"

왜 'SD 대비 %' 가 아니라 'r 목표' 인가
    임의 크기의 노이즈는 "그 정도면 심한 건가?" 에 답할 수 없다.
    논문 Table III 에 운동학 모델의 CoM 추정 정확도가 이미 r 로 보고되어 있으므로
    (AP 는 r = 0.00~0.36), **그 수준을 재현하도록** 신호를 손상시키면
    "실제 관측된 최악의 추정 오차 수준에서도 GRF 가 견고한가" 라는 질문에 직접 답한다.

손상 방법 (stride 단위)
    원 신호 x 를 stride 별로 표준화한 z 와, 동일 분산의 노이즈 n 을 섞는다.
        z' = r * z + sqrt(1 - r^2) * n
    z 와 n 이 독립이고 둘 다 단위분산이면 corr(z', z) = r 이 정확히 성립한다.
    이후 원래의 평균/표준편차로 되돌려 물리 단위를 유지한 뒤 scaler 를 적용한다.
    scaler 는 clean train 으로 fit 된 것을 그대로 쓴다 (배포 상황과 동일).

노이즈 종류 2 가지
    white   : 시간축 백색 가우시안. CNN 이 평활화로 쉽게 지울 수 있어 모델에 유리하다.
    lowfreq : 백색 노이즈를 가우시안 평활(sigma=10 프레임)한 저주파 성분.
              실제 CoM 추정 오차(드리프트·편향)에 더 가깝고 지우기 어렵다.
    둘 다 보고해야 "지우기 쉬운 노이즈만 시험했다" 는 반박을 막을 수 있다.

평가
    추론 시에만 주입 (추가 학습 없음). 최종 epoch 체크포인트(last.pth) 재사용.
    fold 의 test 셋에서 평가하므로 ID / OOD 구분이 그대로 유지된다.
    지표는 GRF 채널별 NRMSE(range) 와 Pearson r.

사용법
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
LOWFREQ_SIGMA = 10.0            # 프레임 단위 가우시안 평활 폭
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
    """dataset 설정에서 CoM 보조채널 인덱스를 계산. acc_x = AP, acc_z = VT."""
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
    """프로세스 간 재현되는 결정적 시드.

    파이썬 str 의 hash() 는 PYTHONHASHSEED 에 따라 프로세스마다 달라져 재현이 안 된다.
    blake2s 로 고정한다.

    ※ 시드에 r 을 넣지 않는다. 넣으면 손상 수준마다 다른 난수 실현이 생겨,
      인접 수준 비교가 "손상을 한 단계 진행한 효과" 와 "잡음 실현이 바뀐 효과" 를
      섞어버린다. r 을 빼면 같은 stride/채널에서 직교화된 잡음 파형이 모든 수준에
      동일하게 재사용되고 혼합계수만 바뀌므로, 증분이 순수한 dose increment 가 된다.
    """
    key = f"{int(sd_)}|{target}|{kind}".encode()
    return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")


def _gauss_kernel(sigma: float) -> np.ndarray:
    rad = int(max(1, round(3 * sigma)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def make_noise(shape: Tuple[int, int], kind: str, rng: np.random.Generator) -> np.ndarray:
    """(N, T) 평균 0 · 단위분산 노이즈. lowfreq 는 평활 후 다시 맞춘다.

    ※ 반드시 중심화해야 한다. 중심화하지 않으면 stride 당 mean(n) 이 0 이 아니어서
      corrupt_to_r 의 복원 mu + sd*z2 가 원 채널의 평균을 보존하지 못하고,
      sqrt(1-r^2)*mean(n)*sd 만큼의 DC 오프셋이 따라붙는다. 평활 노이즈는 유효 자유도가
      낮아 |mean(n)| 이 평균 0.68 에 달해, 손상이 심할수록(=r 이 낮을수록) 오프셋이 커진다.
      그러면 용량-반응이 탈상관 효과와 DC 편향 효과를 뒤섞게 된다.
      상관계수는 평균을 빼고 계산하므로 달성 r 검증만으로는 이 결함이 드러나지 않는다.
    """
    n = rng.standard_normal(shape)
    if kind == "lowfreq":
        k = _gauss_kernel(LOWFREQ_SIGMA)
        n = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, n)
    n = n - n.mean(axis=1, keepdims=True)
    sd = n.std(axis=1, keepdims=True)
    return n / np.maximum(sd, 1e-8)


def corrupt_to_r(x: np.ndarray, r: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    """(N, T) 채널을 원 신호와의 상관이 정확히 r 이 되도록 손상. 평균/SD 는 보존.

    z' = r*z + sqrt(1-r^2)*n 이 corr(z',z)=r 을 주려면 유한표본에서도 <z,n>=0 이어야 한다.
    저주파 노이즈는 평활로 유효 자유도가 T=101 -> ~10 으로 떨어져 stride 하나하나에서는
    |corr(z,n)| 이 평균 0.42 에 달한다(백색이면 1/sqrt(101)=0.10). 그대로 두면 목표 r 이
    stride 마다 크게 흔들린다(목표 0.36 에서 달성 r 의 SD 가 0.43).
    따라서 n 에서 z 성분을 제거(Gram-Schmidt)한 뒤 재정규화한다.
    -> 모든 stride 에서 corr(z', z) = r 이 정확히 성립 (SD 0.000).
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    z = (x - mu) / np.maximum(sd, 1e-8)
    n = make_noise(x.shape, kind, rng)
    # stride 단위 직교화
    zz = np.maximum((z * z).sum(axis=1, keepdims=True), 1e-8)
    n = n - ((n * z).sum(axis=1, keepdims=True) / zz) * z
    n = n / np.maximum(n.std(axis=1, keepdims=True), 1e-8)
    z2 = r * z + np.sqrt(max(0.0, 1.0 - r * r)) * n
    return mu + sd * z2


def nrmse_and_r(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """(N, T) 한 채널: stride 별 NRMSE(range) 평균, 그리고 전체 이어붙인 Pearson r."""
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
    """최종 epoch 가중치만 사용한다. LOSO 에서는 val 폴드가 곧 test 피험자이므로
    검증 기준으로 고른 체크포인트를 쓰면 test set selection 이 된다."""
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
        # 학습 fold 로만 scaler 재현 (main.py 와 동일)
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

    # 제안 모델 run 폴더. 표준 이름은 "proposed" 이고, 구버전 실행 결과는
    # 내부 이름 "no_physics" 폴더에 저장되어 있으므로 둘 다 찾는다.
    _base = os.path.join(args.runs_root, "physres_cnn_attn_ablation")
    search = next((os.path.join(_base, m) for m in ("proposed", "no_physics")
                   if os.path.isdir(os.path.join(_base, m))),
                  os.path.join(_base, "proposed"))
    if not os.path.isdir(search):
        print(f"[error] 경로 없음: {search}", file=sys.stderr)
        sys.exit(1)
    runs = [os.path.join(search, d) for d in sorted(os.listdir(search))
            if re.match(r"^\d{8}-", d) and d.endswith("__auxAPVT")]
    print(f"[scan] auxAPVT run {len(runs)}개 | seeds={args.seeds} | device={args.device}")

    cache, all_rows = MatCache(), []
    for i, rd in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {os.path.basename(rd)}", flush=True)
        try:
            all_rows += run_one(rd, cache, torch.device(args.device), args.seeds)
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)

    if not all_rows:
        print("[warn] 결과 없음")
        sys.exit(0)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[OK] {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
