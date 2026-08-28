#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoM 오프셋 섭동 민감도 — test-time only  (Section III-B)

리뷰어 지적
    "데이터셋 A-B 간 체계적 마커 부착 편향(몸통 오프셋)이 최종 GRF 오차로
     전파되는 정도가 정량화되지 않았음. 간단한 민감도 추정이 도움이 될 것."

설계
    정상 오프셋으로 학습된 모델은 그대로 두고, **추론 입력만** 섭동된 CoM 으로
    바꾼다. "학습된 모델이 이 입력 오차에 얼마나 민감한가" 를 본다.
    학습은 전혀 다시 하지 않는다 (최종 epoch 체크포인트 재사용).

섭동 데이터
    matlab/2_upper_body_model/perturb_torso_offset.m 이 upper_body_model.m 의
    몸통 오프셋 off_sc 에 상수를 더해 재생성한 stride_modeling.
        scadd_m30 = -30도,  scadd_m15 = -15도,  scadd_p15 = +15도,  scadd_p30 = +30도
    Eq.(3) 에서 th_i_fd = th_i_pred + off_i 이므로 off 에 Δ 를 더하면 그 세그먼트의
    기여 벡터가 정확히 Δ 만큼 회전한다. 세 세그먼트가 서로 다른 각도로 회전하므로
    최종 a_CoM 을 통째로 회전시키는 근사로는 대체할 수 없어 원 파이프라인을 재실행했다.

    마커 부착 편향은 stride 마다 동일하게 작용하는 상수이므로, 실측 편차(약 15도)를
    문자 그대로 재현하려면 비례 스케일링이 아니라 가산이 맞다. baseline 을 포함해
    전부 동일 설정(INCLUDE_SC=true, gravity)으로 재생성했으므로 SC 오프셋만이
    유일한 차이다.

    ※ 이 baseline 은 학습에 쓴 modeling 파일과 생성 설정이 달라, 절대 성능이 아니라
      **baseline 대비 상대 민감도**만 해석해야 한다.

평가
    fold 의 test 셋에서 GRF 채널별 NRMSE(range) 와 Pearson r.
    train_val 세션을 봤는지로 ID / OOD 를 나눈다.

사용법
    PERTURB_DIR=../data/perturb python eval_offset_perturbation.py         --runs_root <trained_runs> --out_csv scadd.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from matlab_data_prep import load_all_mat_data, load_mat_struct
from utils_and_modules import build_dataset_from_mat
from main import build_model_from_cfg, _select_indices

# perturb_torso_offset.m 이 만든 섭동 modeling .mat 들이 있는 폴더.
# 기본값은 저장소 기준 상대 경로이고, 데이터를 다른 곳에 두었다면
# 환경변수 PERTURB_DIR 로 덮어쓴다.
PERTURB_DIR = os.environ.get("PERTURB_DIR", "../data/perturb")
# 몸통(SC) 오프셋 가산 섭동. 자세한 설명은 위 docstring 참고.
PERTURBS = [("baseline", "stride_modeling_baseSC.mat"),
            ("-30deg", "stride_modeling_scadd_m30.mat"), ("-15deg", "stride_modeling_scadd_m15.mat"),
            ("+15deg", "stride_modeling_scadd_p15.mat"), ("+30deg", "stride_modeling_scadd_p30.mat")]
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


def nrmse_and_r(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean(axis=1))
    rng = np.maximum(y_true.max(axis=1) - y_true.min(axis=1), 1e-8)
    a, b = y_true.ravel(), y_pred.ravel()
    rr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return float((rmse / rng).mean()), rr


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def find_ckpt(fold_dir: str) -> Optional[str]:
    """최종 epoch 가중치만 사용한다 (논문 프로토콜). best 체크포인트 fallback 없음."""
    last = os.path.join(fold_dir, "last.pth")
    return last if os.path.isfile(last) else None


class Cache:
    """grf/kin 은 data_variant 마다, 섭동 modeling 은 파일마다 한 번씩만 읽는다."""

    def __init__(self) -> None:
        self.gk: Dict[Tuple[str, str], Any] = {}
        self.mdl: Dict[str, Any] = {}

    def get_gk(self, cfg, cfg_dir):
        d = cfg["data"]
        key = (os.path.abspath(os.path.join(cfg_dir, d["grf_mat"])),
               os.path.abspath(os.path.join(cfg_dir, d["kin_mat"])))
        if key not in self.gk:
            print(f"  [load] {os.path.basename(key[1])}", flush=True)
            grf, kin, _ = load_all_mat_data(key[0], d.get("grf_struct", "stride_grf"),
                                            key[1], d.get("kin_struct", "stride_kinematics_arm"))
            self.gk[key] = (grf, kin)
        return self.gk[key]

    def get_mdl(self, path, struct, cfg_dir):
        p = os.path.abspath(os.path.join(cfg_dir, path))
        if p not in self.mdl:
            print(f"  [load] {os.path.basename(p)}", flush=True)
            self.mdl[p] = load_mat_struct(p, struct)
        return self.mdl[p]


def build(dset, grf, kin, mdl):
    tv, te = dset.get("train_val_sessions", []), dset.get("test_sessions", [])
    return build_dataset_from_mat(
        grf=grf, kin=kin, modeling=mdl, sessions=sorted(set(tv) | set(te)),
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


def run_one(run_dir, cache, device) -> List[Dict[str, Any]]:
    name = os.path.basename(run_dir)
    dvar, svar = parse_variant(name)
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return []
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    dset = cfg.get("dataset", {}) or {}
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    grf, kin = cache.get_gk(cfg, cfg_dir)
    struct = cfg["data"].get("modeling_struct", "stride_modeling")
    train_sess = TRAIN_SESS.get(re.sub(r"__aux\w+$", "", svar), "?")
    axes = dset.get("grf_axes", ["Fx", "Fz"])
    rows: List[Dict[str, Any]] = []

    # 섭동본마다 데이터셋을 새로 조립한다. clean scaler 는 baseline 으로 fit 해야 하므로
    # baseline X 를 먼저 만들어 두고 재사용한다.
    built: Dict[str, Any] = {}
    for tag, fname in PERTURBS:
        path = os.path.join(PERTURB_DIR, fname)
        try:
            mdl = cache.get_mdl(path, struct, cfg_dir)
            built[tag] = build(dset, grf, kin, mdl)
        except Exception as e:
            print(f"  [skip {tag}] {e}", file=sys.stderr)
    if "baseline" not in built:
        return []

    X0, y0, info0 = built["baseline"]
    idx_tv = _select_indices(info0, sessions=dset.get("train_val_sessions", []))
    X_tv, info_tv = X0[idx_tv], [info0[i] for i in idx_tv]
    F = X0.shape[-1]

    for val_sub in sorted({s["subject"] for s in info_tv}):
        fd = os.path.join(run_dir, "ckpt", f"fold_{val_sub}")
        ck = find_ckpt(fd) if os.path.isdir(fd) else None
        if ck is None:
            continue
        tr = np.array([s.get("subject") != val_sub for s in info_tv])
        if tr.sum() == 0:
            continue
        scaler = StandardScaler().fit(X_tv[tr].reshape(-1, F))   # clean train 으로만 fit
        model = build_model_from_cfg(cfg.get("model", {}), F, y0.shape[-1]).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        for tag, _ in PERTURBS:
            if tag not in built:
                continue
            Xp, yp, infop = built[tag]
            if Xp.shape[-1] != F:
                continue
            idx_te = _select_indices(infop, subjects=[val_sub],
                                     sessions=dset.get("test_sessions", []))
            if not idx_te:
                continue
            Xt, yt = Xp[idx_te], yp[idx_te]
            sess = np.array([infop[i]["session"] for i in idx_te])
            pred = predict(model, scaler.transform(Xt.reshape(-1, F)).reshape(Xt.shape), device)
            for s in np.unique(sess):
                m = sess == s
                for ci, cn in enumerate(axes):
                    nr, rr = nrmse_and_r(yt[m, :, ci], pred[m, :, ci])
                    rows.append(dict(data=dvar, session=svar, train_sess=train_sess,
                                     test_session=s, subject=val_sub, ch=cn,
                                     pert=tag, nrmse=nr, r=rr))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"  [done] {name}: {len(rows)} rows", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 제안 모델 run 폴더. 표준 이름은 "proposed" 이고, 구버전 실행 결과는
    # 내부 이름 "no_physics" 폴더에 저장되어 있으므로 둘 다 찾는다.
    _base = os.path.join(args.runs_root, "binn")
    search = next((os.path.join(_base, m) for m in ("proposed", "no_physics")
                   if os.path.isdir(os.path.join(_base, m))),
                  os.path.join(_base, "proposed"))
    runs = [os.path.join(search, d) for d in sorted(os.listdir(search))
            if re.match(r"^\d{8}-", d) and d.endswith("__auxAPVT")]
    print(f"[scan] auxAPVT run {len(runs)}개 | 섭동 {len(PERTURBS)}종 | device={args.device}")

    cache, rows = Cache(), []
    for i, rd in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {os.path.basename(rd)}", flush=True)
        try:
            rows += run_one(rd, cache, torch.device(args.device))
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
    if not rows:
        print("[warn] 결과 없음")
        sys.exit(0)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\n[OK] {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
