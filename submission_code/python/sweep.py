#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sweep.py
- config.yaml의 sweep.data_variants × sweep.dataset_variants 조합을 모두 실행
- 각 조합마다 run.name을 자동 변경 → 결과 폴더가 조합별로 분리 저장
"""

from __future__ import annotations
import argparse
import copy
import itertools
import os
import re
import sys
import subprocess
from typing import Any, Dict, List

import yaml

# 그냥 실행
# python sweep.py


# 실행 전에 “몇 개가 돌고 어떤 이름으로 저장되는지”만 확인(dry-run)
# python sweep.py --config config.yaml --dry_run


def sanitize_name(s: str, max_len: int = 160) -> str:
    """폴더/파일명 안전하게(공백/특수문자 제거, 길이 제한)"""
    s = str(s)
    s = re.sub(r"[^\w\.\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] if len(s) > max_len else s


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="base config.yaml 경로")
    ap.add_argument("--dry_run", action="store_true", help="실제 실행 없이 조합/명령만 출력")
    ap.add_argument("--keep_tmp", action="store_true", help="임시 config 파일들을 지우지 않음")
    ap.add_argument("--data_variants", nargs="+", default=None,
                    help="data_variant 이름 필터 (예: percent005 percent010)")
    ap.add_argument("--dataset_variants", nargs="+", default=None,
                    help="dataset_variant 이름 필터 (예: tv_te_ss1 tv_te_ss2)")
    ap.add_argument("--ablation_mode", default=None,
                    help="ablation_mode 오버라이드 (예: proposed, attn_only, no_attn)")
    ap.add_argument("--model_type", default=None,
                    help="model.type 오버라이드 (예: cnn, transformer, binn)")
    ap.add_argument("--use_modeling_input", default=None,
                    help="dataset.use_modeling_input 강제 설정 (true/false)")
    ap.add_argument("--name_suffix", default=None,
                    help="run.name 접미사 (예: cnn_with_model 구분용)")
    ap.add_argument("--output_dir", default=None,
                    help="run.output_dir 오버라이드 (예: ./day1_output)")
    args = ap.parse_args()

    base_cfg_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(base_cfg_path)

    base_cfg = load_yaml(base_cfg_path)

    run_cfg = base_cfg.get("run", {}) or {}
    base_run_name = str(run_cfg.get("name", "exp"))

    sweep = base_cfg.get("sweep", {}) or {}
    data_vars = sweep.get("data_variants", []) or []
    dset_vars = sweep.get("dataset_variants", []) or []

    if not data_vars or not dset_vars:
        raise ValueError(
            "config.yaml에 sweep.data_variants 와 sweep.dataset_variants 가 있어야 합니다."
        )

    # 필터 적용
    if args.data_variants:
        keep = set(args.data_variants)
        data_vars = [d for d in data_vars if d.get("name") in keep]
        print(f"[filter] data_variants → {[d.get('name') for d in data_vars]}")
    if args.dataset_variants:
        keep = set(args.dataset_variants)
        dset_vars = [d for d in dset_vars if d.get("name") in keep]
        print(f"[filter] dataset_variants → {[d.get('name') for d in dset_vars]}")
    if not data_vars or not dset_vars:
        print("[warn] 필터 후 조합 없음 — 종료")
        return

    # ablation_mode 오버라이드 (model.binn.ablation_mode)
    if args.ablation_mode:
        base_cfg.setdefault("model", {}).setdefault("binn", {})
        base_cfg["model"]["binn"]["ablation_mode"] = args.ablation_mode
        # 모델 type도 ablation 모듈로 강제 (proposed 는 ablation 모듈 사용)
        base_cfg["model"]["type"] = "binn"
        print(f"[override] ablation_mode = {args.ablation_mode}, model.type = binn")

    # model_type 오버라이드 (cnn, transformer 등)
    if args.model_type:
        base_cfg.setdefault("model", {})["type"] = args.model_type
        print(f"[override] model.type = {args.model_type}")

    # use_modeling_input 오버라이드
    if args.use_modeling_input is not None:
        v = args.use_modeling_input.lower() in ("1", "true", "yes", "y")
        base_cfg.setdefault("dataset", {})["use_modeling_input"] = v
        print(f"[override] dataset.use_modeling_input = {v}")

    # output_dir 오버라이드
    if args.output_dir is not None:
        base_cfg.setdefault("run", {})["output_dir"] = args.output_dir
        print(f"[override] run.output_dir = {args.output_dir}")

    # 임시 config 저장 폴더
    tmp_dir = os.path.join(base_dir, "_sweep_tmp_configs")
    os.makedirs(tmp_dir, exist_ok=True)

    combos = list(itertools.product(data_vars, dset_vars))
    print("\n==============================")
    print(" Sweep (data_variants × dataset_variants)")
    print(f" - base config : {base_cfg_path}")
    print(f" - base name   : {base_run_name}")
    print(f" - n_runs      : {len(combos)} (= {len(data_vars)} x {len(dset_vars)})")
    print("==============================\n")

    generated: List[str] = []

    for i, (dv, sv) in enumerate(combos, 1):
        dname = sanitize_name(dv.get("name", f"data{i:02d}"))
        sname = sanitize_name(sv.get("name", f"sess{i:02d}"))

        d_data = (dv.get("data", {}) or {})
        s_data = (sv.get("dataset", {}) or {})

        cfg_i = copy.deepcopy(base_cfg)

        # ✅ 부분 덮어쓰기(merge): dataset의 나머지 설정(grf_axes, imu 등)은 유지
        cfg_i.setdefault("data", {})
        cfg_i["data"].update(d_data)

        cfg_i.setdefault("dataset", {})
        cfg_i["dataset"].update(s_data)

        # ✅ 결과 폴더 분리 핵심: run.name을 조합별로 다르게
        cfg_i.setdefault("run", {})
        suf = f"__{args.name_suffix}" if args.name_suffix else ""
        cfg_i["run"]["name"] = f"{base_run_name}{suf}__{dname}__{sname}"

        # 임시 config 저장
        tmp_cfg_path = os.path.join(tmp_dir, f"{sanitize_name(cfg_i['run']['name'])}.yaml")
        save_yaml(cfg_i, tmp_cfg_path)
        generated.append(tmp_cfg_path)

        # main.py 실행
        cmd = [sys.executable, "main.py", "--config", tmp_cfg_path]

        print(f"[{i}/{len(combos)}] RUN = {cfg_i['run']['name']}")
        print(f"  - data_variant    : {dname}")
        print(f"  - dataset_variant : {sname}")
        print("  - cmd:", " ".join(cmd))

        if args.dry_run:
            print("  (dry_run: skip)\n")
            continue

        try:
            subprocess.run(cmd, check=True, cwd=base_dir)
            print("  ✅ done\n")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ failed (returncode={e.returncode})\n")
            # 실패해도 다음 조합 계속
            continue

    if (not args.keep_tmp) and (not args.dry_run):
        for p in generated:
            try:
                os.remove(p)
            except OSError:
                pass

    print("\n[OK] sweep finished.")
    print(f" - tmp configs: {tmp_dir}")


if __name__ == "__main__":
    main()
