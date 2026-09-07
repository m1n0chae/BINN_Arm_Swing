#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sweep.py
- Runs every combination of sweep.data_variants × sweep.dataset_variants in the config.
- run.name is changed automatically per combination, so the output folders are
  kept separate for each run.
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

# plain run:
# python sweep.py


# dry-run — only show how many runs there are and under which names they are saved:
# python sweep.py --config config.yaml --dry_run


def sanitize_name(s: str, max_len: int = 160) -> str:
    """Sanitise folder/file names (strip whitespace/special characters, cap length)."""
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
    ap.add_argument("--config", default="config.yaml", help="path to the base config.yaml")
    ap.add_argument("--dry_run", action="store_true", help="print the combinations/commands without running")
    ap.add_argument("--keep_tmp", action="store_true", help="do not delete the temporary config files")
    ap.add_argument("--data_variants", nargs="+", default=None,
                    help="filter by data_variant name (e.g. percent005 percent010)")
    ap.add_argument("--dataset_variants", nargs="+", default=None,
                    help="filter by dataset_variant name (e.g. tv_te_ss1 tv_te_ss2)")
    ap.add_argument("--ablation_mode", default=None,
                    help="override ablation_mode (e.g. proposed, attn_only, no_attn)")
    ap.add_argument("--model_type", default=None,
                    help="override model.type (e.g. cnn, transformer, binn)")
    ap.add_argument("--use_modeling_input", default=None,
                    help="force dataset.use_modeling_input (true/false)")
    ap.add_argument("--name_suffix", default=None,
                    help="suffix for run.name (e.g. to distinguish cnn_with_model)")
    ap.add_argument("--output_dir", default=None,
                    help="override run.output_dir (e.g. ./day1_output)")
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
            "config.yaml must define sweep.data_variants and sweep.dataset_variants."
        )

    # apply the filters
    if args.data_variants:
        keep = set(args.data_variants)
        data_vars = [d for d in data_vars if d.get("name") in keep]
        print(f"[filter] data_variants → {[d.get('name') for d in data_vars]}")
    if args.dataset_variants:
        keep = set(args.dataset_variants)
        dset_vars = [d for d in dset_vars if d.get("name") in keep]
        print(f"[filter] dataset_variants → {[d.get('name') for d in dset_vars]}")
    if not data_vars or not dset_vars:
        print("[warn] no combinations left after filtering — exiting")
        return

    # ablation_mode override (model.binn.ablation_mode)
    if args.ablation_mode:
        base_cfg.setdefault("model", {}).setdefault("binn", {})
        base_cfg["model"]["binn"]["ablation_mode"] = args.ablation_mode
        # also force the model type to binn (the proposed model uses this module)
        base_cfg["model"]["type"] = "binn"
        print(f"[override] ablation_mode = {args.ablation_mode}, model.type = binn")

    # model_type override (cnn, transformer, ...)
    if args.model_type:
        base_cfg.setdefault("model", {})["type"] = args.model_type
        print(f"[override] model.type = {args.model_type}")

    # use_modeling_input override
    if args.use_modeling_input is not None:
        v = args.use_modeling_input.lower() in ("1", "true", "yes", "y")
        base_cfg.setdefault("dataset", {})["use_modeling_input"] = v
        print(f"[override] dataset.use_modeling_input = {v}")

    # output_dir override
    if args.output_dir is not None:
        base_cfg.setdefault("run", {})["output_dir"] = args.output_dir
        print(f"[override] run.output_dir = {args.output_dir}")

    # folder for the temporary configs
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

        # partial overwrite (merge): the rest of the dataset settings (grf_axes, imu, ...) are kept
        cfg_i.setdefault("data", {})
        cfg_i["data"].update(d_data)

        cfg_i.setdefault("dataset", {})
        cfg_i["dataset"].update(s_data)

        # key to separating the output folders: a distinct run.name per combination
        cfg_i.setdefault("run", {})
        suf = f"__{args.name_suffix}" if args.name_suffix else ""
        cfg_i["run"]["name"] = f"{base_run_name}{suf}__{dname}__{sname}"

        # save the temporary config
        tmp_cfg_path = os.path.join(tmp_dir, f"{sanitize_name(cfg_i['run']['name'])}.yaml")
        save_yaml(cfg_i, tmp_cfg_path)
        generated.append(tmp_cfg_path)

        # run main.py
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
            # keep going with the next combination even on failure
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
