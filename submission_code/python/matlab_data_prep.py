#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matlab_data_prep.py
-------------------
HDF5 기반 MATLAB .mat 파일을 안전하게 파싱하여 Python dict/ndarray 구조로 변환하고,
지정한 최상위 struct들을 로드/덤프하는 유틸리티.

주요 제공 함수:
- load_mat_struct(path, struct_name) -> Dict[str, Any]
- load_all_mat_data(grf_path, grf_struct, kin_path, kin_struct, modeling_path=None, modeling_struct=None) -> (grf, kin, modeling)
- print_mat_tree(path)  # 파일 내부 트리 구조 확인용

호환성:
- MATLAB v7.3(HDF5) 포맷 지원. (필요 강제: pip install h5py)
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Any, Dict, Tuple, Optional

import h5py
import numpy as np


# --------------------------
# Low-level HDF5 → Python
# --------------------------
def _to_py(obj: Any) -> Any:
    """
    Convert an h5py object (Group or Dataset) to Python native structures.
    - Datasets of dtype object: iterate and resolve references.
    - Datasets of numeric types: return numpy arrays.
    - Strings/char arrays: decode to Python str where possible.
    - Groups: return dict of keys -> converted children.
    """
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        # Handle scalar bytes → decode
        if isinstance(data, (bytes, np.bytes_)):
            try:
                return data.decode("utf-8", "ignore")
            except Exception:
                return data
        # If object array (e.g., cell arrays or object references)
        if isinstance(data, np.ndarray) and data.dtype.kind == "O":
            out = []
            for elem in data.flat:
                if isinstance(elem, h5py.Reference):
                    # dereference HDF5 object
                    try:
                        deref = obj.file[elem]
                        out.append(_to_py(deref))
                    except Exception:
                        out.append(None)
                else:
                    # Could be nested python objects or bytes
                    if isinstance(elem, (bytes, np.bytes_)):
                        try:
                            out.append(elem.decode("utf-8", "ignore"))
                        except Exception:
                            out.append(elem)
                    else:
                        out.append(elem)
            return out
        # If char array (MATLAB strings often stored as numpy.bytes_ or np.ndarray of 'S' dtype)
        if isinstance(data, np.ndarray) and data.dtype.kind in ("S", "U"):
            try:
                return data.astype(str)
            except Exception:
                return data
        # Basic numeric or other ndarray/scalar
        return data

    if isinstance(obj, h5py.Group):
        # Convert members to dict with recursion
        return {k: _to_py(v) for k, v in obj.items()}

    # Unknown type: return as-is
    return obj


def _load_hdf5_struct(path: str, struct_name: str) -> Dict[str, Any]:
    """
    Load a top-level group named `struct_name` from an HDF5 (.mat v7.3) file and
    convert to nested Python structures.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAT file not found: {path}")

    with h5py.File(path, "r") as f:
        if struct_name not in f:
            raise KeyError(f"Struct '{struct_name}' not found in {path}. "
                           f"Top-level keys: {list(f.keys())}")
        grp = f[struct_name]
        pyobj = _to_py(grp)
        # Some v7.3 files store struct arrays at top-level: we return dict of first element if needed
        if isinstance(pyobj, list) and pyobj and isinstance(pyobj[0], dict):
            # typical .mat -> /struct_name[0] is the actual dict
            return pyobj[0]
        if isinstance(pyobj, dict):
            return pyobj
        # fallback
        return {"value": pyobj}


# --------------------------
# Public API
# --------------------------
def load_mat_struct(path: str, struct_name: str) -> Dict[str, Any]:
    """
    Public wrapper for _load_hdf5_struct. Returns {} if missing.
    """
    try:
        return _load_hdf5_struct(path, struct_name)
    except Exception as e:
        sys.stderr.write(f"[matlab_data_prep] Warning: {e}\n")
        return {}


def load_all_mat_data(
    grf_path: str,
    grf_struct: str,
    kin_path: str,
    kin_struct: str,
    modeling_path: Optional[str] = None,
    modeling_struct: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load all three main .mat structures as Python dicts. Missing files/keys yield {} for that component.

    Returns:
      (grf_dict, kinematics_dict, modeling_dict)
    """
    grf = load_mat_struct(grf_path, grf_struct) if grf_path else {}
    kin = load_mat_struct(kin_path, kin_struct) if kin_path else {}
    mdl = load_mat_struct(modeling_path, modeling_struct) if (modeling_path and modeling_struct) else {}
    return grf, kin, mdl


# --------------------------
# Utility: HDF5 Tree Printer
# --------------------------
def print_mat_tree(path: str) -> None:
    """
    Pretty-print HDF5 (.mat) file structure for debugging:
      - Top-level groups and datasets
      - Children recursively with dtypes/shapes
    """
    if not os.path.exists(path):
        print(f"[matlab_data_prep] File not found: {path}")
        return

    def _fmt_node(name: str, obj: Any, depth: int = 0):
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}/")
            for k, v in obj.items():
                _fmt_node(k, v, depth + 1)
        elif isinstance(obj, h5py.Dataset):
            dt = obj.dtype
            shape = obj.shape
            print(f"{indent}- {name}: Dataset shape={shape}, dtype={dt}")
        else:
            print(f"{indent}- {name}: {type(obj)}")

    with h5py.File(path, "r") as f:
        print(f"=== HDF5 Tree: {path} ===")
        for k, v in f.items():
            _fmt_node(k, v, 0)


# --------------------------
# CLI for quick inspection
# --------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Load MATLAB .mat (HDF5) and show contents or dump to JSON")
    ap.add_argument("--grf", dest="grf_path", default="", help="Path to GRF .mat (v7.3)")
    ap.add_argument("--g", dest="grf_struct", default="stride_grf", help="Top-level group name for GRF")
    ap.add_argument("--kin", dest="kin_path", default="", help="Path to kinematics .mat")
    ap.add_argument("--k", dest="kin_struct", default="stride_kinematics_arm", help="Top-level group for kinematics")
    ap.add_argument("--mdl", dest="mdl_path", default="", help="(Optional) Path to modeling .mat")
    ap.add_argument("--m", dest="mdl_struct", default="stride_modeling", help="(Optional) Top-level group for modeling")
    ap.add_argument("--dump-json", dest="dump_json", default="", help="If set, dump loaded dicts to this JSON file")
    ap.add_argument("--print-tree", action="store_true", help="Print HDF5 tree of provided files")
    return ap.parse_args()


def main():
    args = _parse_args()

    if args.print_tree:
        if args.grf_path:
            print_mat_tree(args.grf_path)
        if args.kin_path:
            print_mat_tree(args.kin_path)
        if args.mdl_path:
            print_mat_tree(args.mdl_path)
        return

    grf, kin, mdl = load_all_mat_data(
        grf_path=args.grf_path,
        grf_struct=args.grf_struct,
        kin_path=args.kin_path,
        kin_struct=args.kin_struct,
        modeling_path=args.mdl_path,
        modeling_struct=args.mdl_struct,
    )

    # Simple console summary
    print("=== Loaded Structures ===")
    print(f"GRF keys        : {list(grf.keys())[:10]}")
    print(f"Kinematics keys : {list(kin.keys())[:10]}")
    print(f"Modeling keys   : {list(mdl.keys())[:10]}")

    if args.dump_json:
        payload = {"grf": grf, "kinematics": kin, "modeling": mdl}
        with open(args.dump_json, "w", encoding="utf-8") as f:
            import json
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[dumped] -> {args.dump_json}")


# Alias for backward-compatibility with older code
def load_hdf5_struct(path: str, struct_name: str) -> Dict[str, Any]:
    """Alias retained for backward compatibility."""
    return load_mat_struct(path, struct_name)


if __name__ == "__main__":
    main()
