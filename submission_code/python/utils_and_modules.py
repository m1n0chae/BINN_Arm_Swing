# utils_and_modules.py
# =========================================================
# Shared utilities: paths/seeding, data loaders + normalisation,
# LOSO/fixed splits, NRMSE(range)-based training/evaluation,
# assembly of the MATLAB dicts into (X, y, sample_info),
# index-set-based train/val/test loaders, prediction collection.
# =========================================================
from __future__ import annotations
import os, json, time, random, math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

# ---------------------- Path / IO / Seed ----------------------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def expand_and_abs(path: str, base: Optional[str] = None) -> str:
    p = os.path.expandvars(os.path.expanduser(path))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.abspath(p)

def make_run_paths(root: str, model_name: str, seed: int, tag: Optional[str] = None) -> Dict[str, str]:
    root_abs = expand_and_abs(root)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_seed{seed}" + (f"_{tag}" if tag else "")
    run_dir = os.path.join(root_abs, model_name, name)
    ckpt = os.path.join(run_dir, "ckpt")
    fig  = os.path.join(run_dir, "fig")
    tb   = os.path.join(run_dir, "tb")
    for d in (run_dir, ckpt, fig, tb):
        ensure_dir(d)
    return {"run_dir": run_dir, "ckpt_dir": ckpt, "fig_dir": fig, "tb_dir": tb}

def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_box(msg: str):
    bar = "═" + "═"*len(msg) + "═"
    print(f"\n{bar}\n {msg} \n{bar}")

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Loader utilities ----------------------
def make_fold_loaders(
    X: np.ndarray,
    y: np.ndarray,
    sample_info: List[Dict[str, Any]],
    *,
    val_subject: str,
    batch_size: int = 64,
    n_workers: int = 2,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int, int, Optional[StandardScaler]]:
    """
    Build the Train/Val DataLoaders for one LOSO fold.
    - Returns: (train_loader, val_loader, in_F, out_C, seq_len_T, fitted_scaler_or_None)
    """
    idx_tr = [i for i, s in enumerate(sample_info) if s.get("subject") != val_subject]
    idx_vl = [i for i, s in enumerate(sample_info) if s.get("subject") == val_subject]
    if not idx_tr or not idx_vl:
        raise RuntimeError(f"[make_fold_loaders] Empty split for subject={val_subject}")

    Xtr, ytr = X[np.array(idx_tr)], y[np.array(idx_tr)]
    Xvl, yvl = X[np.array(idx_vl)], y[np.array(idx_vl)]

    Ntr, T, F = Xtr.shape
    C = ytr.shape[-1]

    # normalisation: fit on Train only
    scaler = None
    if normalize and F > 0:
        scaler = StandardScaler()
        Xtr2 = Xtr.reshape(-1, F)            # (Ntr*T, F)
        Xvl2 = Xvl.reshape(-1, F)            # (Nvl*T, F)
        Xtr_sc = scaler.fit_transform(Xtr2).reshape(Ntr, T, F).astype(np.float32)
        Xvl_sc = scaler.transform(Xvl2).reshape(len(Xvl), T, F).astype(np.float32)
    else:
        Xtr_sc = Xtr.astype(np.float32)
        Xvl_sc = Xvl.astype(np.float32)

    # tensorise as (B,T,F) / (B,T,C)
    tr_ds = TensorDataset(torch.from_numpy(Xtr_sc), torch.from_numpy(ytr.astype(np.float32)))
    vl_ds = TensorDataset(torch.from_numpy(Xvl_sc), torch.from_numpy(yvl.astype(np.float32)))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=n_workers, drop_last=False)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=False)

    return tr_loader, vl_loader, F, C, T, scaler


def make_loaders_from_index_sets(
    X: np.ndarray,
    y: np.ndarray,
    sample_info: List[Dict[str, Any]],
    *,
    idx_train: List[int],
    idx_val: Optional[List[int]] = None,
    idx_test: Optional[List[int]] = None,
    batch_size: int = 64,
    n_workers: int = 2,
    normalize: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], int, int, int, Optional[StandardScaler]]:
    """
    Build train/val/test DataLoaders from explicit index sets.
    - The scaler is fit on train only; val/test are transformed with it.
    - Returns: (train_loader, val_loader or None, test_loader or None, in_F, out_C, T, scaler)
    """
    idx_train = list(idx_train)
    idx_val   = list(idx_val or [])
    idx_test  = list(idx_test or [])

    if not idx_train:
        raise ValueError("[make_loaders_from_index_sets] idx_train must be non-empty.")

    Xtr, ytr = X[np.array(idx_train)], y[np.array(idx_train)]
    Ntr, T, F = Xtr.shape
    C = ytr.shape[-1]

    Xvl = yvl = None
    Xte = yte = None
    if idx_val:
        Xvl, yvl = X[np.array(idx_val)], y[np.array(idx_val)]
    if idx_test:
        Xte, yte = X[np.array(idx_test)], y[np.array(idx_test)]

    scaler = None
    if normalize and F > 0:
        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr.reshape(-1, F)).reshape(Ntr, T, F).astype(np.float32)
        if Xvl is not None:
            Xvl_sc = scaler.transform(Xvl.reshape(-1, F)).reshape(len(Xvl), T, F).astype(np.float32)
        else:
            Xvl_sc = None
        if Xte is not None:
            Xte_sc = scaler.transform(Xte.reshape(-1, F)).reshape(len(Xte), T, F).astype(np.float32)
        else:
            Xte_sc = None
    else:
        Xtr_sc = Xtr.astype(np.float32)
        Xvl_sc = None if Xvl is None else Xvl.astype(np.float32)
        Xte_sc = None if Xte is None else Xte.astype(np.float32)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_sc), torch.from_numpy(ytr.astype(np.float32)))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=False)

    if Xvl_sc is not None:
        vl_ds = TensorDataset(torch.from_numpy(Xvl_sc), torch.from_numpy(yvl.astype(np.float32)))
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=False)
    else:
        vl_loader = None

    if Xte_sc is not None:
        te_ds = TensorDataset(torch.from_numpy(Xte_sc), torch.from_numpy(yte.astype(np.float32)))
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=False)
    else:
        te_loader = None

    return tr_loader, vl_loader, te_loader, F, C, T, scaler


# ------------------ NRMSE evaluation ------------------
def nrmse_from_loader(loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    """
    Compute NRMSE(range) over a DataLoader.
    - input: (B, T, F), target: (B, T, C), prediction: (B, T, C)
    - RMSE is computed along the time axis (T); the denominator is the
      (max - min) of the target over the same time axis
    - batch means are combined as a count-weighted average
    """
    model.eval()
    total_weighted = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)              # (B,T,F)
            yb = yb.to(device)              # (B,T,C)
            yhat = model(xb)                # (B,T,C)

            # MSE → RMSE along the time axis
            err = yhat - yb                 # (B,T,C)
            mse_t = (err ** 2).mean(dim=1)  # (B,C)  ← mean over time
            rmse = torch.sqrt(mse_t + 1e-12)

            # normalise by the time-axis range
            rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)  # (B,C)
            nrmse_batch = (rmse / rng).mean()  # scalar

            bs = xb.size(0)
            total_weighted += float(nrmse_batch.item()) * bs
            total_count    += bs

    return total_weighted / max(1, total_count)


def collect_predictions(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (y_true, y_pred) for every sample in the loader (numpy arrays).
    """
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yhat = model(xb)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(yhat.cpu().numpy())
    ytrue = np.concatenate(y_true_list, axis=0)  # (N, T, C)
    ypred = np.concatenate(y_pred_list, axis=0)  # (N, T, C)
    return ytrue, ypred


# ------------------ Training (NRMSE tracked, MSE loss) ------------------
def train_collect_nrmse(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, *,
                        epochs: int, lr: float, weight_decay: float,
                        device: Optional[torch.device] = None,
                        log_dir: Optional[str] = None,
                        add_graph_once: bool = True,
                        ) -> Dict[str, Any]:
    """
    Train with MSE loss, track val NRMSE(range) per epoch.

    Only the final-epoch weights are saved, as last.pth. No checkpoint selected by
    validation loss is saved — under LOSO the validation fold IS the test subject,
    so such a selection would amount to test-set selection. The returned best_nrmse
    is the minimum of the learning curve and is used only by the hyperparameter
    search (tune_*.py).
    """
    device = device or get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    writer = SummaryWriter(log_dir) if log_dir else None

    best = float("inf")
    curve: List[float] = []
    train_nrmse_curve: List[float] = []
    first_nrmse_printed = False

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
        train_nrmse_curve.append(nrmse_from_loader(train_loader, model, device))

        nrmse_val = nrmse_from_loader(val_loader, model, device)
        curve.append(nrmse_val)

        if writer:
            writer.add_scalar("val/NRMSE_range", nrmse_val, epoch)

        if add_graph_once and epoch == 1 and writer:
            try:
                bx, _ = next(iter(train_loader))
                writer.add_graph(model, bx.to(device))
            except Exception:
                pass

        if nrmse_val < best:
            best = nrmse_val

        if epoch == 1 and not first_nrmse_printed:
            print(f"[epoch {epoch:03d}] val NRMSE={nrmse_val:.6f}  (first)")
            first_nrmse_printed = True
        if epoch == epochs:
            print(f"[epoch {epoch:03d}] val NRMSE={nrmse_val:.6f}  (last)  | best={best:.6f}")
            # save the final-epoch weights (paper protocol)
            if log_dir:
                ensure_dir(os.path.dirname(log_dir))
                last_path = os.path.join(os.path.dirname(log_dir), "last.pth")
                torch.save(model.state_dict(), last_path)
            last_nrmse = nrmse_val

    if writer:
        writer.close()

    out = {"val_nrmse_curve": curve, "train_nrmse_curve": train_nrmse_curve, "best_nrmse": best}
    if 'last_path' in locals(): out["last_path"] = last_path
    if 'last_nrmse' in locals(): out["last_nrmse"] = last_nrmse
    return out


# ------------------ NRMSE curve plotting ------------------
def safe_get(nested: Dict[str, Any], dotted_key: str) -> Any:
    """e.g. safe_get(d, 'sacrum.pos_x')"""
    cur: Any = nested
    for k in dotted_key.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            raise KeyError(f"Key not found: {dotted_key}")
    return cur

def build_dataset_from_mat(
    grf: Dict[str, Any],
    kin: Dict[str, Any],
    modeling: Optional[Dict[str, Any]] = None,
    *,
    sessions: List[str],
    grf_sides: List[str],
    grf_axes: List[str],
    body_parts: List[str],
    axes_pos: List[str],
    axes_vel: List[str],
    axes_acc: List[str],
    axes_imu_acc_local: List[str],
    axes_imu_acc_global: List[str],
    axes_imu_gyro_local: List[str],
    axes_imu_gyro_global: List[str],
    add_stride_duration_scalar: bool = True,
    window_mode: str = "percent",
    win_pct: Tuple[float, float] = (0.0, 100.0),
    win_time: Tuple[float, float] = (0.0, 0.5),
    use_modeling_input: bool = False,
    modeling_points_to_use: Optional[List[str]] = None,
    modeling_fields_to_use: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Collect *every stride* from the GRF/kinematics/modeling dicts as samples and
    return X(N,T,F), y(N,T,C), sample_info(list[dict]).
      - kin: kin[bp][subject][session][day][field] = [stride1, stride2, ...]
      - grf: grf[subject][session][day][side][axis] = [stride1, stride2, ...]
      - modeling (optional): modeling[point][subject][session][day][field] = [stride...]
    """
    modeling = modeling or {}
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    sample_info: List[Dict[str, Any]] = []

    # helper: safely extract the k-th stride from a field list
    def _get_stride(lst, k):
        return None if not (isinstance(lst, (list, tuple)) and len(lst) > k and lst[k] is not None) \
                     else np.asarray(lst[k]).reshape(-1)

    # ─────────────────────────────────────────────────────────────────
    # Iterate over Subject / Session / Day
    # ─────────────────────────────────────────────────────────────────
    for subj, sess_dict in grf.items():
        if not isinstance(sess_dict, dict):
            continue
        for sess in sessions:
            day_dict = sess_dict.get(sess, {})
            if not isinstance(day_dict, dict) or not day_dict:
                continue

            # sort Day keys
            days = [d for d in day_dict.keys() if str(d).lower().startswith("day")]
            days.sort(key=lambda s: int("".join([c for c in str(s) if c.isdigit()] or "0")))

            for day in days:
                side_root = day_dict.get(day, {})
                if not isinstance(side_root, dict) or not side_root:
                    continue

                # 1) collect the per-axis GRF stride lists for this Day
                per_axis_lists: List[List[np.ndarray]] = []
                for side in grf_sides:
                    side_node = side_root.get(side, {})
                    if not isinstance(side_node, dict):
                        continue
                    for ax in grf_axes:
                        key = next((k for k in (ax, ax.lower(), ax.upper(), ax[-1:].lower(), ax[-1:].upper())
                                    if k in side_node), None)
                        if key is None:
                            continue
                        seqs = side_node.get(key, [])
                        if isinstance(seqs, (list, tuple)) and len(seqs) > 0:
                            per_axis_lists.append(seqs)

                if not per_axis_lists:
                    continue

                # number of strides common to all axes
                n_stride = min(len(lst) for lst in per_axis_lists)
                if n_stride <= 0:
                    continue

                # 2) stride loop
                for k_idx in range(n_stride):
                    # 2-1) y(target; GRF)
                    y_chs: List[np.ndarray] = []
                    valid_y = True
                    for lst in per_axis_lists:
                        a = _get_stride(lst, k_idx)
                        if a is None or a.size == 0:
                            valid_y = False
                            break
                        y_chs.append(a)
                    if not valid_y or not y_chs:
                        continue

                    # 2-2) x (inputs; kinematics + imu + optional modeling)
                    x_feats: List[np.ndarray] = []

                    # kinematics/IMU
                    for bp in body_parts:
                        leaf = kin.get(bp, {}).get(subj, {}).get(sess, {}).get(day, {})
                        if not isinstance(leaf, dict):
                            continue

                        # pos/vel/acc
                        for ax in axes_pos:
                            a = _get_stride(leaf.get(f"pos_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)
                        for ax in axes_vel:
                            a = _get_stride(leaf.get(f"vel_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)
                        for ax in axes_acc:
                            a = _get_stride(leaf.get(f"acc_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)

                        # IMU acc local/global
                        for ax in axes_imu_acc_local:
                            a = _get_stride(leaf.get(f"imu_acc_local_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)
                        for ax in axes_imu_acc_global:
                            # try the current key first, then the legacy schema
                            a = _get_stride(leaf.get(f"imu_acc_global_{ax}", None), k_idx)
                            if a is None:
                                a = _get_stride(leaf.get(f"imu_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)

                        # IMU gyro local/global
                        for ax in axes_imu_gyro_local:
                            a = _get_stride(leaf.get(f"imu_gyro_local_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)
                        for ax in axes_imu_gyro_global:
                            a = _get_stride(leaf.get(f"imu_gyro_global_{ax}", None), k_idx)
                            if a is None:
                                a = _get_stride(leaf.get(f"imu_gyro_{ax}", None), k_idx)
                            if a is not None and a.size: x_feats.append(a)

                    # modeling input (optional)
                    if use_modeling_input and modeling and modeling_points_to_use and modeling_fields_to_use:
                        for mp in modeling_points_to_use:
                            mleaf = modeling.get(mp, {}).get(subj, {}).get(sess, {}).get(day, {})
                            if not isinstance(mleaf, dict):
                                continue
                            for mf in modeling_fields_to_use:
                                a = _get_stride(mleaf.get(mf, None), k_idx)
                                if a is not None and a.size: x_feats.append(a)

                    # 2-3) synchronise lengths within the stride (minimum over channels)
                    all_lens = [len(a) for a in y_chs] + ([len(a) for a in x_feats] if x_feats else [])
                    T_k = min(all_lens) if all_lens else 0
                    if T_k < 2:
                        continue

                    Y_k = np.stack([a[:T_k] for a in y_chs], axis=-1).astype(np.float32)  # (T_k, C)
                    if x_feats:
                        X_k = np.stack([a[:T_k] for a in x_feats], axis=-1).astype(np.float32)  # (T_k, F)
                    else:
                        X_k = np.zeros((T_k, 0), dtype=np.float32)

                    # 2-4) stride-duration scalar (optional)
                    if add_stride_duration_scalar:
                        sac = kin.get("sacrum", {}).get(subj, {}).get(sess, {}).get(day, {})
                        tlist = sac.get("time", None)
                        dur = None
                        if isinstance(tlist, (list, tuple)) and len(tlist) > k_idx and tlist[k_idx] is not None:
                            t_arr = np.asarray(tlist[k_idx]).reshape(-1)
                            if t_arr.size >= 2:
                                dur = float(t_arr[-1] - t_arr[0])
                        if dur is None:
                            dur = float(T_k) / 100.0  # 100 Hz approximation when no time info
                        X_k = np.concatenate([X_k, np.full((T_k, 1), dur, dtype=np.float32)], axis=1)

                    # 2-5) apply the window
                    if window_mode.lower() == "percent":
                        i0 = int(round(T_k * float(win_pct[0]) / 100.0))
                        i1 = int(round(T_k * float(win_pct[1]) / 100.0))
                        i0 = max(0, min(i0, T_k - 1))
                        i1 = max(i0 + 1, min(i1, T_k))
                        X_win = X_k[i0:i1]; Y_win = Y_k[i0:i1]
                    else:
                        # time mode: exact only when per-stride time exists; otherwise uniform-grid approximation
                        sac = kin.get("sacrum", {}).get(subj, {}).get(sess, {}).get(day, {})
                        tlist = sac.get("time", None)
                        if isinstance(tlist, (list, tuple)) and len(tlist) > k_idx and tlist[k_idx] is not None:
                            t_ref = np.asarray(tlist[k_idx]).reshape(-1)[:T_k]
                        else:
                            t_ref = np.arange(T_k, dtype=float) / 100.0
                        mask = (t_ref >= float(win_time[0])) & (t_ref <= float(win_time[1]))
                        idx = np.nonzero(mask)[0]
                        if idx.size < 2:
                            continue
                        X_win = X_k[idx]; Y_win = Y_k[idx]

                    # append the sample
                    X_list.append(X_win)
                    Y_list.append(Y_win)
                    sample_info.append({"subject": subj, "session": sess, "day": day, "stride_idx": k_idx + 1})

    # crop every sample to the common length (min T) and stack
    if not X_list:
        raise RuntimeError("No samples assembled from provided MAT dictionaries.")

    # ------------------------------------------------------------------
    # Drop samples with a deficient channel count.
    #
    # This function silently skips fields missing from the .mat (it simply does
    # not append to x_feats). If a requested field is entirely absent for one
    # subject/session, only those samples end up with a smaller F, and the
    # np.stack below would die with
    #   ValueError: all input arrays must have the same shape
    # without saying which subject caused it.
    #
    # Real case: in merged_stride_kinematics_arm_gyro_day1.mat the sacrum body
    #   part has all six imu_acc_local_* / imu_gyro_local_* fields empty for
    #   S012 / ss3. (L_Wrist is complete for every session, so wrist-based
    #   experiments never exposed this.)
    #
    # The most frequent channel count is treated as canonical and deficient
    # samples are dropped, but what was dropped is always printed — this is
    # not information to silently swallow.
    # ------------------------------------------------------------------
    feat_counts = [x.shape[1] for x in X_list]
    if len(set(feat_counts)) > 1:
        vals, cnts = np.unique(np.asarray(feat_counts), return_counts=True)
        F_expected = int(vals[int(np.argmax(cnts))])

        dropped: Dict[str, int] = {}
        keep_idx = []
        for i, fc in enumerate(feat_counts):
            if fc == F_expected:
                keep_idx.append(i)
            else:
                si = sample_info[i]
                key = f"{si.get('subject')}/{si.get('session')}/{si.get('day')} (F={fc})"
                dropped[key] = dropped.get(key, 0) + 1

        print(f"[build_dataset] ⚠ dropped {len(X_list) - len(keep_idx)} strides with mismatched "
              f"channel counts (canonical F={F_expected}, out of {len(X_list)} total):")
        for key in sorted(dropped):
            print(f"    - {key}: {dropped[key]} strides")

        X_list = [X_list[i] for i in keep_idx]
        Y_list = [Y_list[i] for i in keep_idx]
        sample_info = [sample_info[i] for i in keep_idx]
        if not X_list:
            raise RuntimeError("No samples remain after the channel-count filter.")

    min_T = min(x.shape[0] for x in X_list)
    X_out = np.stack([x[:min_T] for x in X_list], axis=0)
    y_out = np.stack([y[:min_T] for y in Y_list], axis=0)
    return X_out.astype(np.float32), y_out.astype(np.float32), sample_info

def nrmse_by_channel(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    """
    Compute NRMSE(range) per channel.
    Returns: numpy array of shape (C,) — each channel's NRMSE(range), sample mean.
    Definition:
      RMSE(time) / range(time) per sample/channel → mean over the sample axis.
    """
    model.eval()
    sum_per_ch: Optional[torch.Tensor] = None  # (C,)
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)          # (B,T,F)
            yb = yb.to(device)          # (B,T,C)
            yhat = model(xb)            # (B,T,C)

            # per-sample/channel RMSE over time
            err = yhat - yb                           # (B,T,C)
            mse_t = (err ** 2).mean(dim=1)           # (B,C)
            rmse  = torch.sqrt(mse_t + 1e-12)        # (B,C)

            # per-sample/channel normalization by time-range
            rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)  # (B,C)
            nrmse_bc = rmse / rng                    # (B,C)

            batch_sum = nrmse_bc.sum(dim=0)          # (C,)
            sum_per_ch = batch_sum.cpu() if sum_per_ch is None else sum_per_ch + batch_sum.cpu()
            total_samples += xb.size(0)

    if sum_per_ch is None or total_samples == 0:
        return np.array([], dtype=np.float32)
    return (sum_per_ch / float(total_samples)).numpy()  # (C,)