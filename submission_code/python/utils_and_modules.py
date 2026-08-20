# utils_and_modules.py
# =========================================================
# 공통 유틸/경로/시드, 데이터 로더+표준화, LOSO/fixed 스플릿,
# NRMSE(range) 전용 학습/평가, fold별 NRMSE 곡선 시각화,
# (옵션) MATLAB dict → (X,y,sample_info) 조립 유틸
# + 인덱스 기반 train/val/test 로더 생성, 예측 수집 유틸 추가
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
import matplotlib.pyplot as plt

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
    (기존) LOSO 한 fold의 Train/Val DataLoader를 만듭니다.
    - NOTE: 논문용 LOSO에서는 이 함수 대신 make_loaders_from_index_sets를 쓰는 것을 권장
    - 반환: (train_loader, val_loader, in_F, out_C, seq_len_T, fitted_scaler_or_None)
    """
    idx_tr = [i for i, s in enumerate(sample_info) if s.get("subject") != val_subject]
    idx_vl = [i for i, s in enumerate(sample_info) if s.get("subject") == val_subject]
    if not idx_tr or not idx_vl:
        raise RuntimeError(f"[make_fold_loaders] Empty split for subject={val_subject}")

    Xtr, ytr = X[np.array(idx_tr)], y[np.array(idx_tr)]
    Xvl, yvl = X[np.array(idx_vl)], y[np.array(idx_vl)]

    Ntr, T, F = Xtr.shape
    C = ytr.shape[-1]

    # 표준화: Train만 fit
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

    # (B,T,F) / (B,T,C) 그대로 텐서화
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
    주어진 인덱스 집합으로 train/val/test DataLoader를 생성.
    - 표준화는 train에서만 fit, 나머지는 transform.
    - 반환: (train_loader, val_loader or None, test_loader or None, in_F, out_C, T, scaler)
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
    DataLoader에서 NRMSE(range)를 계산합니다.
    - 입력: (B, T, F), 타깃: (B, T, C), 예측: (B, T, C)
    - 시간축(T)에 대해 RMSE를 계산하고, 분모는 같은 시간축의 (max - min)
    - 배치 단위 평균을 가중 평균으로 종합
    """
    model.eval()
    total_weighted = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)              # (B,T,F)
            yb = yb.to(device)              # (B,T,C)
            yhat = model(xb)                # (B,T,C)

            # 시간축에 대해 MSE → RMSE
            err = yhat - yb                 # (B,T,C)
            mse_t = (err ** 2).mean(dim=1)  # (B,C)  ← 시간축 평균
            rmse = torch.sqrt(mse_t + 1e-12)

            # 시간축 범위로 정규화
            rng = (yb.max(dim=1).values - yb.min(dim=1).values).clamp_min(1e-8)  # (B,C)
            nrmse_batch = (rmse / rng).mean()  # 스칼라

            bs = xb.size(0)
            total_weighted += float(nrmse_batch.item()) * bs
            total_count    += bs

    return total_weighted / max(1, total_count)


def collect_predictions(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 loader에서 (y_true, y_pred)를 모두 수집 (numpy 반환).
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

    마지막 epoch 가중치만 last.pth 로 저장한다. 검증 기준으로 고른 체크포인트는
    저장하지 않는다 — LOSO 에서는 val 폴드가 곧 test 피험자이므로 그런 선택은
    test set selection 이 된다. 반환되는 best_nrmse 는 학습곡선의 최솟값으로,
    하이퍼파라미터 탐색(tune_*.py)에서만 쓰인다.
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
            # 마지막 epoch 가중치 저장 (논문 프로토콜)
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
def plot_mean_std_shaded(curves: List[List[float]], out_path: str,
                         title: str = "Validation NRMSE (mean ± sd)",
                         ylabel: str = "NRMSE (range)") -> None:
    """Average ± std of per-epoch NRMSE curves across folds (for a given seed)."""
    if not curves:
        print("[plot] no curves; skip")
        return
    A = np.array(curves, dtype=float)  # (n_folds, epochs)
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    x = np.arange(1, mu.size + 1)

    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(9, 5))
    plt.grid(True, alpha=0.3)
    plt.plot(x, mu, label="mean", linewidth=2)
    if np.any(sd > 0):
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.25, label="±1 SD")
    plt.xlabel("epoch"); plt.ylabel(ylabel)
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_series_mean_std_with_raw(
    info: List[Dict[str, Any]],
    y_true: np.ndarray,            # (N, T, C)
    y_pred: np.ndarray,            # (N, T, C)
    save_path: str,
    *,
    channels: Optional[List[int]] = None,       # None → 모든 채널
    trials_include: Optional[List[str]] = None, # 예: ["walk1","walk3"]
    title: str = "Validation • per-channel"
) -> None:
    """
    채널별 subplot으로 3개 그림 저장:
      1) <base>.png          : raw + True mean(검정) + Pred mean(컬러 점선) + 각 ±1sd 음영
      2) <base>_meanstd.png  : mean ± sd 만
      3) <base>_mean.png     : mean 선만
    """
    import os
    import matplotlib.pyplot as plt

    # --- 기본 검사 ---
    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError("y_true, y_pred는 (N, T, C) 형태여야 합니다.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true와 y_pred의 shape가 동일해야 합니다.")
    N, T, C = y_true.shape

    # trial 필터
    mask = np.ones((N,), dtype=bool)
    if trials_include:
        names = np.array([str(d.get("trial", "")) for d in info])
        mask = np.isin(names, trials_include)

    yt = y_true[mask]  # (Nf,T,C)
    yp = y_pred[mask]  # (Nf,T,C)
    if yt.shape[0] == 0:
        print("[plot] trials_include에 매칭되는 샘플이 없습니다. 스킵합니다.")
        return

    # 채널 선택
    if channels is None:
        chans = list(range(C))
    else:
        chans = [int(c) for c in channels if 0 <= int(c) < C]
        if not chans:
            print("[plot] 유효한 channels가 없습니다. 스킵합니다.")
            return

    # 저장 경로 준비
    base, ext = os.path.splitext(save_path)
    if ext.lower() != ".png":
        ext = ".png"
        save_path = base + ext
    ensure_dir(os.path.dirname(save_path) or ".")

    # 안전한 색상 사이클
    try:
        cycle = plt.rcParams.get("axes.prop_cycle", None)
        colors = cycle.by_key().get("color", []) if hasattr(cycle, "by_key") else []
    except Exception:
        colors = []
    if not colors:
        colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                  "#9467bd","#8c564b","#e377c2","#7f7f7f","#17beca"]

    # 채널별 통계
    mu_t = {c: yt[:, :, c].mean(axis=0) for c in chans}
    sd_t = {c: yt[:, :, c].std(axis=0)  for c in chans}
    mu_p = {c: yp[:, :, c].mean(axis=0) for c in chans}
    sd_p = {c: yp[:, :, c].std(axis=0)  for c in chans}
    x = np.arange(T)

    # ---------- 1) RAW + mean ± sd ----------
    figA, axesA = plt.subplots(len(chans), 1, figsize=(10, 3.4*len(chans)), sharex=True)
    if len(chans) == 1:
        axesA = [axesA]
    for k, c in enumerate(chans):
        ax  = axesA[k]
        col = colors[k % len(colors)]
        # raw
        for i in range(yt.shape[0]):
            ax.plot(x, yt[i,:,c], color="#666666", alpha=0.35, linewidth=0.8)               # True raw
            ax.plot(x, yp[i,:,c], color=col,     alpha=0.35, linewidth=0.8, linestyle="--") # Pred raw
        # mean ± sd
        ax.fill_between(x, mu_t[c]-sd_t[c], mu_t[c]+sd_t[c], color="#bbbbbb", alpha=0.35, label="True ±sd")
        ax.fill_between(x, mu_p[c]-sd_p[c], mu_p[c]+sd_p[c], color=col,       alpha=0.20, label="Pred ±sd")
        ax.plot(x, mu_t[c], color="black", linewidth=2.0, label=f"True ch{c}")
        ax.plot(x, mu_p[c], color=col,     linewidth=2.0, linestyle="--", label=f"Pred ch{c}")
        ax.set_ylabel("amp"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
        ax.set_title(f"ch{c} • raw + mean ± sd", fontsize=11)
        if c == 0:
            ax.set_ylim(-0.5, 0.5)
        elif c == 1:
            ax.set_ylim(0.5, 2.0)

    axesA[-1].set_xlabel("time index")
    figA.suptitle(f"{title} — raw + mean ± sd", fontsize=13)
    figA.tight_layout(rect=[0,0,1,0.96])
    figA.savefig(save_path, dpi=150); plt.close(figA)
    # print(f"[OK] saved: {save_path}")

    # ---------- 2) mean ± sd only ----------
    out_meanstd = f"{base}_meanstd{ext}"
    figB, axesB = plt.subplots(len(chans), 1, figsize=(10, 3.0*len(chans)), sharex=True)
    if len(chans) == 1:
        axesB = [axesB]
    for k, c in enumerate(chans):
        ax  = axesB[k]
        col = colors[k % len(colors)]
        ax.fill_between(x, mu_t[c]-sd_t[c], mu_t[c]+sd_t[c], color="#bbbbbb", alpha=0.35, label="True ±sd")
        ax.fill_between(x, mu_p[c]-sd_p[c], mu_p[c]+sd_p[c], color=col,       alpha=0.20, label="Pred ±sd")
        ax.plot(x, mu_t[c], color="black", linewidth=2.2, label=f"True ch{c}")
        ax.plot(x, mu_p[c], color=col,     linewidth=2.2, linestyle="--", label=f"Pred ch{c}")
        ax.set_ylabel("amp"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
        ax.set_title(f"ch{c} • mean ± sd", fontsize=11)
        if c == 0:
            ax.set_ylim(-0.5, 0.5)
        elif c == 1:
            ax.set_ylim(0.5, 2.0)

    axesB[-1].set_xlabel("time index")
    figB.suptitle(f"{title} — mean ± sd", fontsize=13)
    figB.tight_layout(rect=[0,0,1,0.96])
    figB.savefig(out_meanstd, dpi=150); plt.close(figB)
    # print(f"[OK] saved: {out_meanstd}")

    # ---------- 3) mean only ----------
    out_mean = f"{base}_mean{ext}"
    figC, axesC = plt.subplots(len(chans), 1, figsize=(10, 2.8*len(chans)), sharex=True)
    if len(chans) == 1:
        axesC = [axesC]
    for k, c in enumerate(chans):
        ax  = axesC[k]
        col = colors[k % len(colors)]
        ax.plot(x, mu_t[c], color="black", linewidth=3.0, label=f"True ch{c}")
        ax.plot(x, mu_p[c], color=col,     linewidth=3.0, linestyle="--", label=f"Pred ch{c}")
        ax.set_ylabel("amp"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
        ax.set_title(f"ch{c} • mean only", fontsize=11)
        if c == 0:
            ax.set_ylim(-0.5, 0.5)
        elif c == 1:
            ax.set_ylim(0.5, 2.0)

    axesC[-1].set_xlabel("time index")
    figC.suptitle(f"{title} — mean only", fontsize=13)
    figC.tight_layout(rect=[0,0,1,0.96])
    figC.savefig(out_mean, dpi=150); plt.close(figC)
    # print(f"[OK] saved: {out_mean}")


# ------------------ (Optional) MATLAB dict -> X,y assembler ------------------
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
    GRF/kinematics/modeling dict에서 *모든 stride*를 샘플로 수집해
    X(N,T,F), y(N,T,C), sample_info(list[dict])를 반환합니다.
      - kin: kin[bp][subject][session][day][field] = [stride1, stride2, ...]
      - grf: grf[subject][session][day][side][axis] = [stride1, stride2, ...]
      - modeling(옵션): modeling[point][subject][session][day][field] = [stride...]
    """
    modeling = modeling or {}
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    sample_info: List[Dict[str, Any]] = []

    # 유틸: field 리스트에서 k번째 stride를 안전히 추출
    def _get_stride(lst, k):
        return None if not (isinstance(lst, (list, tuple)) and len(lst) > k and lst[k] is not None) \
                     else np.asarray(lst[k]).reshape(-1)

    # ─────────────────────────────────────────────────────────────────
    # Subject / Session / Day 순회
    # ─────────────────────────────────────────────────────────────────
    for subj, sess_dict in grf.items():
        if not isinstance(sess_dict, dict):
            continue
        for sess in sessions:
            day_dict = sess_dict.get(sess, {})
            if not isinstance(day_dict, dict) or not day_dict:
                continue

            # Day 키 정렬
            days = [d for d in day_dict.keys() if str(d).lower().startswith("day")]
            days.sort(key=lambda s: int("".join([c for c in str(s) if c.isdigit()] or "0")))

            for day in days:
                side_root = day_dict.get(day, {})
                if not isinstance(side_root, dict) or not side_root:
                    continue

                # 1) 이 Day에서 사용할 GRF 축별 stride 리스트를 수집
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

                # 축들 간 공통 stride 개수
                n_stride = min(len(lst) for lst in per_axis_lists)
                if n_stride <= 0:
                    continue

                # 2) stride 루프
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

                    # 2-2) x(inputs; kinematics + imu + (옵션) modeling)
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
                            # 최신 키 → 구 스키마 순으로 시도
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

                    # modeling 입력(옵션)
                    if use_modeling_input and modeling and modeling_points_to_use and modeling_fields_to_use:
                        for mp in modeling_points_to_use:
                            mleaf = modeling.get(mp, {}).get(subj, {}).get(sess, {}).get(day, {})
                            if not isinstance(mleaf, dict):
                                continue
                            for mf in modeling_fields_to_use:
                                a = _get_stride(mleaf.get(mf, None), k_idx)
                                if a is not None and a.size: x_feats.append(a)

                    # 2-3) stride 내 길이 동기화 (모든 채널 최소 길이)
                    all_lens = [len(a) for a in y_chs] + ([len(a) for a in x_feats] if x_feats else [])
                    T_k = min(all_lens) if all_lens else 0
                    if T_k < 2:
                        continue

                    Y_k = np.stack([a[:T_k] for a in y_chs], axis=-1).astype(np.float32)  # (T_k, C)
                    if x_feats:
                        X_k = np.stack([a[:T_k] for a in x_feats], axis=-1).astype(np.float32)  # (T_k, F)
                    else:
                        X_k = np.zeros((T_k, 0), dtype=np.float32)

                    # 2-4) stride duration 스칼라(옵션)
                    if add_stride_duration_scalar:
                        sac = kin.get("sacrum", {}).get(subj, {}).get(sess, {}).get(day, {})
                        tlist = sac.get("time", None)
                        dur = None
                        if isinstance(tlist, (list, tuple)) and len(tlist) > k_idx and tlist[k_idx] is not None:
                            t_arr = np.asarray(tlist[k_idx]).reshape(-1)
                            if t_arr.size >= 2:
                                dur = float(t_arr[-1] - t_arr[0])
                        if dur is None:
                            dur = float(T_k) / 100.0  # 시간 정보 없으면 100 Hz 근사
                        X_k = np.concatenate([X_k, np.full((T_k, 1), dur, dtype=np.float32)], axis=1)

                    # 2-5) 윈도우 적용
                    if window_mode.lower() == "percent":
                        i0 = int(round(T_k * float(win_pct[0]) / 100.0))
                        i1 = int(round(T_k * float(win_pct[1]) / 100.0))
                        i0 = max(0, min(i0, T_k - 1))
                        i1 = max(i0 + 1, min(i1, T_k))
                        X_win = X_k[i0:i1]; Y_win = Y_k[i0:i1]
                    else:
                        # time 모드: stride별 time이 있는 경우만 정확, 없으면 등간격 근사
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

                    # 최종 추가
                    X_list.append(X_win)
                    Y_list.append(Y_win)
                    sample_info.append({"subject": subj, "session": sess, "day": day, "stride_idx": k_idx + 1})

    # 모든 샘플을 공통 길이(min T)로 잘라서 stack
    if not X_list:
        raise RuntimeError("No samples assembled from provided MAT dictionaries.")

    # ------------------------------------------------------------------
    # [revision] 채널 수가 모자란 샘플 제거
    #
    # 이 함수는 .mat 에 없는 필드를 조용히 건너뛴다(x_feats.append 를 안 함).
    # 따라서 특정 피험자/세션에서 요청한 필드가 통째로 비어 있으면 그 샘플만
    # F가 작아지고, 원본은 아래 np.stack 에서
    #   ValueError: all input arrays must have the same shape
    # 로 죽어버린다 (어느 피험자 때문인지 알려주지 않는다).
    #
    # 실제 사례: merged_stride_kinematics_arm_gyro_day1.mat 의 sacrum body part는
    #   S012 / ss3 에서 imu_acc_local_* , imu_gyro_local_* 6개가 전부 비어 있다.
    #   (L_Wrist 는 전 세션 완전하므로 손목 기반 실험에서는 드러나지 않았다.)
    #
    # 여기서는 최빈 채널 수를 정상으로 보고 그에 못 미치는 샘플을 버리되,
    # 무엇을 몇 개 버렸는지 반드시 출력한다. 조용히 넘어가면 안 되는 정보다.
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

        print(f"[build_dataset] ⚠ 채널 수 불일치로 {len(X_list) - len(keep_idx)}개 stride 제외 "
              f"(정상 F={F_expected}, 전체 {len(X_list)}개 중):")
        for key in sorted(dropped):
            print(f"    - {key}: {dropped[key]} strides")

        X_list = [X_list[i] for i in keep_idx]
        Y_list = [Y_list[i] for i in keep_idx]
        sample_info = [sample_info[i] for i in keep_idx]
        if not X_list:
            raise RuntimeError("채널 수 필터 후 남은 샘플이 없습니다.")

    min_T = min(x.shape[0] for x in X_list)
    X_out = np.stack([x[:min_T] for x in X_list], axis=0)
    y_out = np.stack([y[:min_T] for y in Y_list], axis=0)
    return X_out.astype(np.float32), y_out.astype(np.float32), sample_info

def nrmse_by_channel(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    """
    채널별 NRMSE(range)를 계산합니다.
    반환: shape (C,) numpy array — 각 채널의 NRMSE(range) (샘플 평균)
    정의:
      각 샘플/채널에 대해 RMSE(time) / range(time) 계산 → 샘플 축 평균
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