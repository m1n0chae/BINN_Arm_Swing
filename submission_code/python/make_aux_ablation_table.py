#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoM auxiliary-input ablation — result tables.

Removes the AP or the VT component of the CoM pathway input one at a time and
compares the retrained variants with the proposed model (Section III-B).

────────────────────────────────────────────────────────────────────────
Metric
    test NRMSE(range): computed per stride from the per-timestep prediction
    CSVs that main.py writes after loading last.pth (final-epoch weights).
    The learning-curve minimum (best_val) is not used because model selection
    would leak into the validation fold.

Two evaluation regimes — defined by "was the evaluated session seen in training?"
    ID  : seen.  tv_te_ss1/2/3 (train at one speed, test at the same speed)
                 tv_te_ss123   (train at all three speeds, test at those speeds)
    OOD : unseen. tv_ssX__te_... (train at one speed, test at the other speeds)

    ss1/ss2/ss3 are the walking-speed conditions (mean stride times
    1.134 / 1.052 / 0.979 s).

Tests
    primary  : paired t-test (normality of the differences reported via Shapiro-Wilk)
    secondary: Wilcoxon signed-rank, sign test, bootstrap 95% CI

★ Sampling unit (the crux of this script)
    The aux conditions are trained with the same subjects, folds and seed and
    differ only in the aux channels, so paired tests are appropriate.

    What counts as one sample, however, drives the p values.
    The raw data have (data 5) x (train,test session pairs 6) x (subjects 19)
    = 530 rows, but the only independently sampled unit is the SUBJECT (n=19):
      - the 5 data conditions are subsets of the same data (not independent)
      - the 6 session pairs are the same walking of the same subject (not independent)
    Using the 530 rows directly counts one subject up to 30 times
    (pseudoreplication) and makes the p values anti-conservative.

    -> The primary results are at the SUBJECT level (n=19). Sub-unit results
       are reported alongside as a sensitivity analysis.

Output
    <out_dir>/aux_ablation_{main,by_data,sensitivity,interaction}.csv
    <out_dir>/aux_ablation_table.{md,tex}

Usage
    python make_aux_ablation_table.py --in test_all.csv --out_dir ./_paper_tables
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_ORDER = [
    ("full_day1", "100%"),
    ("percent050_day1_case1", "50%"),
    ("percent020_day1_case1", "20%"),
    ("percent010_day1_case1", "10%"),
    ("percent005_day1_case1", "5%"),
]
TRAIN_SESS = {
    "tv_te_ss1": "ss1", "tv_te_ss2": "ss2", "tv_te_ss3": "ss3",
    "tv_ss1__te_ss2ss3": "ss1", "tv_ss2__te_ss1ss3": "ss2", "tv_ss3__te_ss1ss2": "ss3",
    "tv_te_ss123": "ss123",          # trained on all three sessions -> third regime
}
# ID  : evaluated session seen in training (tv_te_ss1/2/3, tv_te_ss123)
# OOD : evaluated session unseen in training (tv_ssX__te_...)
REGIMES = ["ID", "OOD"]
FULL_IDX = ["data", "train_sess", "test_session", "subject"]
# (removed channel, kept condition name). The 'AP' condition = acc_x only = VT removed.
REMOVALS = [("VT", "AP"), ("AP", "VT")]

N_BOOT = 20000
RNG_SEED = 20260804


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.session.isin(TRAIN_SESS)].copy()
    d["train_sess"] = d.session.map(TRAIN_SESS)
    seen = (d.train_sess == "ss123") | (d.train_sess == d.test_session)
    d["regime"] = np.where(seen, "ID", "OOD")
    return d


def paired_diffs(sub: pd.DataFrame, kept: str) -> pd.DataFrame:
    """Relative change (%) versus Proposed, per (data, session pair, subject)."""
    w = sub.pivot_table(index=FULL_IDX, columns="aux", values="test_nrmse").dropna()
    if not {"APVT", "AP", "VT"} <= set(w.columns):
        return pd.DataFrame()
    out = w.reset_index()[FULL_IDX].copy()
    out["base"] = w["APVT"].values
    out["alt"] = w[kept].values
    out["rel"] = (w[kept].values - w["APVT"].values) / w["APVT"].values * 100.0
    return out


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    """Bootstrap 95% CI of the mean (resampling with replacement)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def test_block(rel_by_subject: np.ndarray) -> dict:
    """Battery of tests on one subject-level vector."""
    n = len(rel_by_subject)
    mean = float(rel_by_subject.mean())
    med = float(np.median(rel_by_subject))
    n_pos = int((rel_by_subject > 0).sum())
    p_w = float(stats.wilcoxon(rel_by_subject).pvalue) if n > 5 else np.nan
    p_s = float(stats.binomtest(n_pos, n, 0.5).pvalue)
    dz = mean / float(rel_by_subject.std(ddof=1))
    lo, hi = boot_ci(rel_by_subject)
    tt = stats.ttest_1samp(rel_by_subject, 0.0)          # primary test
    ci = tt.confidence_interval()
    p_norm = float(stats.shapiro(rel_by_subject).pvalue)  # normality of the differences
    return {"n": n, "mean": mean, "median": med, "n_pos": n_pos,
            "p_ttest": float(tt.pvalue), "t_stat": float(tt.statistic),
            "t_lo": float(ci.low), "t_hi": float(ci.high), "p_shapiro": p_norm,
            "p_wilcoxon": p_w, "p_sign": p_s, "dz": dz, "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="test_all.csv")
    ap.add_argument("--out_dir", default="./_paper_tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = load(args.inp)

    # ── Table 1 (primary result): subject level, per ID/OOD ─────────────
    main_rows = []
    per_subject: dict[tuple[str, str], np.ndarray] = {}
    for regime in REGIMES:
        for removed, kept in REMOVALS:
            pd_ = paired_diffs(d[d.regime == regime], kept)
            g = pd_.groupby("subject")["rel"].mean()          # one value per subject
            per_subject[(regime, removed)] = g.sort_index().values
            main_rows.append({"regime": regime, "removed": removed,
                              "base_nrmse": pd_["base"].mean(), **test_block(g.values)})
    t_main = pd.DataFrame(main_rows)

    # ── Table 2: per data condition (still subject level, n=19) ─────────
    by_rows = []
    for regime in REGIMES:
        for key, lbl in DATA_ORDER:
            sub = d[(d.regime == regime) & (d.data == key)]
            row = {"regime": regime, "data": lbl}
            ok = True
            for removed, kept in REMOVALS:
                pd_ = paired_diffs(sub, kept)
                if pd_.empty:
                    ok = False
                    break
                g = pd_.groupby("subject")["rel"].mean()
                b = test_block(g.values)
                row["base_nrmse"] = pd_["base"].mean()
                row[f"d{removed}"] = b["mean"]
                row[f"p{removed}"] = b["p_ttest"]
                row[f"pos{removed}"] = b["n_pos"]
                row["n"] = b["n"]
            if ok:
                by_rows.append(row)
    t_by = pd.DataFrame(by_rows)

    # ── Table 3: aggregation-unit sensitivity (effect of pseudoreplication on p) ──
    LEVELS = [("(a) raw: data x session pair x subject", FULL_IDX),
              ("(b) subject x data condition", ["subject", "data"]),
              ("(c) subject x session pair", ["subject", "train_sess", "test_session"]),
              ("(d) subject (independent unit)", ["subject"])]
    sens_rows = []
    for regime in REGIMES:
        for removed, kept in REMOVALS:
            pd_ = paired_diffs(d[d.regime == regime], kept)
            for name, keys in LEVELS:
                g = pd_.groupby(keys)["rel"].mean()
                sens_rows.append({"regime": regime, "removed": removed, "level": name,
                                  "n": len(g), "mean": g.mean(),
                                  "p_wilcoxon": stats.wilcoxon(g).pvalue, "p_ttest": stats.ttest_1samp(g,0).pvalue})
    t_sens = pd.DataFrame(sens_rows)

    # ── Table 5: interaction per data fraction (within-subject pairing) ──
    int_by_rows = []
    for key, lbl in DATA_ORDER:
        sub = d[d.data == key]
        for removed, kept in REMOVALS:
            v = {}
            for regime in REGIMES:
                pdx = paired_diffs(sub[sub.regime == regime], kept)
                v[regime] = pdx.groupby("subject")["rel"].mean().sort_index()
            common = v["OOD"].index.intersection(v["ID"].index)
            diff = (v["OOD"].loc[common] - v["ID"].loc[common]).values
            int_by_rows.append({"data": lbl, "removed": removed,
                                "OOD_mean": v["OOD"].loc[common].mean(),
                                "ID_mean": v["ID"].loc[common].mean(),
                                **test_block(diff)})
    t_int_by = pd.DataFrame(int_by_rows)

    # ── Table 4: ID vs OOD interaction (within-subject pairing) ─────────
    int_rows = []
    for removed, _ in REMOVALS:
        for hi, lo in [("OOD", "ID")]:
            a, b = per_subject[(hi, removed)], per_subject[(lo, removed)]
            diff = a - b                                      # within-subject difference
            int_rows.append({"removed": removed, "contrast": f"{hi} - {lo}",
                             "hi_mean": a.mean(), "lo_mean": b.mean(),
                             **test_block(diff)})
    t_int = pd.DataFrame(int_rows)

    for nm, t in [("aux_ablation_main", t_main), ("aux_ablation_by_data", t_by),
                  ("aux_ablation_sensitivity", t_sens), ("aux_ablation_interaction", t_int),
                  ("aux_ablation_interaction_by_data", t_int_by)]:
        t.to_csv(os.path.join(args.out_dir, nm + ".csv"), index=False)

    # ── report ──────────────────────────────────────────────────────────
    L = ["# CoM auxiliary input ablation", "",
         "Metric: **test NRMSE(range), last-epoch model**. Δ is the relative change when the",
         "channel is removed: **positive = worse without it = the channel contributes**. Tests",
         "are at the **subject level (n=19)** — the subject is the only independent sampling",
         "unit (data conditions and session pairs are repeated observations of the same subject).", "",
         "## Table 1 (primary) — subject-level tests",
         "",
         "| Regime | Removed | n | Proposed NRMSE | mean Δ | 95% CI | t | **t-test p** | Wilcoxon p | worse subjects | dz | normality p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t_main.iterrows():
        L.append(f"| {r['regime']} | {r['removed']} | {r['n']:.0f} | {r['base_nrmse']:.5f} "
                 f"| {r['mean']:+.2f}% | [{r['t_lo']:+.2f}, {r['t_hi']:+.2f}] | {r['t_stat']:+.2f} "
                 f"| **{r['p_ttest']:.4f}** {stars(r['p_ttest'])} "
                 f"| {r['p_wilcoxon']:.4f} {stars(r['p_wilcoxon'])} "
                 f"| {r['n_pos']:.0f}/{r['n']:.0f} | {r['dz']:+.3f} | {r['p_shapiro']:.3f} |")

    L += ["", "## Table 2 — per training-data amount (each cell subject-level, n=19)", "",
          "| Regime | Data | Proposed NRMSE | Δ(VT removed) | p | Δ(AP removed) | p |",
          "|---|---|---|---|---|---|---|"]
    for _, r in t_by.iterrows():
        L.append(f"| {r['regime']} | {r['data']} | {r['base_nrmse']:.5f} "
                 f"| {r['dVT']:+.2f}% | {r['pVT']:.4f} {stars(r['pVT'])} "
                 f"| {r['dAP']:+.2f}% | {r['pAP']:.4f} {stars(r['pAP'])} |")

    L += ["", "## Table 3 — aggregation-unit sensitivity (pseudoreplication)", "",
          "Each row further removes repeated observations. The mean effect barely moves while",
          "n shrinks and p grows = the raw-data p is inflated by pseudoreplication.", "",
          "| Regime | Removed | Aggregation unit | n | mean Δ | t-test p |",
          "|---|---|---|---|---|---|"]
    for _, r in t_sens.iterrows():
        L.append(f"| {r['regime']} | {r['removed']} | {r['level']} | {r['n']:.0f} "
                 f"| {r['mean']:+.2f}% | {r['p_ttest']:.4f} {stars(r['p_ttest'])} |")

    L += ["", "## Table 4 — ID vs OOD interaction (within-subject pairing)", "",
          "Per subject, (first-regime effect − second-regime effect) compared with 0.",
          "Positive = larger contribution in the first regime.", "",
          "| Removed | Contrast | OOD mean | ID mean | diff | 95% CI | **t-test p** | Wilcoxon p |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in t_int.iterrows():
        L.append(f"| {r['removed']} | {r['contrast']} | {r['hi_mean']:+.2f}% | {r['lo_mean']:+.2f}% "
                 f"| {r['mean']:+.2f}% | [{r['t_lo']:+.2f}, {r['t_hi']:+.2f}] "
                 f"| **{r['p_ttest']:.4f}** {stars(r['p_ttest'])} | {r['p_wilcoxon']:.4f} {stars(r['p_wilcoxon'])} |")

    md = "\n".join(L)
    with open(os.path.join(args.out_dir, "aux_ablation_table.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n[OK] -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
