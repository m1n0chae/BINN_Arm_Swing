#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[revision] 트랙 A — CoM auxiliary input ablation 결과 표 생성

리뷰 대응: "CoM VT, AP Auxiliary Input 을 각각 하나씩 없애는 ablation"

────────────────────────────────────────────────────────────────────────
지표
    test NRMSE(range).  main.py 가 last.pth(마지막 에폭)를 로드해 저장한
    per-timestep 예측 CSV 에서 stride 단위로 계산한 값.
    학습곡선 최솟값(best_val)은 model selection 이 val 에 들어가므로 쓰지 않는다.

평가 regime 2종  — 기준은 "평가 세션을 학습에서 봤는가"
    ID  : 봤음.  tv_te_ss1/2/3 (한 속도 학습 후 같은 속도)
                 tv_te_ss123   (세 속도 학습 후 그 세 속도)
    OOD : 못 봄. tv_ssX__te_... (한 속도 학습 후 나머지 속도)

    ss1/ss2/ss3 는 보행 속도 조건이다 (평균 stride 시간 1.134 / 1.052 / 0.979 초).

검정
    주 검정  : 대응표본 t-검정 (차이의 정규성 Shapiro-Wilk 로 함께 보고)
    보조     : Wilcoxon signed-rank, 부호검정, 부트스트랩 95% CI

★ 검정 단위 (이 스크립트의 핵심)
    aux 조건은 같은 피험자·같은 fold·같은 seed 로 학습되고 aux 채널만 다르다.
    v2 실행에서 한 data 조건의 aux 3개를 동일 머신에서 돌렸고 재현성 차이가
    0.000%(비트 단위 동일)이므로, 조건 간 차이는 오직 aux 채널에서 온다.
    따라서 대응표본(paired) 검정이 맞다.

    다만 "무엇을 하나의 표본으로 볼 것인가"가 p값을 좌우한다.
    원자료는 (data 5) x (train,test 세션쌍 6) x (피험자 19) 로 530행이지만,
    독립적으로 추출된 단위는 **피험자 19명뿐**이다.
      - 5개 data 조건은 같은 데이터의 부분집합 (독립 아님)
      - 6개 세션쌍은 같은 피험자의 같은 보행 (독립 아님)
    530행을 그대로 쓰면 한 피험자를 최대 30번 세는 유사반복(pseudoreplication)
    이 되어 p값이 실제보다 작게 나온다(anti-conservative).

    -> 주 결과는 **피험자 단위(n=19)**. 하위 단위 결과는 민감도 분석으로 함께 보고.

출력
    <out_dir>/aux_ablation_{main,by_data,sensitivity,interaction}.csv
    <out_dir>/aux_ablation_table.{md,tex}

사용법
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
    "tv_te_ss123": "ss123",          # 3세션 모두 학습 -> 제3의 regime
}
# ID  : 평가 세션을 학습에서 봤음 (tv_te_ss1/2/3, tv_te_ss123)
# OOD : 평가 세션을 학습에서 못 봄 (tv_ssX__te_...)
REGIMES = ["ID", "OOD"]
FULL_IDX = ["data", "train_sess", "test_session", "subject"]
# (제거한 채널, 남긴 조건명).  'AP' 조건 = acc_x 만 = VT 를 제거한 것
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
    """(data, 세션쌍, 피험자) 마다 [Proposed 대비 상대변화 %] 를 계산."""
    w = sub.pivot_table(index=FULL_IDX, columns="aux", values="test_nrmse").dropna()
    if not {"APVT", "AP", "VT"} <= set(w.columns):
        return pd.DataFrame()
    out = w.reset_index()[FULL_IDX].copy()
    out["base"] = w["APVT"].values
    out["alt"] = w[kept].values
    out["rel"] = (w[kept].values - w["APVT"].values) / w["APVT"].values * 100.0
    return out


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    """평균의 부트스트랩 95% CI (표본을 복원추출)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def test_block(rel_by_subject: np.ndarray) -> dict:
    """피험자 단위 벡터 하나에 대한 검정 묶음."""
    n = len(rel_by_subject)
    mean = float(rel_by_subject.mean())
    med = float(np.median(rel_by_subject))
    n_pos = int((rel_by_subject > 0).sum())
    p_w = float(stats.wilcoxon(rel_by_subject).pvalue) if n > 5 else np.nan
    p_s = float(stats.binomtest(n_pos, n, 0.5).pvalue)
    dz = mean / float(rel_by_subject.std(ddof=1))
    lo, hi = boot_ci(rel_by_subject)
    tt = stats.ttest_1samp(rel_by_subject, 0.0)          # 주 검정
    ci = tt.confidence_interval()
    p_norm = float(stats.shapiro(rel_by_subject).pvalue)  # 차이의 정규성
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

    # ── Table 1 (주 결과): 피험자 단위, ID/OOD 별 ───────────────────────
    main_rows = []
    per_subject: dict[tuple[str, str], np.ndarray] = {}
    for regime in REGIMES:
        for removed, kept in REMOVALS:
            pd_ = paired_diffs(d[d.regime == regime], kept)
            g = pd_.groupby("subject")["rel"].mean()          # 피험자당 1값으로 축약
            per_subject[(regime, removed)] = g.sort_index().values
            main_rows.append({"regime": regime, "removed": removed,
                              "base_nrmse": pd_["base"].mean(), **test_block(g.values)})
    t_main = pd.DataFrame(main_rows)

    # ── Table 2: 데이터 조건별 (여전히 피험자 단위 n=19) ────────────────
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

    # ── Table 3: 집계 단위 민감도 (유사반복이 p에 미치는 영향) ──────────
    LEVELS = [("(a) 원자료: data x 세션쌍 x 피험자", FULL_IDX),
              ("(b) 피험자 x 데이터조건", ["subject", "data"]),
              ("(c) 피험자 x 세션쌍", ["subject", "train_sess", "test_session"]),
              ("(d) 피험자 (독립 단위)", ["subject"])]
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

    # ── Table 5: 데이터%별 상호작용 (피험자 내 대응) ────────────────────
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

    # ── Table 4: ID vs OOD 상호작용 (피험자 내 대응) ────────────────────
    int_rows = []
    for removed, _ in REMOVALS:
        for hi, lo in [("OOD", "ID")]:
            a, b = per_subject[(hi, removed)], per_subject[(lo, removed)]
            diff = a - b                                      # 같은 피험자 내 차이
            int_rows.append({"removed": removed, "contrast": f"{hi} - {lo}",
                             "hi_mean": a.mean(), "lo_mean": b.mean(),
                             **test_block(diff)})
    t_int = pd.DataFrame(int_rows)

    for nm, t in [("aux_ablation_main", t_main), ("aux_ablation_by_data", t_by),
                  ("aux_ablation_sensitivity", t_sens), ("aux_ablation_interaction", t_int),
                  ("aux_ablation_interaction_by_data", t_int_by)]:
        t.to_csv(os.path.join(args.out_dir, nm + ".csv"), index=False)

    # ── 출력 ────────────────────────────────────────────────────────────
    L = ["# CoM auxiliary input ablation", "",
         "지표: **test NRMSE(range), last-epoch 모델**. Δ는 해당 채널을 제거했을 때의 상대 변화율로,",
         "**양수 = 제거 시 악화 = 그 채널이 기여함**. 검정 단위는 **피험자(n=19)** — 피험자가",
         "유일한 독립 추출 단위이기 때문 (데이터 조건과 세션쌍은 같은 피험자의 중복 관측).", "",
         "## Table 1 (주 결과) — 피험자 단위 검정",
         "",
         "| Regime | 제거 채널 | n | Proposed NRMSE | 평균 Δ | 95% CI | t | **t-검정 p** | Wilcoxon p | 악화 피험자 | dz | 정규성 p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t_main.iterrows():
        L.append(f"| {r['regime']} | {r['removed']} | {r['n']:.0f} | {r['base_nrmse']:.5f} "
                 f"| {r['mean']:+.2f}% | [{r['t_lo']:+.2f}, {r['t_hi']:+.2f}] | {r['t_stat']:+.2f} "
                 f"| **{r['p_ttest']:.4f}** {stars(r['p_ttest'])} "
                 f"| {r['p_wilcoxon']:.4f} {stars(r['p_wilcoxon'])} "
                 f"| {r['n_pos']:.0f}/{r['n']:.0f} | {r['dz']:+.3f} | {r['p_shapiro']:.3f} |")

    L += ["", "## Table 2 — 데이터량별 (각 셀도 피험자 단위 n=19)", "",
          "| Regime | Data | Proposed NRMSE | Δ(VT 제거) | p | Δ(AP 제거) | p |",
          "|---|---|---|---|---|---|---|"]
    for _, r in t_by.iterrows():
        L.append(f"| {r['regime']} | {r['data']} | {r['base_nrmse']:.5f} "
                 f"| {r['dVT']:+.2f}% | {r['pVT']:.4f} {stars(r['pVT'])} "
                 f"| {r['dAP']:+.2f}% | {r['pAP']:.4f} {stars(r['pAP'])} |")

    L += ["", "## Table 3 — 집계 단위 민감도 (유사반복 영향)", "",
          "아래로 갈수록 중복 관측을 걷어낸 것. 평균 효과는 거의 그대로인데 n 이 줄면서",
          "p 가 커진다 = 원자료 p 는 유사반복으로 부풀려진 값이다.", "",
          "| Regime | 제거 | 집계 단위 | n | 평균 Δ | t-검정 p |",
          "|---|---|---|---|---|---|"]
    for _, r in t_sens.iterrows():
        L.append(f"| {r['regime']} | {r['removed']} | {r['level']} | {r['n']:.0f} "
                 f"| {r['mean']:+.2f}% | {r['p_ttest']:.4f} {stars(r['p_ttest'])} |")

    L += ["", "## Table 4 — ID vs OOD 상호작용 (피험자 내 대응)", "",
          "피험자마다 (앞 regime 효과 − 뒤 regime 효과) 를 구해 0 과 비교. 양수 = 앞쪽에서 더 크게 기여.", "",
          "| 제거 채널 | 대비 | OOD 평균 | ID 평균 | 차이 | 95% CI | **t-검정 p** | Wilcoxon p |",
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
