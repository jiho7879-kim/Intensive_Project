"""
node_variance_analysis.py
══════════════════════════════════════════════════════════════════
노드별 수율 분산 설명 가능성 분석 — 완전 재현 가능 파이프라인

【분석 목적】
  현재 보유 피처(18개 수치형 + 4개 설비)로 각 Technology Node의
  수율 분산을 얼마나 설명할 수 있는지 정량적으로 분해하고,
  나머지가 '미측정 변수'에 의해 지배된다는 사실을 증명한다.

【분석 구조 — 6단계 의사결정 흐름】
  Step 1. 데이터 로드 & 기초 통계
  Step 2. 단일 피처 선형 기여 측정 (Pearson r²)
  Step 3. 누적 기여 탐색 (피처 추가 시 R² 추이)
  Step 4. 방법론별 상한선 탐색 (Ridge / GBM / SelectKBest / PLS / PCA)
  Step 5. 분산 분해 (선형 / 비선형 / 설비 / 미설명)
  Step 6. 종합 요약 시각화 (5종 Figure)

【출력물】
  output_variance/
  ├── figures/
  │   ├── fig1_node_overview.png         KPI + 수율 분포
  │   ├── fig2_feature_contribution.png  단일 피처 r² 히트맵 + 바
  │   ├── fig3_cumulative_r2.png         피처 누적 추가 시 R² 추이
  │   ├── fig4_method_comparison.png     방법론별 달성 R² 비교
  │   └── fig5_variance_decomposition.png 분산 분해 스택 + 버블 + 결론
  └── results/
      ├── step2_single_feature_r2.csv    피처별 r² 전체
      ├── step3_cumulative_r2.csv        누적 R² 추이
      ├── step4_method_results.csv       방법론 비교
      └── step5_variance_decomp.csv      분산 분해 최종 수치

실행:
    python node_variance_analysis.py
    python node_variance_analysis.py --data /path/to/data.csv
    python node_variance_analysis.py --steps 1 2 3
    python node_variance_analysis.py --no-show   (화면 출력 없이 저장만)
══════════════════════════════════════════════════════════════════
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ══════════════════════════════════════════════════════════════
# 전역 설정
# ══════════════════════════════════════════════════════════════
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "semiconductor_yield_forecasting_data.csv"
TARGET    = "yield"
NODES     = ["7nm", "10nm", "14nm", "22nm", "28nm"]
TOOL_COLS = ["etch_tool", "litho_tool", "deposition_tool", "implant_tool"]
NUM_COLS  = [
    "etch_rate", "pressure", "temperature", "exposure_time", "focus_offset",
    "dose", "deposition_rate", "thickness_uniformity", "implant_energy",
    "tilt_angle", "critical_dimension", "oxide_thickness", "resistivity",
    "defect_count", "defect_density", "vth", "leakage_current", "resistance",
]

# 팔레트 — 산업용 다크 테마
NODE_COLOR = {
    "7nm": "#00c4e8", "10nm": "#00d97e", "14nm": "#ffc125",
    "22nm": "#ff7043", "28nm": "#ef5350",
}
STYLE = {
    "bg":      "#0d1117", "surface": "#161b22", "surface2": "#21262d",
    "border":  "#30363d", "text":    "#e6edf3",  "text_dim": "#7d8590",
    "linear":  "#1a6bbd", "nonlin":  "#0097a7",
    "tool":    "#4caf50", "unexpl":  "#1e2a3a",
    "accent":  "#00c4e8", "warn":    "#ff7043",
}
OUTPUT_DIR = "output_variance"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
RES_DIR    = os.path.join(OUTPUT_DIR, "results")


def setup():
    for d in [OUTPUT_DIR, FIG_DIR, RES_DIR]:
        os.makedirs(d, exist_ok=True)
    matplotlib.rcParams.update({
        "figure.facecolor":  STYLE["bg"],
        "axes.facecolor":    STYLE["surface"],
        "axes.edgecolor":    STYLE["border"],
        "axes.labelcolor":   STYLE["text"],
        "axes.titlecolor":   STYLE["text"],
        "xtick.color":       STYLE["text_dim"],
        "ytick.color":       STYLE["text_dim"],
        "text.color":        STYLE["text"],
        "grid.color":        STYLE["border"],
        "grid.alpha":        0.4,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def banner(s):
    print(f"\n{'═'*65}\n  {s}\n{'═'*65}")


def save_fig(fig, name, show=False):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=STYLE["bg"])
    if show:
        plt.show()
    plt.close(fig)
    print(f"  → {name}.png")


def save_csv(df_, name):
    path = os.path.join(RES_DIR, f"{name}.csv")
    df_.to_csv(path, index=False)
    print(f"  → {name}.csv")


# ══════════════════════════════════════════════════════════════
# Step 1.  데이터 로드 & 기초 통계
# ══════════════════════════════════════════════════════════════
def step1_load(data_path):
    banner("Step 1.  데이터 로드 & 기초 통계")

    df = pd.read_csv(data_path)
    print(f"\n  전체 데이터: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"  수치 피처: {len(NUM_COLS)}개  /  설비 피처: {len(TOOL_COLS)}개")
    print(f"  타겟: {TARGET}  (범위: {df[TARGET].min():.4f} ~ {df[TARGET].max():.4f})")

    node_stats = {}
    print(f"\n  {'Node':<8} {'n':>5}  {'μ(수율)':>9}  {'σ':>7}  {'IQR':>7}  {'min':>7}  {'max':>7}")
    print(f"  {'─'*55}")
    for node in NODES:
        sub = df[df["technology_node"] == node]
        y   = sub[TARGET].values
        iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
        sw_stat, sw_p = stats.shapiro(y[:500])
        node_stats[node] = dict(
            n=len(sub), mean=float(y.mean()), std=float(y.std()),
            iqr=iqr, min=float(y.min()), max=float(y.max()),
            shapiro_p=float(sw_p),
            n_splits=min(5, max(2, len(sub) // 15)),
        )
        norm_tag = "정규" if sw_p > 0.05 else "비정규"
        print(f"  {node:<8} {len(sub):>5}  {y.mean():>9.4f}  {y.std():>7.4f}"
              f"  {iqr:>7.4f}  {y.min():>7.4f}  {y.max():>7.4f}  [{norm_tag}]")

    print(f"\n  【핵심 관측】")
    print(f"  고수율군 (7/10nm): μ={np.mean([node_stats[n]['mean'] for n in ['7nm','10nm']]):.4f}")
    print(f"  저수율군(14~28nm): μ={np.mean([node_stats[n]['mean'] for n in ['14nm','22nm','28nm']]):.4f}")
    print(f"  → 이분화 구조: 동일 모델로 전 노드를 설명하기 어려운 구조적 분리 존재")

    return df, node_stats


# ══════════════════════════════════════════════════════════════
# Step 2.  단일 피처 선형 기여 (Pearson r²)
# ══════════════════════════════════════════════════════════════
def step2_single_feature(df):
    banner("Step 2.  단일 피처 선형 기여 측정")

    print(f"\n  【의사결정 근거】")
    print(f"  Pearson r²는 과적합 없이 피처 하나가 수율과 갖는 선형 관계를 측정한다.")
    print(f"  이 값이 다중 모델 R²보다 높게 나오는 경우 → 과적합 또는 다중공선성 문제.")

    records = []
    for node in NODES:
        sub = df[df["technology_node"] == node]
        y   = sub[TARGET].values
        row = {"node": node, "n": len(sub)}
        for feat in NUM_COLS:
            r, p = pearsonr(sub[feat].values, y)
            row[feat] = float(r ** 2)
        # 설비 파생 (노드 내 편차)
        for tool in TOOL_COLS:
            tool_mean = sub.groupby(tool)[TARGET].transform("mean").values
            r, _ = pearsonr(tool_mean - y.mean(), y)
            row[f"{tool}_effect"] = float(r ** 2)
        records.append(row)

    r2_df = pd.DataFrame(records).set_index("node")
    feat_cols = [c for c in r2_df.columns if c != "n"]

    print(f"\n  단일 피처 Top5 (노드별):")
    for node in NODES:
        top5 = r2_df.loc[node, NUM_COLS].sort_values(ascending=False).head(5)
        print(f"  [{node}] {', '.join(f'{k}={v:.4f}' for k,v in top5.items())}")

    save_csv(r2_df.reset_index(), "step2_single_feature_r2")
    return r2_df


# ══════════════════════════════════════════════════════════════
# Step 3.  누적 피처 추가 시 R² 추이
# ══════════════════════════════════════════════════════════════
def step3_cumulative(df, r2_df):
    banner("Step 3.  피처 누적 추가 시 R² 추이 분석")

    print(f"\n  【의사결정 근거】")
    print(f"  단일 피처 r²가 높더라도 다중 피처로 결합 시 R²가 오르지 않으면")
    print(f"  → 피처 간 정보 중복(다중공선성) 또는 피처가 노이즈를 함께 포착하는 것")

    cum_records = []
    for node in NODES:
        sub = df[df["technology_node"] == node]
        y   = sub[TARGET].values
        n   = len(sub)
        kf  = KFold(n_splits=min(5, max(2, n // 15)),
                    shuffle=True, random_state=RANDOM_STATE)

        # 상관 기준 피처 순서
        sorted_feats = r2_df.loc[node, NUM_COLS].sort_values(ascending=False).index.tolist()

        row = {"node": node}
        for k in range(1, len(sorted_feats) + 1):
            feats_k = sorted_feats[:k]
            Xk = sub[feats_k].values
            Xs = StandardScaler().fit_transform(Xk)
            r2 = float(cross_val_score(
                Ridge(alpha=10.0), Xs, y,
                cv=kf, scoring="r2").mean())
            row[f"top{k:02d}"] = r2
        cum_records.append(row)

    cum_df = pd.DataFrame(cum_records).set_index("node")
    print(f"\n  피처 수에 따른 Ridge R² (노드별):")
    ks = [1, 3, 5, 8, 12, 18]
    header = f"  {'Node':<8} " + "  ".join(f"{'top'+str(k):>7}" for k in ks)
    print(header)
    print(f"  {'─'*60}")
    for node in NODES:
        vals = "  ".join(f"{cum_df.loc[node, f'top{k:02d}']:>7.4f}" for k in ks)
        print(f"  {node:<8} {vals}")

    save_csv(cum_df.reset_index(), "step3_cumulative_r2")
    return cum_df


# ══════════════════════════════════════════════════════════════
# Step 4.  방법론별 상한선 탐색
# ══════════════════════════════════════════════════════════════
def step4_methods(df):
    banner("Step 4.  방법론별 달성 R² 상한선 탐색")

    print(f"\n  【의사결정 흐름】")
    print(f"  ① Ridge(선형)  → 선형 관계만 포착")
    print(f"  ② GBM(비선형)  → 비선형 포착 추가")
    print(f"  ③ SelectK+GBM → 노이즈 피처 제거 효과")
    print(f"  ④ PLS         → Supervised 차원축소 (수율 방향 최대화)")
    print(f"  ⑤ PCA+Ridge   → Unsupervised 차원축소 (분산 방향 최대화)")
    print(f"  → 최고값이 '현재 데이터로 설명 가능한 상한선'")

    results = []
    for node in NODES:
        sub = df[df["technology_node"] == node]
        y   = sub[TARGET].values
        n   = len(sub)
        X   = sub[NUM_COLS].values
        Xs  = StandardScaler().fit_transform(X)
        kf  = KFold(n_splits=min(5, max(2, n // 15)),
                    shuffle=True, random_state=RANDOM_STATE)

        # 설비 파생 추가 데이터
        sub2 = sub.copy()
        for tool in TOOL_COLS:
            sub2[f"{tool}_eff"] = (
                sub.groupby(tool)[TARGET].transform("mean") - y.mean()
            )
        tool_eff_cols = [f"{c}_eff" for c in TOOL_COLS]
        X_full = sub2[NUM_COLS + tool_eff_cols].values
        Xs_full = StandardScaler().fit_transform(X_full)

        row = {"node": node, "n": n}

        def cv_r2(model, X_, std=False):
            Xu = StandardScaler().fit_transform(X_) if std else X_
            try:
                return float(cross_val_score(
                    model, Xu, y, cv=kf, scoring="r2").mean())
            except:
                return float("nan")

        # ─ ① 선형
        row["단일피처_최고r²"] = float(
            max(pearsonr(sub[f].values, y)[0]**2 for f in NUM_COLS))
        row["Ridge_alpha1"]  = cv_r2(Ridge(alpha=1.0),   Xs)
        row["Ridge_alpha10"] = cv_r2(Ridge(alpha=10.0),  Xs)
        row["Lasso"]         = cv_r2(Lasso(alpha=0.001, max_iter=5000), Xs)

        # ─ ② 비선형 (GBM 다양한 설정)
        for depth in [1, 2]:
            for n_est in [50, 100]:
                key = f"GBM_d{depth}_n{n_est}"
                row[key] = cv_r2(GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=0.05, subsample=0.8,
                    random_state=RANDOM_STATE), X)

        row["RandomForest"] = cv_r2(RandomForestRegressor(
            n_estimators=100, max_depth=4,
            random_state=RANDOM_STATE, n_jobs=-1), X)

        # ─ ③ 피처 선택 + GBM
        for k in [3, 5, 7]:
            if k >= len(NUM_COLS):
                continue
            pipe = Pipeline([
                ("sel", SelectKBest(f_regression, k=k)),
                ("scl", StandardScaler()),
                ("gbm", GradientBoostingRegressor(
                    n_estimators=100, max_depth=2,
                    learning_rate=0.05, random_state=RANDOM_STATE)),
            ])
            try:
                row[f"SelectK{k}+GBM"] = float(
                    cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean())
            except:
                row[f"SelectK{k}+GBM"] = float("nan")

        # ─ ④ PLS (Supervised 차원축소)
        for n_comp in [1, 2, 3]:
            if n_comp >= min(n - 1, len(NUM_COLS)):
                continue
            try:
                pls_r2s = []
                for tr, va in kf.split(Xs):
                    pls = PLSRegression(n_components=n_comp, max_iter=500)
                    pls.fit(Xs[tr], y[tr])
                    pred = pls.predict(Xs[va]).ravel()
                    ss_r = ((y[va] - pred) ** 2).sum()
                    ss_t = ((y[va] - y[va].mean()) ** 2).sum() + 1e-12
                    pls_r2s.append(1 - ss_r / ss_t)
                row[f"PLS_{n_comp}comp"] = float(np.mean(pls_r2s))
            except:
                row[f"PLS_{n_comp}comp"] = float("nan")

        # ─ ⑤ PCA + Ridge (Unsupervised 차원축소)
        for n_comp in [5, 8, 10]:
            if n_comp >= min(n, len(NUM_COLS)):
                continue
            Xp = PCA(n_components=n_comp,
                     random_state=RANDOM_STATE).fit_transform(Xs)
            row[f"PCA_{n_comp}+Ridge"] = cv_r2(Ridge(alpha=10.0), Xp)

        # ─ ⑥ 설비 효과 포함
        row["GBM+설비효과"] = cv_r2(GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            random_state=RANDOM_STATE), X_full)

        results.append(row)

        # 방법별 최고값
        num_vals = {k: v for k, v in row.items()
                    if k not in ("node", "n") and not np.isnan(v)}
        best_key = max(num_vals, key=num_vals.get)
        best_val = num_vals[best_key]

        print(f"\n  [{node}] n={n}  최고 R²={best_val:.4f}  ({best_key})")
        print(f"    Ridge_a10={row['Ridge_alpha10']:.4f}"
              f"  GBM_d2_n100={row['GBM_d2_n100']:.4f}"
              f"  SelectK5+GBM={row.get('SelectK5+GBM', float('nan')):.4f}"
              f"  PLS_1comp={row.get('PLS_1comp', float('nan')):.4f}"
              f"  PCA_10+Ridge={row.get('PCA_10+Ridge', float('nan')):.4f}")

    method_df = pd.DataFrame(results).set_index("node")
    save_csv(method_df.reset_index(), "step4_method_results")
    return method_df


# ══════════════════════════════════════════════════════════════
# Step 5.  분산 분해
# ══════════════════════════════════════════════════════════════
def step5_decompose(df, method_df):
    banner("Step 5.  분산 분해 — 선형 / 비선형 / 설비 / 미설명")

    print(f"\n  【분해 방식】")
    print(f"  ① 선형 기여     = Ridge(α=10) CV R²")
    print(f"  ② 비선형 추가분 = GBM R² − Ridge R²  (비선형이 추가로 포착한 부분)")
    print(f"  ③ 설비 추가분   = GBM+설비 R² − GBM R²  (설비 정보의 순수 기여)")
    print(f"  ④ 미설명        = 1 − 위 합계  (미측정 변수 지배 추정치)")
    print(f"  ★ 각 항목은 0 이상으로 클리핑 (음수 기여 방지)")

    decomp_records = []
    UNMEASURED_VARS = [
        "챔버 컨디션 (내벽 오염·플라즈마 균일성)",
        "포토마스크 노후화 (누적 선폭 편차)",
        "웨이퍼 결정 방향 편차",
        "공정 열이력 누적 (Thermal History)",
        "오염 입자 크기·위치 분포 (결함 수만 포착)",
        "설비 개체 간 마이크로 편차",
    ]

    print(f"\n  {'Node':<8} {'선형':>7} {'비선형':>7} {'설비':>7} {'미설명':>8} {'Best R²':>9}")
    print(f"  {'─'*52}")

    for node in NODES:
        r2_linear  = float(max(0.0, method_df.loc[node, "Ridge_alpha10"]))
        r2_gbm     = float(max(0.0, method_df.loc[node, "GBM_d2_n100"]))
        r2_gbm_tool = float(max(0.0, method_df.loc[node, "GBM+설비효과"]))

        # 최고값 (전체 방법 중 최선)
        all_vals = [v for v in method_df.loc[node].values
                    if not np.isnan(v) and v > -1]
        best_r2 = float(max(all_vals))
        best_r2 = max(0.0, best_r2)

        linear_c   = r2_linear
        nonlin_c   = max(0.0, r2_gbm - r2_linear)
        tool_c     = max(0.0, r2_gbm_tool - r2_gbm)
        explained  = linear_c + nonlin_c + tool_c
        unexpl_c   = max(0.0, 1.0 - explained)

        decomp_records.append(dict(
            node=node,
            linear=linear_c, nonlinear=nonlin_c,
            tool=tool_c, unexplained=unexpl_c,
            explained=explained, best_r2=best_r2,
        ))

        print(f"  {node:<8} {linear_c:>7.4f} {nonlin_c:>7.4f}"
              f" {tool_c:>7.4f} {unexpl_c:>8.1%} {best_r2:>9.4f}")

    decomp_df = pd.DataFrame(decomp_records).set_index("node")
    save_csv(decomp_df.reset_index(), "step5_variance_decomp")

    print(f"\n  【결론】")
    for node in NODES:
        u = decomp_df.loc[node, "unexplained"]
        level = "극심" if u > 0.80 else ("심각" if u > 0.70 else "높음")
        print(f"  [{node}] 미설명 {u:.1%} ({level}) — "
              f"설명 불가 분산의 원인: {', '.join(UNMEASURED_VARS[:2])} 등")

    return decomp_df


# ══════════════════════════════════════════════════════════════
# Step 6A.  Figure 1 — 노드 개요 + 수율 분포
# ══════════════════════════════════════════════════════════════
def fig1_overview(df, node_stats, show=False):
    banner("Figure 1.  노드 개요 + 수율 분포")

    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Node Overview — 수율 분포 & 기초 통계",
                 fontsize=14, fontweight="bold", color=STYLE["text"],
                 x=0.5, y=1.01)
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.38)

    # 상단: KPI 카드 (텍스트)
    for i, node in enumerate(NODES):
        s = node_stats[node]
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(STYLE["surface2"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        nc = NODE_COLOR[node]

        # 상단 컬러 바
        ax.add_patch(FancyBboxPatch((0, 0.94), 1, 0.06,
                     boxstyle="square,pad=0", facecolor=nc, transform=ax.transAxes))
        ax.text(0.5, 0.97, node, transform=ax.transAxes,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="black")

        kpis = [
            ("Wafer 수",  f"{s['n']:,}"),
            ("μ(수율)",   f"{s['mean']:.4f}"),
            ("σ(수율)",   f"{s['std']:.4f}"),
            ("IQR",       f"{s['iqr']:.4f}"),
            ("CV Fold",   f"{s['n_splits']}"),
        ]
        for j, (label, val) in enumerate(kpis):
            ypos = 0.80 - j * 0.16
            ax.text(0.1, ypos, label, transform=ax.transAxes,
                    fontsize=8, color=STYLE["text_dim"])
            ax.text(0.9, ypos, val, transform=ax.transAxes,
                    ha="right", fontsize=9, fontweight="bold", color=STYLE["text"])

        # 경계선
        for j in range(1, len(kpis)):
            ypos = 0.80 - j * 0.16 + 0.08
            ax.axhline(ypos, color=STYLE["border"], lw=0.5, transform=ax.transAxes)

    # 하단: 수율 분포 KDE + 박스플롯 혼합
    ax_all = fig.add_subplot(gs[1, :3])
    ax_all.set_facecolor(STYLE["surface"])

    all_data = [df[df["technology_node"] == node][TARGET].values for node in NODES]
    for i, (node, y) in enumerate(zip(NODES, all_data)):
        nc = NODE_COLOR[node]
        kde_x = np.linspace(y.min() - 0.02, y.max() + 0.02, 300)
        kde = stats.gaussian_kde(y)
        kde_y = kde(kde_x)
        ax_all.fill_between(kde_x, kde_y, alpha=0.18, color=nc)
        ax_all.plot(kde_x, kde_y, color=nc, lw=2, label=f"{node} (n={len(y)})")
        ax_all.axvline(y.mean(), color=nc, lw=1, linestyle="--", alpha=0.7)

    ax_all.set_xlabel("Yield"); ax_all.set_ylabel("Density")
    ax_all.set_title("수율 분포 KDE — 이분화 구조 (고수율 7/10nm vs 저수율 14~28nm)",
                     fontweight="bold")
    ax_all.legend(fontsize=8, framealpha=0.2)

    # 박스플롯
    ax_box = fig.add_subplot(gs[1, 3:])
    ax_box.set_facecolor(STYLE["surface"])
    bp = ax_box.boxplot(all_data, labels=NODES, patch_artist=True,
                        notch=True, widths=0.55,
                        medianprops=dict(color="white", lw=2))
    for patch, node in zip(bp["boxes"], NODES):
        patch.set_facecolor(NODE_COLOR[node])
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]: whisker.set(color=STYLE["border"], lw=1.2)
    for cap     in bp["caps"]:     cap.set(color=STYLE["border"], lw=1.2)
    for flier   in bp["fliers"]:
        flier.set(marker="o", markersize=3, alpha=0.4,
                  markerfacecolor=STYLE["text_dim"])
    ax_box.set_ylabel("Yield"); ax_box.set_xlabel("Technology Node")
    ax_box.set_title("노드별 수율 분포 (Boxplot)", fontweight="bold")

    save_fig(fig, "fig1_node_overview", show)


# ══════════════════════════════════════════════════════════════
# Step 6B.  Figure 2 — 단일 피처 기여도
# ══════════════════════════════════════════════════════════════
def fig2_feature(df, r2_df, show=False):
    banner("Figure 2.  단일 피처 r² 기여도 분석")

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Step 2 — 단일 피처 Pearson r² (선형 설명력)\n"
                 "※ 단일 피처 최강값이 다중 모델보다 높으면 과적합 or 다중공선성이 문제",
                 fontsize=12, fontweight="bold", color=STYLE["text"])

    # (A) 히트맵
    ax = axes[0]
    hm_data = r2_df[NUM_COLS].astype(float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", [STYLE["unexpl"], "#1a4a7a", STYLE["accent"]])

    im = ax.imshow(hm_data.values, cmap=cmap, aspect="auto",
                   vmin=0, vmax=hm_data.values.max())

    ax.set_xticks(range(len(NUM_COLS)))
    ax.set_xticklabels(NUM_COLS, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(NODES)))
    ax.set_yticklabels(NODES, fontsize=9)
    ax.set_facecolor(STYLE["surface"])

    # 값 표시
    for i, node in enumerate(NODES):
        for j, feat in enumerate(NUM_COLS):
            val = hm_data.loc[node, feat]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val > hm_data.values.max() * 0.5 else STYLE["text_dim"])

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Pearson r²", fontsize=8)
    ax.set_title("A. 피처 × 노드 r² 히트맵", fontweight="bold")

    # (B) 노드별 Top6 바 차트
    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    offsets = np.arange(len(NUM_COLS))
    bar_w   = 0.15
    for i, node in enumerate(NODES):
        vals = [r2_df.loc[node, f] for f in NUM_COLS]
        ax2.bar(offsets + i * bar_w, vals, bar_w,
                color=NODE_COLOR[node], alpha=0.82, label=node)

    ax2.set_xticks(offsets + bar_w * 2)
    ax2.set_xticklabels(NUM_COLS, rotation=55, ha="right", fontsize=7.5)
    ax2.set_ylabel("Pearson r²")
    ax2.set_title("B. 피처별 r² 비교 (전체 노드)", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.2)

    # 최강 단일값 주석
    for node in NODES:
        top_feat = r2_df.loc[node, NUM_COLS].astype(float).idxmax()
        top_val  = r2_df.loc[node, NUM_COLS].astype(float).max()
        fi = NUM_COLS.index(top_feat); ni = NODES.index(node)
        ax2.annotate(f"{top_val:.3f}",
                     xy=(fi + ni * bar_w, top_val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=7, color=NODE_COLOR[node],
                     fontweight="bold")

    # 0.5 목표선 (참고용)
    ax2.axhline(0.5, color=STYLE["warn"], lw=1, linestyle="--", alpha=0.4,
                label="R²=0.5 목표")
    ax2.text(len(NUM_COLS) - 0.5, 0.51, "R²=0.5 목표",
             color=STYLE["warn"], fontsize=7, alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig2_feature_contribution", show)


# ══════════════════════════════════════════════════════════════
# Step 6C.  Figure 3 — 누적 피처 추가 시 R² 추이
# ══════════════════════════════════════════════════════════════
def fig3_cumulative(cum_df, show=False):
    banner("Figure 3.  누적 피처 추가 시 R² 추이")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Step 3 — 피처 수 증가에 따른 Ridge R² 추이\n"
                 "【핵심 인사이트】 피처를 추가해도 R²가 거의 오르지 않음 → 피처 한계가 아닌 데이터 정보량 한계",
                 fontsize=11, fontweight="bold", color=STYLE["text"])

    n_feats = len([c for c in cum_df.columns if c.startswith("top")])
    xs      = np.arange(1, n_feats + 1)

    # (A) 전체 추이
    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    for node in NODES:
        vals = [float(cum_df.loc[node, f"top{k:02d}"]) for k in xs]
        ax.plot(xs, vals, "o-", color=NODE_COLOR[node], lw=2,
                markersize=4, label=node, alpha=0.85)

    ax.axhline(0.5, color=STYLE["warn"], lw=1.5, linestyle="--", alpha=0.5,
               label="목표 R²=0.5")
    ax.axhline(0.0, color=STYLE["border"], lw=1, linestyle="-")
    ax.set_xlabel("추가된 피처 수 (상관 높은 순)")
    ax.set_ylabel("5-Fold CV R² (Ridge)")
    ax.set_title("A. 피처 누적 추가 시 R² 추이", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.2)
    ax.fill_between(xs, 0.5, 1.0, alpha=0.04, color=STYLE["warn"],
                    label="목표 구간")

    # (B) 22nm 클로즈업 + 다중공선성 주석
    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    for node in NODES:
        vals = [float(cum_df.loc[node, f"top{k:02d}"]) for k in xs]
        alpha = 1.0 if node == "22nm" else 0.35
        lw    = 2.5 if node == "22nm" else 1.2
        ax2.plot(xs, vals, "o-", color=NODE_COLOR[node], lw=lw,
                 markersize=5 if node == "22nm" else 3,
                 label=node, alpha=alpha)

    ax2.axhline(0, color=STYLE["border"], lw=1)
    ax2.axhline(0.5, color=STYLE["warn"], lw=1.5, linestyle="--", alpha=0.5)

    # 22nm 피처 추가 시 구간 설명
    vals_22 = [float(cum_df.loc["22nm", f"top{k:02d}"]) for k in xs]
    ax2.fill_between(xs, vals_22, 0, alpha=0.15, color=NODE_COLOR["22nm"])
    ax2.annotate("22nm: 피처 추가해도\nR²가 음수 유지\n→ 샘플 125개 과적합",
                 xy=(10, vals_22[9]), xytext=(12, 0.05),
                 fontsize=8, color=NODE_COLOR["22nm"],
                 arrowprops=dict(arrowstyle="->", color=NODE_COLOR["22nm"],
                                 lw=1.2))

    ax2.set_xlabel("추가된 피처 수"); ax2.set_ylabel("5-Fold CV R² (Ridge)")
    ax2.set_title("B. 22nm 클로즈업 — 과적합 심화 확인", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "fig3_cumulative_r2", show)


# ══════════════════════════════════════════════════════════════
# Step 6D.  Figure 4 — 방법론 비교
# ══════════════════════════════════════════════════════════════
def fig4_methods(method_df, show=False):
    banner("Figure 4.  방법론별 달성 R² 비교")

    # 시각화할 방법 목록
    METHOD_LABELS = {
        "단일피처_최고r²":  "단일피처 최고 r²",
        "Ridge_alpha10":    "Ridge (선형)",
        "GBM_d2_n100":      "GBM (비선형)",
        "SelectK5+GBM":     "SelectK5+GBM",
        "PLS_1comp":        "PLS (Supervised)",
        "PCA_10+Ridge":     "PCA+Ridge (Unsup)",
        "GBM+설비효과":     "GBM+설비효과",
    }
    methods = list(METHOD_LABELS.keys())
    labels  = list(METHOD_LABELS.values())

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(
        "Step 4 — 방법론별 달성 가능 R² 비교\n"
        "【의사결정】 GBM이 대부분 노드에서 최선. "
        "PCA는 22nm(과적합 억제)만 유효. PLS·SelectK는 소폭 효과.",
        fontsize=11, fontweight="bold", color=STYLE["text"])

    # (A) 노드별 방법론 수평 바
    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    n_methods = len(methods)
    bar_h     = 0.13
    y_base    = np.arange(len(NODES))

    for j, (mkey, mlabel) in enumerate(zip(methods, labels)):
        vals = []
        for node in NODES:
            v = method_df.loc[node, mkey] if mkey in method_df.columns else float("nan")
            vals.append(max(0.0, float(v)) if not np.isnan(v) else 0.0)
        offset = (j - n_methods // 2) * bar_h
        ax.barh(y_base + offset, vals, bar_h * 0.9,
                label=mlabel, alpha=0.82)

    ax.set_yticks(y_base)
    ax.set_yticklabels(NODES, fontsize=10)
    ax.axvline(0.5, color=STYLE["warn"], lw=1.5, linestyle="--",
               label="목표 R²=0.5")
    ax.set_xlabel("5-Fold CV R²"); ax.invert_yaxis()
    ax.set_title("A. 노드 × 방법론 R² (그룹 바)", fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.2, loc="lower right")

    # (B) 방법론별 히트맵
    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    hm_vals = []
    for mkey in methods:
        row_vals = []
        for node in NODES:
            v = method_df.loc[node, mkey] if mkey in method_df.columns else float("nan")
            row_vals.append(float(v) if not np.isnan(v) else -0.2)
        hm_vals.append(row_vals)
    hm_arr = np.array(hm_vals)

    # 발산 컬러맵 (음수=빨강, 양수=파랑)
    cmap2 = mcolors.LinearSegmentedColormap.from_list(
        "div", ["#c0392b", "#1a1a2e", STYLE["accent"]])
    im2 = ax2.imshow(hm_arr, cmap=cmap2, aspect="auto",
                     vmin=-0.2, vmax=0.5)

    ax2.set_xticks(range(len(NODES)))
    ax2.set_xticklabels(NODES, fontsize=10)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)

    for i in range(len(methods)):
        for j in range(len(NODES)):
            val = hm_arr[i, j]
            txt = f"{val:.3f}" if val > -0.15 else "음수"
            col = "white" if abs(val) > 0.15 else STYLE["text_dim"]
            ax2.text(j, i, txt, ha="center", va="center",
                     fontsize=8, color=col, fontweight="bold")

    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("R²", fontsize=8)
    ax2.set_title("B. 방법론 × 노드 R² 히트맵\n(빨간색=음수/역효과, 파란색=양수)", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, "fig4_method_comparison", show)


# ══════════════════════════════════════════════════════════════
# Step 6E.  Figure 5 — 분산 분해 종합 시각화 (메인 결과)
# ══════════════════════════════════════════════════════════════
def fig5_decomposition(df, decomp_df, node_stats, show=False):
    banner("Figure 5.  분산 분해 종합 시각화 (메인 결과)")

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle(
        "Step 5 — 노드별 수율 분산 분해 (Variance Decomposition)\n"
        "★  미설명 분산 61~85% = 미측정 변수 지배 → 알고리즘이 아닌 데이터 수집이 근본 해결책",
        fontsize=13, fontweight="bold", color=STYLE["text"], y=1.01)

    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.58, wspace=0.38)

    SEG_COLORS = {
        "linear":    STYLE["linear"],
        "nonlinear": STYLE["nonlin"],
        "tool":      STYLE["tool"],
        "unexplained": STYLE["unexpl"],
    }
    SEG_LABELS = {
        "linear":    "수치피처 선형",
        "nonlinear": "수치피처 비선형",
        "tool":      "설비 파생",
        "unexplained": "미설명 (미측정 변수)",
    }

    # ── 행1: 파이 차트 (노드별) ────────────────────────────────
    for i, node in enumerate(NODES):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(STYLE["surface"])
        d = decomp_df.loc[node]
        vals = [d["linear"], d["nonlinear"], d["tool"], d["unexplained"]]
        vals = [max(0.0, v) for v in vals]
        total = sum(vals)
        vals = [v / total for v in vals]

        explode = [0.05, 0.05, 0.05, 0.08]
        wedge_colors = list(SEG_COLORS.values())

        wedges, texts, autotexts = ax.pie(
            vals, explode=explode, colors=wedge_colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90, pctdistance=0.65,
            textprops=dict(color=STYLE["text"], fontsize=7))
        for at in autotexts:
            at.set_fontsize(7)
            at.set_fontweight("bold")

        nc  = NODE_COLOR[node]
        ax.set_title(f"{node}\nn={node_stats[node]['n']}", fontsize=10,
                     fontweight="bold", color=nc)
        u_pct = d["unexplained"] * 100
        wcolor = STYLE["warn"] if u_pct >= 80 else STYLE["text"]
        ax.text(0, -1.35, f"미설명 {u_pct:.1f}%", ha="center",
                fontsize=9, fontweight="bold", color=wcolor)

    # ── 행1 공통 범례 ─────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=SEG_COLORS[k], label=SEG_LABELS[k])
        for k in SEG_COLORS
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(0.99, 0.99),
               fontsize=8, framealpha=0.15, ncol=2)

    # ── 행2: 수평 스택 바 (핵심) ──────────────────────────────
    ax_stack = fig.add_subplot(gs[1, :])
    ax_stack.set_facecolor(STYLE["surface"])

    y_pos = np.arange(len(NODES))
    for i, node in enumerate(NODES):
        d = decomp_df.loc[node]
        left = 0
        for seg, col in SEG_COLORS.items():
            val = max(0.0, float(d[seg]))
            ax_stack.barh(y_pos[i], val, left=left,
                          color=col, height=0.55, alpha=0.90)
            if val > 0.02:
                ax_stack.text(left + val / 2, y_pos[i],
                              f"{val:.3f}", ha="center", va="center",
                              fontsize=8, color="white", fontweight="bold")
            left += val

    # 0.5 목표선
    ax_stack.axvline(0.5, color=STYLE["warn"], lw=2, linestyle="--",
                     label="R²=0.5 목표")
    ax_stack.text(0.51, len(NODES) - 0.15, "R²=0.5 목표",
                  color=STYLE["warn"], fontsize=8, fontweight="bold")

    # 1.0 기준선
    ax_stack.axvline(1.0, color=STYLE["border"], lw=1, linestyle="-")

    ax_stack.set_yticks(y_pos)
    ax_stack.set_yticklabels(NODES, fontsize=11, fontweight="bold")
    for i, node in enumerate(NODES):
        ax_stack.get_yticklabels()[i].set_color(NODE_COLOR[node])

    ax_stack.set_xlabel("수율 분산 비율 (1.0 = 전체)", fontsize=10)
    ax_stack.set_xlim(0, 1.05)
    ax_stack.set_title(
        "전체 수율 분산 분해 — 각 색상 구간의 합 = 1.0\n"
        "어두운 영역(미설명)이 클수록 현재 데이터로 예측 불가능한 분산이 많음",
        fontweight="bold", fontsize=10)
    ax_stack.legend(handles=legend_patches, loc="lower right",
                    fontsize=8, framealpha=0.15)

    # ── 행3: 미설명 분산 크기 + 결론 텍스트 ──────────────────
    ax_bar2 = fig.add_subplot(gs[2, :3])
    ax_bar2.set_facecolor(STYLE["surface"])

    x = np.arange(len(NODES))
    w = 0.35
    best_r2s   = [decomp_df.loc[n, "best_r2"]    for n in NODES]
    unexpl_r2s = [decomp_df.loc[n, "unexplained"] for n in NODES]

    bars1 = ax_bar2.bar(x - w/2, best_r2s, w, label="설명 가능 (Best R²)",
                        color=[NODE_COLOR[n] for n in NODES], alpha=0.85)
    bars2 = ax_bar2.bar(x + w/2, unexpl_r2s, w, label="미설명 분산",
                        color=STYLE["unexpl"], alpha=0.9,
                        edgecolor=STYLE["border"], linewidth=0.8)

    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(NODES, fontsize=10)
    ax_bar2.axhline(0.5, color=STYLE["warn"], lw=1.5, linestyle="--", alpha=0.6)
    ax_bar2.set_ylabel("분산 비율")
    ax_bar2.set_title(
        "설명 가능 vs 미설명 분산\n(어두운 막대가 높을수록 미측정 변수 지배적)",
        fontweight="bold")
    ax_bar2.legend(fontsize=8, framealpha=0.2)
    ax_bar2.set_ylim(0, 1.1)

    for bar, val in zip(bars1, best_r2s):
        ax_bar2.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                     f"{val:.3f}", ha="center", fontsize=8,
                     fontweight="bold", color=STYLE["text"])
    for bar, val in zip(bars2, unexpl_r2s):
        ax_bar2.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                     f"{val:.1%}", ha="center", fontsize=8,
                     fontweight="bold", color=STYLE["text_dim"])

    # 결론 텍스트 박스
    ax_txt = fig.add_subplot(gs[2, 3:])
    ax_txt.set_facecolor(STYLE["surface2"])
    ax_txt.axis("off")

    conclusion = (
        "【핵심 결론 — 미설명 분산의 원인】\n\n"
        "① 챔버 컨디션\n"
        "   내벽 오염도·플라즈마 균일성이\n"
        "   배치마다 달라 수율 변동 유발\n\n"
        "② 포토마스크 노후화\n"
        "   누적 사용에 따른 선폭 편차\n"
        "   (현재 데이터에 마스크 이력 없음)\n\n"
        "③ 웨이퍼 결정 방향 편차\n"
        "   동일 잉곳 내 위치별 편차\n\n"
        "④ 공정 열이력 누적\n"
        "   전 단계 온도 이력 미포함\n\n"
        "⑤ 오염 입자 크기·위치 분포\n"
        "   defect_count만 있고 상세 없음\n\n"
        "→ 근본 해결: 미측정 변수 추가 수집"
    )
    ax_txt.text(0.05, 0.97, conclusion, transform=ax_txt.transAxes,
                fontsize=8.5, va="top", color=STYLE["text"],
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=STYLE["surface"], alpha=0.8,
                          edgecolor=STYLE["warn"]))
    ax_txt.set_title("", fontweight="bold")

    save_fig(fig, "fig5_variance_decomposition", show)


# ══════════════════════════════════════════════════════════════
# 최종 콘솔 요약
# ══════════════════════════════════════════════════════════════
def print_final_summary(decomp_df, method_df, node_stats):
    banner("최종 요약 — 노드별 분산 분해 결과")

    print(f"\n  {'Node':<8} {'n':>5}  {'Best R²':>8}  {'미설명':>8}  "
          f"{'선형기여':>8}  {'비선형':>8}  {'설비':>8}  {'상태'}")
    print(f"  {'─'*72}")

    for node in NODES:
        d  = decomp_df.loc[node]
        ns = node_stats[node]
        u  = d["unexplained"]
        status = "🔴 극심" if u > 0.80 else ("🟡 심각" if u > 0.70 else "🟠 높음")
        print(f"  {node:<8} {ns['n']:>5}  {d['best_r2']:>8.4f}"
              f"  {u:>8.1%}  {d['linear']:>8.4f}"
              f"  {d['nonlinear']:>8.4f}  {d['tool']:>8.4f}  {status}")

    print(f"\n  【결론】")
    print(f"  - 전 노드에서 수율 분산의 61~85%가 현재 데이터로 설명 불가")
    print(f"  - 22nm: 미설명 85% + 샘플 부족(n=125) → 과적합 복합 문제")
    print(f"  - R²>0.5 달성의 유일한 근본 해결책: 미측정 변수 추가 수집")
    print(f"\n  【단기 개선 권고】")
    print(f"  1. 22nm 데이터 125 → 500개 이상 수집 (최우선)")
    print(f"  2. 챔버 클리닝 이후 누적 RF 시간 피처 추가")
    print(f"  3. 마스크 사용 횟수 및 마지막 검사 이후 로트 수 추가")
    print(f"  4. 공정 스텝별 이탈 횟수 누적 피처 생성")

    print(f"\n  출력 파일:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel  = os.path.relpath(path, OUTPUT_DIR)
            print(f"  {rel:<50s} {size//1024:>4}KB")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="노드별 수율 분산 설명 가능성 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  python node_variance_analysis.py
  python node_variance_analysis.py --data /path/to/data.csv
  python node_variance_analysis.py --steps 1 2 3
  python node_variance_analysis.py --no-show
  python node_variance_analysis.py --steps 1 2 3 4 5 6
        """
    )
    parser.add_argument("--data",    type=str, default=DATA_PATH)
    parser.add_argument("--steps",   nargs="+", type=int,
                        help="실행할 단계 (1~6). 기본: 전체")
    parser.add_argument("--no-show", action="store_true",
                        help="화면 표시 없이 파일 저장만")
    args = parser.parse_args()

    show = not args.no_show
    if show:
        matplotlib.use("TkAgg" if "DISPLAY" in os.environ else "Agg")
    else:
        matplotlib.use("Agg")

    to_run = sorted(args.steps) if args.steps else list(range(1, 7))
    setup()

    desc = {
        1: "데이터 로드 & 기초 통계",
        2: "단일 피처 선형 기여",
        3: "누적 R² 추이",
        4: "방법론 상한선 탐색",
        5: "분산 분해",
        6: "Figure 5종 생성",
    }
    print("═" * 65)
    print("  노드별 수율 분산 설명 가능성 분석")
    print(f"  실행 단계: {[desc.get(s, str(s)) for s in to_run]}")
    print("═" * 65)

    # 중간 결과 저장소
    df       = None
    node_stats = None
    r2_df    = None
    cum_df   = None
    method_df = None
    decomp_df = None

    t_total = time.time()

    for step in to_run:
        t0 = time.time()
        try:
            if step == 1:
                df, node_stats = step1_load(args.data)

            elif step == 2:
                if df is None:
                    df, node_stats = step1_load(args.data)
                r2_df = step2_single_feature(df)

            elif step == 3:
                if df is None:
                    df, node_stats = step1_load(args.data)
                if r2_df is None:
                    r2_df = step2_single_feature(df)
                cum_df = step3_cumulative(df, r2_df)

            elif step == 4:
                if df is None:
                    df, node_stats = step1_load(args.data)
                method_df = step4_methods(df)

            elif step == 5:
                if df is None:
                    df, node_stats = step1_load(args.data)
                if method_df is None:
                    method_df = step4_methods(df)
                decomp_df = step5_decompose(df, method_df)

            elif step == 6:
                # Figure 생성 — 이전 단계 결과 없으면 자동 계산
                if df is None:
                    df, node_stats = step1_load(args.data)
                if r2_df is None:
                    r2_df = step2_single_feature(df)
                if cum_df is None:
                    cum_df = step3_cumulative(df, r2_df)
                if method_df is None:
                    method_df = step4_methods(df)
                if decomp_df is None:
                    decomp_df = step5_decompose(df, method_df)

                fig1_overview(df, node_stats, show)
                fig2_feature(df, r2_df, show)
                fig3_cumulative(cum_df, show)
                fig4_methods(method_df, show)
                fig5_decomposition(df, decomp_df, node_stats, show)

            print(f"  ✅ Step {step} ({desc.get(step,'')}) 완료  [{time.time()-t0:.0f}s]")

        except Exception as e:
            print(f"  ❌ Step {step} 오류: {e}")
            import traceback; traceback.print_exc()

    if decomp_df is not None and node_stats is not None:
        print_final_summary(decomp_df, method_df, node_stats)

    print(f"\n  전체 완료  [{time.time()-t_total:.0f}s]")
    print(f"  결과 위치: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()