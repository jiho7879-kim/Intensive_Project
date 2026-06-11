"""
step3_baseline_models.py  (v2 — EDA 반영 버전)
─────────────────────────────────────────────────────────────────
Phase 1 — 전체 모델군 베이스라인 스크리닝 (17개 모델)

  ▶ 선형 계열  (7개): Linear, Ridge, Lasso, ElasticNet,
                       Huber, Bayesian Ridge, PLS
  ▶ 비선형     (3개): SVR(RBF), KNN, MLP
  ▶ 단일 트리  (1개): Decision Tree
  ▶ 앙상블     (6개): Bagging, Random Forest, Extra Trees,
                       AdaBoost, Gradient Boosting, HistGBM

EDA 반영 피처셋 3종 (A/B/C) 모두에 대해 평가하고,
최적 피처셋을 자동 선택하여 Phase 2로 전달합니다.

실행:  python step3_baseline_models.py
출력:  output/baseline_results.csv
       output/baseline_results_by_fs.csv   ← 피처셋별 비교
       output/figures/fig8_baseline_comparison.png
       output/figures/fig8b_featureset_comparison.png
─────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, HuberRegressor, BayesianRidge)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor,
                               BaggingRegressor, HistGradientBoostingRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
    PROCESS_FEATURES, TARGET, ALL_FEATURES, ENCODED_NODE_COL,
    RANDOM_STATE, CV_FOLDS,
)
from font_setting import set_korean_font
set_korean_font()
# ═══════════════════════════════════════════════════════════════
# 0. 데이터 로드 & 피처셋 3종 구성
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 3: PHASE 1 — BASELINE SCREENING (17 models, 3 feature sets)")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df[ENCODED_NODE_COL] = le.fit_transform(df["technology_node"])

# ── 피처셋 A: 원본 (EDA 미반영)
FS_A = PROCESS_FEATURES + [ENCODED_NODE_COL]

# ── 피처셋 B: EDA — 다중공선성 제거 (defect_density, r=0.9996 with thickness_uniformity)
FS_B = [f for f in PROCESS_FEATURES if f != "defect_density"] + [ENCODED_NODE_COL]

# ── 피처셋 C: EDA + 피처 엔지니어링 (이분화 그룹 / 복합지수 / 상호작용)
df["is_advanced_node"]    = df["technology_node"].isin(["7nm", "10nm"]).astype(int)
df["defect_composite"]    = (df["defect_count"] * df["defect_density"]).apply(np.log1p)
df["cd_vth_product"]      = df["critical_dimension"] * df["vth"]
df["uniformity_x_defect"] = df["thickness_uniformity"] * df["defect_count"]
df["node_x_cd"]           = df[ENCODED_NODE_COL] * df["critical_dimension"]
FS_C = FS_B + [
    "is_advanced_node", "defect_composite",
    "cd_vth_product", "uniformity_x_defect", "node_x_cd",
]

FEATURE_SETS = {
    "A-Baseline (원본 19개)":          FS_A,
    "B-EDA 다중공선성 제거 (18개)":    FS_B,
    "C-EDA + 피처엔지니어링 (23개)":   FS_C,
}

y = df[TARGET].values
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n▶ 피처셋 구성:")
for name, fset in FEATURE_SETS.items():
    print(f"   {name}: {len(fset)}개")

# ═══════════════════════════════════════════════════════════════
# 1. 모델 정의 (카테고리 + 스케일링 여부 포함)
# ═══════════════════════════════════════════════════════════════
CAT_COLORS = {
    "linear": "#3498db", "nonlinear": "#e67e22",
    "tree":   "#95a5a6", "ensemble":  "#e74c3c",
}

MODELS = {
    # name: (estimator, needs_scaling, category)
    "Linear Regression": (LinearRegression(),                                          True,  "linear"),
    "Ridge":             (Ridge(alpha=1.0),                                             True,  "linear"),
    "Lasso":             (Lasso(alpha=0.001, max_iter=5000),                            True,  "linear"),
    "ElasticNet":        (ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),         True,  "linear"),
    "Huber Regressor":   (HuberRegressor(epsilon=1.35, max_iter=300),                   True,  "linear"),
    "Bayesian Ridge":    (BayesianRidge(),                                              True,  "linear"),
    "PLS Regression":    (PLSRegression(n_components=8),                                True,  "linear"),
    "SVR (RBF)":         (SVR(kernel="rbf", C=1.0),                                     True,  "nonlinear"),
    "KNN (k=5)":         (KNeighborsRegressor(n_neighbors=5),                           True,  "nonlinear"),
    "MLP (2 layers)":    (MLPRegressor(hidden_layer_sizes=(128, 64),
                                       max_iter=500, random_state=RANDOM_STATE),         True,  "nonlinear"),
    "Decision Tree":     (DecisionTreeRegressor(max_depth=8,
                                                random_state=RANDOM_STATE),              False, "tree"),
    "Bagging (DT)":      (BaggingRegressor(
                              estimator=DecisionTreeRegressor(max_depth=6),
                              n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),    False, "ensemble"),
    "Random Forest":     (RandomForestRegressor(
                              n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),   False, "ensemble"),
    "Extra Trees":       (ExtraTreesRegressor(
                              n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),   False, "ensemble"),
    "AdaBoost":          (AdaBoostRegressor(
                              n_estimators=100, random_state=RANDOM_STATE),              False, "ensemble"),
    "Gradient Boosting": (GradientBoostingRegressor(
                              n_estimators=100, random_state=RANDOM_STATE),              False, "ensemble"),
    "HistGBM":           (HistGradientBoostingRegressor(
                              max_iter=100, random_state=RANDOM_STATE),                  False, "ensemble"),
}

# ═══════════════════════════════════════════════════════════════
# 2. 피처셋 C 기준 베이스라인 스크리닝 (주 실험)
# ═══════════════════════════════════════════════════════════════
print(f"\n▶ Phase 1 — 피처셋 C 기준 5-Fold CV ({len(MODELS)}개 모델)\n")
print(f"  {'Model':<22s}  {'R² mean':>8s}  {'R² std':>7s}  "
      f"{'RMSE':>7s}  {'MAE':>7s}  {'Time':>6s}")
print("  " + "─" * 65)

X_C   = df[FS_C].values
X_C_s = StandardScaler().fit_transform(X_C)

phase1_results = {}
for name, (model, use_scaled, category) in MODELS.items():
    t0  = time.time()
    Xu  = X_C_s if use_scaled else X_C
    cv  = cross_validate(model, Xu, y, cv=kf,
                         scoring=["r2", "neg_mean_squared_error",
                                  "neg_mean_absolute_error"],
                         n_jobs=-1)
    r2m  =  cv["test_r2"].mean()
    r2s  =  cv["test_r2"].std()
    rmse = np.sqrt(-cv["test_neg_mean_squared_error"].mean())
    mae  = -cv["test_neg_mean_absolute_error"].mean()
    elapsed = time.time() - t0
    phase1_results[name] = dict(
        R2_mean=r2m, R2_std=r2s, RMSE=rmse, MAE=mae,
        category=category, elapsed=elapsed,
    )
    print(f"  {name:<22s}  {r2m:>8.4f}  {r2s:>7.4f}  "
          f"{rmse:>7.4f}  {mae:>7.4f}  {elapsed:>5.1f}s")

p1_df = (pd.DataFrame(phase1_results).T
         .reset_index().rename(columns={"index": "model"})
         .sort_values("R2_mean", ascending=False).reset_index(drop=True))
p1_df["rank"] = range(1, len(p1_df) + 1)
p1_df.to_csv(f"{OUTPUT_DIR}/baseline_results.csv", index=False)

# 카테고리별 요약
print(f"\n▶ 카테고리별 평균 R²:")
cat_avg = p1_df.groupby("category")["R2_mean"].apply(
    lambda s: s.astype(float).mean()).sort_values(ascending=False)
for cat, avg in cat_avg.items():
    print(f"   {cat:<12s}: {avg:.4f}")

# ═══════════════════════════════════════════════════════════════
# 3. EDA 피처셋 비교 (Top 3 앙상블 모델로 A/B/C 비교)
# ═══════════════════════════════════════════════════════════════
print("\n▶ EDA 피처셋 A / B / C 비교 (GBM, RF, HistGBM 기준)...")

probe_models = {
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGBM": HistGradientBoostingRegressor(
        max_iter=100, random_state=RANDOM_STATE),
}

fs_comparison = []
for mname, model in probe_models.items():
    row = {"Model": mname}
    for fs_name, fset in FEATURE_SETS.items():
        Xf  = df[fset].values
        r2  = (cross_validate(model, Xf, y, cv=kf, scoring=["r2"], n_jobs=-1)
               ["test_r2"].mean())
        row[fs_name] = round(float(r2), 4)
    fs_comparison.append(row)

fs_df = pd.DataFrame(fs_comparison)
fs_df.to_csv(f"{OUTPUT_DIR}/baseline_results_by_fs.csv", index=False)

print(f"\n  피처셋별 R² 비교:")
print(fs_df.to_string(index=False))

# 최적 피처셋 결정
fs_means = {name: fs_df[name].mean() for name in FEATURE_SETS}
best_fs_name = max(fs_means, key=fs_means.get)
print(f"\n  → 최적 피처셋: {best_fs_name}  (평균 R²={fs_means[best_fs_name]:.4f})")

# ═══════════════════════════════════════════════════════════════
# 4. Fig 8: Phase 1 종합 비교 (3-Panel)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(19, 8))
fig.suptitle(
    "Fig 8: Phase 1 — 전체 모델군 베이스라인 스크리닝\n"
    f"(피처셋 C 기준, {CV_FOLDS}-Fold CV, 17개 모델)",
    fontsize=13, fontweight="bold"
)

# Panel 1: R² 수평 막대
ax = axes[0]
colors_p1 = [CAT_COLORS[p1_df.loc[p1_df["model"] == m, "category"].values[0]]
             for m in p1_df["model"]]
bars = ax.barh(p1_df["model"], p1_df["R2_mean"].astype(float),
               color=colors_p1, alpha=0.85)
ax.errorbar(p1_df["R2_mean"].astype(float), range(len(p1_df)),
            xerr=p1_df["R2_std"].astype(float),
            fmt="none", color="black", capsize=3, linewidth=1.5)
ax.invert_yaxis()
ax.axvline(0.60, color="orange", linestyle="--", linewidth=1.5, label="R²=0.60 기준선")
legend_elems = [mpatches.Patch(fc=v, label=k) for k, v in CAT_COLORS.items()]
ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
ax.set_xlabel("R² (5-Fold CV)", fontsize=11)
ax.set_title("R² 비교 (오차막대=±1std)", fontweight="bold")
for bar, val in zip(bars, p1_df["R2_mean"].astype(float)):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7.5)

# Panel 2: RMSE
ax2 = axes[1]
p1s = p1_df.sort_values("RMSE")
bars2 = ax2.barh(p1s["model"], p1s["RMSE"].astype(float),
                 color=[CAT_COLORS[p1s.loc[p1s["model"] == m, "category"].values[0]]
                        for m in p1s["model"]], alpha=0.85)
ax2.invert_yaxis()
ax2.set_xlabel("RMSE (↓우수)", fontsize=11)
ax2.set_title("RMSE 비교", fontweight="bold")
for bar, val in zip(bars2, p1s["RMSE"].astype(float)):
    ax2.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=7.5)

# Panel 3: 카테고리별 평균 R²
ax3 = axes[2]
cat_avg_sorted = cat_avg.sort_values(ascending=False)
bars3 = ax3.bar(cat_avg_sorted.index, cat_avg_sorted.values,
                color=[CAT_COLORS.get(c, "#95a5a6") for c in cat_avg_sorted.index],
                alpha=0.85, width=0.5)
ax3.set_ylabel("평균 R² (5-CV)", fontsize=11)
ax3.set_title("모델 카테고리별 평균 R²", fontweight="bold")
ax3.set_ylim(0, cat_avg_sorted.max() * 1.12)
for bar, val in zip(bars3, cat_avg_sorted.values):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.004,
             f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_baseline_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[Fig 8] Baseline comparison 저장 완료")

# ═══════════════════════════════════════════════════════════════
# 5. Fig 8b: 피처셋 A / B / C 비교 히트맵
# ═══════════════════════════════════════════════════════════════
import seaborn as sns

comp_mat = fs_df.set_index("Model")
delta_B  = comp_mat.iloc[:, 1] - comp_mat.iloc[:, 0]
delta_C  = comp_mat.iloc[:, 2] - comp_mat.iloc[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Fig 8b: EDA 피처셋 반영 효과 비교 (A → B → C)",
             fontsize=13, fontweight="bold")

sns.heatmap(comp_mat.astype(float), ax=axes[0],
            annot=True, fmt=".4f", cmap="RdYlGn",
            linewidths=0.5, annot_kws={"size": 11, "fontweight": "bold"},
            vmin=comp_mat.values.astype(float).min() - 0.002,
            vmax=comp_mat.values.astype(float).max() + 0.002)
axes[0].set_title("R² (5-Fold CV)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=15, labelsize=8)

x_d = np.arange(len(comp_mat)); w = 0.3
axes[1].bar(x_d - w/2, delta_B.values.astype(float) * 1000,
            w, label="B-A (다중공선성 제거)", color="#3498db", alpha=0.85)
axes[1].bar(x_d + w/2, delta_C.values.astype(float) * 1000,
            w, label="C-A (EDA+피처엔지니어링)", color="#e74c3c", alpha=0.85)
axes[1].set_xticks(x_d)
axes[1].set_xticklabels(comp_mat.index, fontsize=10)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("R² 개선량 (×10⁻³)")
axes[1].set_title("Baseline(A) 대비 EDA 반영 개선량", fontweight="bold")
axes[1].legend(fontsize=9)
for bar in axes[1].patches:
    h = bar.get_height()
    if abs(h) > 0.05:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     h + (0.05 if h >= 0 else -0.18),
                     f"{h:+.2f}‰", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8b_featureset_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 8b] Feature set comparison 저장 완료")

# ═══════════════════════════════════════════════════════════════
# 6. 최종 요약 출력
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("▶ Phase 1 Top 5 후보 (Phase 2 진출)")
print("─" * 65)
top5 = p1_df[p1_df["category"] == "ensemble"].head(5)
for _, row in top5.iterrows():
    bar = "█" * int(float(row["R2_mean"]) * 40)
    print(f"  {int(row['rank']):2d}. {row['model']:<22s}  "
          f"R²={float(row['R2_mean']):.4f}  {bar}")

print(f"\n  최적 피처셋: {best_fs_name}")
print("\n[STEP 3 완료] baseline_results.csv · baseline_results_by_fs.csv · Fig 8/8b 생성")
