"""
step2_feature_selection.py
─────────────────────────────────────────────
3-Method Ensemble Feature Selection
  1) Pearson |r|      – 선형 의존성
  2) Mutual Information – 비선형 의존성
  3) Random Forest Importance – 복합 상호작용

Technology Node별 개별 상관관계 분석 포함.
Tool Group 내 피처 기여도 분석 포함.
─────────────────────────────────────────────
실행:  python step2_feature_selection.py
출력:  output/feature_scores.csv
       output/figures/fig7_feature_selection.png
       output/figures/fig7b_node_correlation.png
       output/figures/fig7c_toolgroup_contribution.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
    PROCESS_FEATURES, TARGET, ALL_FEATURES, ENCODED_NODE_COL,
    TOOL_GROUPS, TECH_NODES, NODE_COLORS, FEATURE_LABELS, RANDOM_STATE,
)
from font_setting import set_korean_font
set_korean_font()
# ═══════════════════════════════════════════
# 0. 데이터 로드 & 전처리
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 2: FEATURE SELECTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df[ENCODED_NODE_COL] = le.fit_transform(df["technology_node"])

X = df[ALL_FEATURES].values
y = df[TARGET].values

print(f"\n▶ Feature matrix shape : {X.shape}")
print(f"▶ Target shape         : {y.shape}")

# ═══════════════════════════════════════════
# 1. Method 1 – Pearson |r|
# ═══════════════════════════════════════════
print("\n▶ [1/3] Pearson |r| 계산 중...")
pearson_scores = np.array([
    abs(stats.pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])
])
print("  완료")

# ═══════════════════════════════════════════
# 2. Method 2 – Mutual Information
# ═══════════════════════════════════════════
print("▶ [2/3] Mutual Information 계산 중...")
mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
print("  완료")

# ═══════════════════════════════════════════
# 3. Method 3 – Random Forest Importance
# ═══════════════════════════════════════════
print("▶ [3/3] Random Forest Importance 계산 중...")
rf_sel = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X, y)
rf_scores = rf_sel.feature_importances_
print("  완료")

# ═══════════════════════════════════════════
# 4. 앙상블 점수 계산
# ═══════════════════════════════════════════
def normalize(arr: np.ndarray) -> np.ndarray:
    r = arr - arr.min()
    return r / (r.max() + 1e-10)

pearson_n = normalize(pearson_scores)
mi_n      = normalize(mi_scores)
rf_n      = normalize(rf_scores)

ensemble_score = (pearson_n + mi_n + rf_n) / 3

feature_df = pd.DataFrame({
    "feature":         ALL_FEATURES,
    "description":     [FEATURE_LABELS.get(f, f) for f in ALL_FEATURES],
    "pearson_abs_r":   pearson_scores,
    "pearson_norm":    pearson_n,
    "mutual_info":     mi_scores,
    "mi_norm":         mi_n,
    "rf_importance":   rf_scores,
    "rf_norm":         rf_n,
    "ensemble_score":  ensemble_score,
}).sort_values("ensemble_score", ascending=False).reset_index(drop=True)

feature_df["rank"] = range(1, len(feature_df) + 1)

print("\n▶ Feature Selection 결과:")
print(feature_df[["rank", "feature", "pearson_abs_r",
                   "mi_norm", "rf_norm", "ensemble_score"]].round(4).to_string(index=False))

# 저장
feature_df.to_csv(f"{OUTPUT_DIR}/feature_scores.csv", index=False)
print(f"\n▶ feature_scores.csv 저장: {OUTPUT_DIR}/feature_scores.csv")

# ═══════════════════════════════════════════
# 5. Fig 7: 3-Method 비교 Bar Chart
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 7))
fig.suptitle("Fig 7: Feature Selection – 3-Method Comparison",
             fontsize=13, fontweight="bold")

methods  = ["pearson_abs_r", "mi_norm", "rf_norm", "ensemble_score"]
titles   = ["Pearson |r|", "Mutual Information\n(normalized)",
            "RF Importance\n(normalized)", "Ensemble Score\n(Final ★)"]
palettes = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

for ax, method, title, color in zip(axes, methods, titles, palettes):
    sorted_df = feature_df.sort_values(method, ascending=True)
    bars = ax.barh(sorted_df["feature"], sorted_df[method], color=color, alpha=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Score")
    # Ensemble 패널에서 Top 10 강조
    if method == "ensemble_score":
        top10 = feature_df.head(10)["feature"].tolist()
        for bar, feat in zip(bars, sorted_df["feature"]):
            if feat in top10:
                bar.set_edgecolor("red"); bar.set_linewidth(2.0)
        ax.set_title(title + "\n(빨간 테두리 = Top 10)", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_feature_selection.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 7] Feature selection 저장 완료")

# ═══════════════════════════════════════════
# 6. Fig 7b: Technology Node별 피처-Yield 상관
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, len(TECH_NODES),
                         figsize=(4 * len(TECH_NODES), 7), sharey=False)
fig.suptitle("Fig 7b: Feature–Yield Pearson r by Technology Node",
             fontsize=12, fontweight="bold")

for ax, node in zip(axes, TECH_NODES):
    sub  = df[df["technology_node"] == node]
    corr = (sub[PROCESS_FEATURES + [TARGET]]
            .corr()[TARGET].drop(TARGET).sort_values())
    ax.barh(corr.index, corr.values,
            color=["#e74c3c" if v < 0 else "#2ecc71" for v in corr.values],
            alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(node, fontweight="bold", color=NODE_COLORS[node])
    ax.set_xlabel("Pearson r")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7b_node_correlation.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 7b] Node-level correlation 저장 완료")

# ═══════════════════════════════════════════
# 7. Fig 7c: Tool Group 내 피처 기여도
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, len(TOOL_GROUPS), figsize=(14, 4))
fig.suptitle("Fig 7c: Ensemble Score within Each Tool Group",
             fontsize=12, fontweight="bold")

tool_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
for ax, (tool, features), color in zip(axes, TOOL_GROUPS.items(), tool_colors):
    tg = feature_df[feature_df["feature"].isin(features)][["feature", "ensemble_score"]]
    ax.bar(tg["feature"], tg["ensemble_score"], color=color, alpha=0.85)
    ax.set_title(tool, fontsize=9, fontweight="bold")
    ax.set_ylabel("Ensemble Score"); ax.tick_params(axis="x", rotation=25, labelsize=8)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7c_toolgroup_contribution.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 7c] Tool group contribution 저장 완료")

# ═══════════════════════════════════════════
# 8. 요약 출력
# ═══════════════════════════════════════════
print("\n" + "─" * 50)
print("▶ 피처 선택 최종 순위 (Ensemble Score 기준)")
print("─" * 50)
for _, row in feature_df.iterrows():
    bar = "█" * int(row["ensemble_score"] * 40)
    print(f"  {int(row['rank']):2d}. {row['feature']:<24s}  {row['ensemble_score']:.4f}  {bar}")

print("\n[STEP 2 완료] feature_scores.csv + 3개 Figure 생성")
