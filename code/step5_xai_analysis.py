"""
step5_xai_analysis.py
─────────────────────────────────────────────
XAI (Explainable AI) – 5-Method Voting 유의인자 추출

  Method 1: GB Built-in Feature Importance
  Method 2: Random Forest Importance
  Method 3: Extra Trees Importance
  Method 4: Permutation Importance  (model-agnostic)
  Method 5: Pearson |r|             (statistical baseline)

모든 방법의 점수를 [0,1] 정규화 후 단순 평균 → Ensemble Score.
Technology Node별 개별 XAI 분석 포함.
─────────────────────────────────────────────
실행:  python step5_xai_analysis.py
전제:  step4 에서 output/best_model.pkl 생성 완료
출력:  output/xai_results.csv
       output/figures/fig10_xai_voting.png
       output/figures/fig11_xai_by_node.png
       output/figures/fig12_diagnostics.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
    PROCESS_FEATURES, TARGET, ALL_FEATURES, ENCODED_NODE_COL,
    TECH_NODES, NODE_COLORS, FEATURE_LABELS, RANDOM_STATE,
    FS_B_BASE, FS_C_FEATURES
)
from font_setting import set_korean_font
set_korean_font()
FEATURE_LABELS['is_advanced_node'] = 'custom1'
FEATURE_LABELS['defect_composite'] = 'custom2'
FEATURE_LABELS['cd_vth_product'] = 'custom3'
FEATURE_LABELS['uniformity_x_defect'] = 'custom4'
FEATURE_LABELS['node_x_cd'] = 'custom5'

# ═══════════════════════════════════════════
# 0. 데이터 & 모델 로드
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 5: XAI ANALYSIS  (5-Method Voting)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df[ENCODED_NODE_COL] = le.fit_transform(df["technology_node"])

# ── 피처셋 C: EDA 반영 + 피처 엔지니어링
FS_B = [f for f in PROCESS_FEATURES if f != "defect_density"] + [ENCODED_NODE_COL]
df["is_advanced_node"]    = df["technology_node"].isin(["7nm", "10nm"]).astype(int)
df["defect_composite"]    = (df["defect_count"] * df["defect_density"]).apply(np.log1p)
df["cd_vth_product"]      = df["critical_dimension"] * df["vth"]
df["uniformity_x_defect"] = df["thickness_uniformity"] * df["defect_count"]
df["node_x_cd"]           = df[ENCODED_NODE_COL] * df["critical_dimension"]
FS_C = FS_B + [
    "is_advanced_node", "defect_composite",
    "cd_vth_product", "uniformity_x_defect", "node_x_cd",
]

# X = df[ALL_FEATURES].values
X = df[FS_C].values
y = df[TARGET].values

best_model = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")
y_pred     = best_model.predict(X)
print(f"\n▶ 모델 로드 완료: {type(best_model).__name__}")
print(f"▶ Train R²={r2_score(y, y_pred):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y, y_pred)):.4f}")


# ── 정규화 헬퍼 ──────────────────────────
def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-Max normalization to [0, 1]."""
    r = arr - arr.min()
    return r / (r.max() + 1e-10)


# ═══════════════════════════════════════════
# 1. Method 1 – GB Built-in Importance
# ═══════════════════════════════════════════
print("\n▶ [1/5] GB Built-in Importance...", end=" ", flush=True)
gb_imp = best_model.feature_importances_
gb_n   = normalize(gb_imp)
print("완료")

# ═══════════════════════════════════════════
# 2. Method 2 – Random Forest Importance
# ═══════════════════════════════════════════
print("▶ [2/5] Random Forest Importance...", end=" ", flush=True)
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X, y)
rf_imp = rf.feature_importances_
rf_n   = normalize(rf_imp)
print("완료")

# ═══════════════════════════════════════════
# 3. Method 3 – Extra Trees Importance
# ═══════════════════════════════════════════
print("▶ [3/5] Extra Trees Importance...", end=" ", flush=True)
et = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
et.fit(X, y)
et_imp = et.feature_importances_
et_n   = normalize(et_imp)
print("완료")

# ═══════════════════════════════════════════
# 4. Method 4 – Permutation Importance
# ═══════════════════════════════════════════
print("▶ [4/5] Permutation Importance...", end=" ", flush=True)
perm_result = permutation_importance(best_model, X, y,
                                     n_repeats=10, random_state=RANDOM_STATE,
                                     n_jobs=-1)
perm_imp = np.clip(perm_result.importances_mean, 0, None)
perm_n   = normalize(perm_imp)
print("완료")

# ═══════════════════════════════════════════
# 5. Method 5 – Pearson |r|
# ═══════════════════════════════════════════
print("▶ [5/5] Pearson |r|...", end=" ", flush=True)
pearson_imp = np.array([
    abs(stats.pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])
])
pearson_n = normalize(pearson_imp)
print("완료")

# ═══════════════════════════════════════════
# 6. Ensemble Voting Score
# ═══════════════════════════════════════════
ensemble = (gb_n + rf_n + et_n + perm_n + pearson_n) / 5
# print(len(FS_C_FEATURES))
# print(len([FEATURE_LABELS.get(f, f) for f in ALL_FEATURES]))
# print(len(gb_n))
# print(len(rf_n))
# print(len(et_n))
# print(len(perm_n))
# print(len(pearson_n))
# print(len(ensemble))
xai_df = pd.DataFrame({
    "feature":       FS_C_FEATURES,
    "description":   [FEATURE_LABELS.get(f, f) for f in FS_C_FEATURES],
    "GB_importance": gb_n,
    "RF_importance": rf_n,
    "ET_importance": et_n,
    "Permutation":   perm_n,
    "Pearson":       pearson_n,
    "Ensemble_Score": ensemble,
}).sort_values("Ensemble_Score", ascending=False).reset_index(drop=True)
xai_df["rank"] = range(1, len(xai_df) + 1)

print("\n▶ XAI Voting 결과:")
print(xai_df[["rank", "feature",
              "GB_importance", "RF_importance", "ET_importance",
              "Permutation", "Pearson", "Ensemble_Score"]].round(4).to_string(index=False))

xai_df.to_csv(f"{OUTPUT_DIR}/xai_results.csv", index=False)
print(f"\n▶ xai_results.csv 저장: {OUTPUT_DIR}/xai_results.csv")

# ═══════════════════════════════════════════
# 7. Fig 10: XAI Voting Matrix Heatmap + Bar
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.suptitle("Fig 10: XAI – 5-Method Voting Feature Importance",
             fontsize=13, fontweight="bold")

# ── Left: Heatmap ──
method_cols  = ["GB_importance", "RF_importance", "ET_importance",
                "Permutation", "Pearson"]
method_labels = ["GB", "RF", "ET", "Permutation", "Pearson"]
heatmap_data  = xai_df[method_cols].values

im = axes[0].imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
axes[0].set_xticks(range(len(method_labels)))
axes[0].set_xticklabels(method_labels, fontsize=10, fontweight="bold")
axes[0].set_yticks(range(len(xai_df)))
axes[0].set_yticklabels(xai_df["feature"], fontsize=8)
axes[0].set_title("Voting Matrix\n(정규화된 각 방법 점수)", fontweight="bold")
plt.colorbar(im, ax=axes[0], label="Normalized Importance")
for i in range(len(xai_df)):
    for j in range(len(method_labels)):
        val = heatmap_data[i, j]
        axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if val > 0.6 else "black")

# ── Right: Ensemble Bar ──
colors_bar = (["#e74c3c"] * 3 +          # Top 3
              ["#e67e22"] * 5 +          # Top 4~8
              ["#3498db"] * (len(xai_df) - 8))
axes[1].barh(range(len(xai_df)), xai_df["Ensemble_Score"].values,
             color=colors_bar, alpha=0.85)
axes[1].set_yticks(range(len(xai_df)))
axes[1].set_yticklabels(xai_df["feature"], fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel("Ensemble Voting Score")
axes[1].set_title("Ensemble 유의인자 순위\n(빨=Top3, 주황=Top4~8, 파=나머지)",
                  fontweight="bold")
for i, val in enumerate(xai_df["Ensemble_Score"]):
    axes[1].text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)
# 범례 patches
from matplotlib.patches import Patch
legend_elems = [Patch(fc="#e74c3c", label="Top 3"),
                Patch(fc="#e67e22", label="Top 4~8"),
                Patch(fc="#3498db", label="Others")]
axes[1].legend(handles=legend_elems, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_xai_voting.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 10] XAI voting 저장 완료")

# ═══════════════════════════════════════════
# 8. Fig 11: Technology Node별 XAI
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, len(TECH_NODES),
                         figsize=(4 * len(TECH_NODES), 7))
fig.suptitle("Fig 11: XAI Importance by Technology Node (RF)",
             fontsize=12, fontweight="bold")

for ax, node in zip(axes, TECH_NODES):
    sub  = df[df["technology_node"] == node]
    X_sub = sub[ALL_FEATURES].values
    y_sub = sub[TARGET].values
    rf_n_model = RandomForestRegressor(n_estimators=100,
                                       random_state=RANDOM_STATE, n_jobs=-1)
    rf_n_model.fit(X_sub, y_sub)
    imp = (pd.Series(rf_n_model.feature_importances_, index=ALL_FEATURES)
           .sort_values(ascending=True))
    ax.barh(imp.index, imp.values, color=NODE_COLORS[node], alpha=0.85)
    ax.set_title(f"{node}", fontweight="bold", color=NODE_COLORS[node])
    ax.set_xlabel("RF Importance")

fig.suptitle("Fig 11: XAI Importance by Technology Node (RF)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig11_xai_by_node.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 11] XAI by node 저장 완료")

# ═══════════════════════════════════════════
# 9. Fig 12: Model Diagnostics
# ═══════════════════════════════════════════
residuals = y - y_pred

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Fig 12: Model Diagnostics – Predicted vs Actual & Residuals",
             fontsize=13, fontweight="bold")

# 12-A: Pred vs Actual (전체)
for node in TECH_NODES:
    mask = df["technology_node"] == node
    axes[0, 0].scatter(y[mask], y_pred[mask], alpha=0.4, s=15,
                       color=NODE_COLORS[node], label=node)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()],
                "k--", linewidth=1.5, label="Perfect")
axes[0, 0].set_xlabel("Actual Yield"); axes[0, 0].set_ylabel("Predicted Yield")
axes[0, 0].set_title(f"Predicted vs Actual  (R²={r2_score(y, y_pred):.3f})")
axes[0, 0].legend(fontsize=7, markerscale=2)

# 12-B: Residual Plot
axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=8, color="#34495e")
axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 1].set_xlabel("Predicted Yield"); axes[0, 1].set_ylabel("Residual")
axes[0, 1].set_title("Residual Plot")

# 12-C: Residual Histogram
axes[0, 2].hist(residuals, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
axes[0, 2].set_xlabel("Residual"); axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title(f"Residual Dist  (μ={residuals.mean():.4f}, σ={residuals.std():.4f})")

# 12-D~F: Node별 Pred vs Actual (7nm, 10nm, 14nm)
for i, node in enumerate(["7nm", "10nm", "14nm"]):
    ax = axes[1, i]
    mask = df["technology_node"] == node
    y_n, yp_n = y[mask], y_pred[mask]
    ax.scatter(y_n, yp_n, alpha=0.5, s=20, color=NODE_COLORS[node])
    ax.plot([y_n.min(), y_n.max()], [y_n.min(), y_n.max()], "k--", linewidth=1.5)
    r2_n = r2_score(y_n, yp_n)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{node}  (R²={r2_n:.3f})")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig12_diagnostics.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 12] Diagnostics 저장 완료")

# ═══════════════════════════════════════════
# 10. 최종 요약 출력
# ═══════════════════════════════════════════
print("\n" + "═" * 60)
print("▶ XAI 최종 유의인자 순위")
print("═" * 60)
for _, row in xai_df.iterrows():
    bar = "█" * int(row["Ensemble_Score"] * 30)
    print(f"  {int(row['rank']):2d}. {row['feature']:<24s}  "
          f"{row['Ensemble_Score']:.4f}  {bar}")

print("\n[STEP 5 완료] xai_results.csv + 3개 Figure 생성")
