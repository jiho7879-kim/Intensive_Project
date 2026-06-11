"""
step1_eda.py
─────────────────────────────────────────────
탐색적 데이터 분석 (EDA)
- 기초 통계 출력
- Yield 분포 (전체 + node별)
- 피처-Yield 상관계수 (전체 + node별)
- 전체 상관행렬 히트맵
- 상위 피처 Scatter Plot
- Tool Group Boxplot
- Yield by Node & Product Type (Violin)
- ANOVA 검정
─────────────────────────────────────────────
실행:  python step1_eda.py
출력:  output/figures/fig1_*.png ~ fig6_*.png
       output/eda_summary.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
    PROCESS_FEATURES, TARGET, TOOL_GROUPS,
    TECH_NODES, NODE_COLORS, FEATURE_LABELS,
)

from font_setting import set_korean_font
set_korean_font()
# ═══════════════════════════════════════════
# 0. 데이터 로드
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 1: EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n▶ 데이터 shape : {df.shape}")
print(f"▶ 컬럼        : {df.columns.tolist()}")
print(f"▶ 결측값      :\n{df.isnull().sum()}")
print(f"\n▶ 기초 통계:\n{df[PROCESS_FEATURES + [TARGET]].describe().round(4)}")

# ═══════════════════════════════════════════
# 1. Yield 분포 (전체 + node별)
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Fig 1: Yield Distribution by Technology Node",
             fontsize=14, fontweight="bold")

ax_all = axes[0, 0]
ax_all.hist(df[TARGET], bins=40, color="#34495e", edgecolor="white", alpha=0.8)
ax_all.axvline(df[TARGET].mean(), color="red", linestyle="--",
               label=f"Mean={df[TARGET].mean():.3f}")
ax_all.set_title("Overall"); ax_all.set_xlabel("Yield"); ax_all.set_ylabel("Count")
ax_all.legend()

for i, node in enumerate(TECH_NODES):
    ax = axes[(i + 1) // 3, (i + 1) % 3]
    sub = df[df["technology_node"] == node][TARGET]
    ax.hist(sub, bins=25, color=NODE_COLORS[node], edgecolor="white", alpha=0.8)
    ax.axvline(sub.mean(), color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"{node}  (n={len(sub)}, μ={sub.mean():.3f})")
    ax.set_xlabel("Yield"); ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_yield_dist.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n[Fig 1] Yield distribution 저장 완료")

# ═══════════════════════════════════════════
# 2. 피처-Yield 상관계수 (node별)
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Fig 2: Feature–Yield Correlation by Technology Node",
             fontsize=14, fontweight="bold")

corr_all = (df[PROCESS_FEATURES + [TARGET]]
            .corr()[TARGET].drop(TARGET).sort_values())
axes[0, 0].barh(corr_all.index, corr_all.values,
                color=["#e74c3c" if v < 0 else "#2ecc71" for v in corr_all.values])
axes[0, 0].axvline(0, color="black", linewidth=0.8)
axes[0, 0].set_title("All Nodes"); axes[0, 0].set_xlabel("Pearson r")

for i, node in enumerate(TECH_NODES):
    ax = axes[(i + 1) // 3, (i + 1) % 3]
    sub = df[df["technology_node"] == node]
    corr = sub[PROCESS_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    ax.barh(corr.index, corr.values,
            color=["#e74c3c" if v < 0 else "#2ecc71" for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(node); ax.set_xlabel("Pearson r")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_correlation_by_node.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 2] Correlation by node 저장 완료")

# ═══════════════════════════════════════════
# 3. 전체 상관행렬 히트맵
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df[PROCESS_FEATURES + [TARGET]].corr()
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.3, annot_kws={"size": 7})
ax.set_title("Fig 3: Full Feature Correlation Matrix",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_full_corr_matrix.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 3] Full correlation matrix 저장 완료")

# 3-1. 노드별 상관행렬 히트맵 (노드별 상세 분석)
nodes = df['technology_node'].unique()
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, node in enumerate(nodes):
    node_df = df[df['technology_node'] == node][PROCESS_FEATURES + [TARGET]]
    corr_node = node_df.corr()
    
    sns.heatmap(corr_node, ax=axes[i], annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.3, annot_kws={"size": 7}, cbar=False)
    axes[i].set_title(f"Node: {node}", fontsize=12, fontweight="bold")

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3b_node_corr_matrices.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 3b] 노드별 상관행렬 저장 완료")

# 3-2. '신호등 대시보드' 생성 (액션 중심 표)
dashboard_list = []
# 수치형 컬럼만 추출 (문자열 포함 방지)
numeric_features = df[PROCESS_FEATURES].select_dtypes(include=['number']).columns.tolist()

for node in df['technology_node'].unique():
    node_df = df[df['technology_node'] == node]
    # 수치형 데이터와 타겟만으로 상관관계 계산
    corr_series = node_df[numeric_features + [TARGET]].corr()[TARGET].drop(TARGET)
    
    worst_factor = corr_series.idxmin() if corr_series.min() < -0.2 else "없음"
    best_factor = corr_series.idxmax() if corr_series.max() > 0.05 else "없음"
    
    dashboard_list.append({
        "Technology Node": node,
        "Action: Reduce (↓)": worst_factor,
        "Action: Increase (↑)": best_factor
    })

dashboard_df = pd.DataFrame(dashboard_list)
dashboard_df.to_csv(f"{FIGURES_DIR}/action_dashboard.csv", index=False, encoding='utf-8-sig')
print("[Dashboard] 신호등 대시보드 저장 완료")

# ═══════════════════════════════════════════
# 3-3. 수율 영향도 '버블차트' (영향력 시각화)
# ═══════════════════════════════════════════
plt.figure(figsize=(12, 8))
bubble_data = []

for node in df['technology_node'].unique():
    node_df = df[df['technology_node'] == node]
    # 여기서도 수치형 컬럼만 사용
    corr_series = node_df[numeric_features + [TARGET]].corr()[TARGET].drop(TARGET)
    for feature, corr in corr_series.items():
        bubble_data.append({"Node": node, "Feature": feature, "Correlation": corr})

bubble_df = pd.DataFrame(bubble_data)
plt.scatter(bubble_df["Node"], bubble_df["Feature"], 
            s=bubble_df["Correlation"].abs() * 500, 
            c=bubble_df["Correlation"], cmap="RdYlGn", alpha=0.6, edgecolors="grey")

plt.colorbar(label="Correlation (Red: Negative, Green: Positive)")
plt.title("Fig 3c: Yield Influence Bubble Chart", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3c_bubble_chart.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 3c] 버블차트 저장 완료")

# ═══════════════════════════════════════════
# 4. 상위 피처 Scatter Plot
# ═══════════════════════════════════════════
top_features = ["defect_count", "defect_density", "thickness_uniformity",
                "vth", "critical_dimension"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Fig 4: Top Feature vs Yield Scatter (by Technology Node)",
             fontsize=13, fontweight="bold")
axes_flat = axes.flatten()

for i, feat in enumerate(top_features):
    ax = axes_flat[i]
    for node in TECH_NODES:
        sub = df[df["technology_node"] == node]
        ax.scatter(sub[feat], sub[TARGET], alpha=0.3, s=10,
                   color=NODE_COLORS[node], label=node)
    r, p = stats.pearsonr(df[feat], df[TARGET])
    ax.set_xlabel(feat); ax.set_ylabel("Yield")
    ax.set_title(f"{feat}\n(r={r:.3f}, p={p:.2e})")
    if i == 0:
        ax.legend(fontsize=7, markerscale=2)

axes_flat[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_scatter_top_features.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 4] Scatter top features 저장 완료")

# ═══════════════════════════════════════════
# 5. Tool Group Boxplot
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fig 5: Tool Group – Yield Distribution (Boxplot)",
             fontsize=13, fontweight="bold")
tool_repr = {
    "etch_tool":       "etch_rate",
    "litho_tool":      "focus_offset",
    "deposition_tool": "thickness_uniformity",
    "implant_tool":    "implant_energy",
}
for idx, (tool_col, feat) in enumerate(tool_repr.items()):
    ax = axes[idx // 2, idx % 2]
    tools = sorted(df[tool_col].unique())
    data_by_tool = [df[df[tool_col] == t][TARGET].values for t in tools]
    bp = ax.boxplot(data_by_tool, labels=tools, patch_artist=True)
    colors_bp = plt.cm.Set2(np.linspace(0, 1, len(tools)))
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
    ax.set_title(f"{tool_col} → Yield")
    ax.set_xlabel(tool_col); ax.set_ylabel("Yield")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_tool_boxplot.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 5] Tool boxplot 저장 완료")

# ═══════════════════════════════════════════
# 6. Yield by Node (Violin) & Product Type
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 6: Yield by Technology Node and Product Type",
             fontsize=13, fontweight="bold")

node_data = [df[df["technology_node"] == n][TARGET].values for n in TECH_NODES]
parts = axes[0].violinplot(node_data, showmeans=True, showmedians=True)
axes[0].set_xticks(range(1, len(TECH_NODES) + 1))
axes[0].set_xticklabels(TECH_NODES)
axes[0].set_ylabel("Yield"); axes[0].set_xlabel("Technology Node")
axes[0].set_title("Violin by Technology Node")
for pc, node in zip(parts["bodies"], TECH_NODES):
    pc.set_facecolor(NODE_COLORS[node]); pc.set_alpha(0.7)

prod_types = sorted(df["product_type"].unique())
prod_data  = [df[df["product_type"] == p][TARGET].values for p in prod_types]
axes[1].boxplot(prod_data, labels=prod_types, patch_artist=True)
axes[1].set_ylabel("Yield"); axes[1].set_xlabel("Product Type")
axes[1].set_title("Yield by Product Type")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_yield_by_groups.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 6] Yield by groups 저장 완료")

# ═══════════════════════════════════════════
# 7. ANOVA 검정
# ═══════════════════════════════════════════
print("\n" + "─" * 40)
print("▶ ANOVA: Technology Node별 Yield 차이")
groups   = [df[df["technology_node"] == n][TARGET].values for n in TECH_NODES]
f_stat, p_val = stats.f_oneway(*groups)
print(f"  F-statistic = {f_stat:.3f},  p-value = {p_val:.4e}")
for node in TECH_NODES:
    sub = df[df["technology_node"] == node][TARGET]
    print(f"  {node}: mean={sub.mean():.4f}, std={sub.std():.4f}, n={len(sub)}")

# ═══════════════════════════════════════════
# 8. EDA 요약 CSV 저장
# ═══════════════════════════════════════════
summary_rows = []
for node in TECH_NODES:
    sub = df[df["technology_node"] == node][TARGET]
    summary_rows.append({
        "technology_node": node,
        "n": len(sub),
        "mean_yield": sub.mean(),
        "std_yield":  sub.std(),
        "min_yield":  sub.min(),
        "max_yield":  sub.max(),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTPUT_DIR}/eda_summary.csv", index=False)
print(f"\n▶ EDA 요약 저장: {OUTPUT_DIR}/eda_summary.csv")
print("\n[STEP 1 완료] 총 6개 Figure + eda_summary.csv 생성")
