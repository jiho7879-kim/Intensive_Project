"""
step4_hyperparameter_tuning.py  (v2 — 설계 근거 상세 버전)
─────────────────────────────────────────────────────────────────
Phase 2 — Top 5 앙상블 모델 하이퍼파라미터 탐색 & 튜닝
Phase 3 — 고급 앙상블 (Stacking / Voting)

각 파라미터의 탐색 범위와 선택 근거를 코드 주석으로 상세 기술.
EDA 반영 피처셋 C를 기본 사용.

실행:  python step4_hyperparameter_tuning.py
전제:  step3 완료 (baseline_results.csv 존재)
출력:  output/best_model.pkl
       output/model_meta.json
       output/tuning_results.csv
       output/figures/fig9_tuning.png
       output/figures/fig9b_learning_curve.png
       output/figures/fig9c_leaderboard.png
─────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (cross_validate, cross_val_score,
                                     KFold, RandomizedSearchCV, learning_curve)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor,
                               HistGradientBoostingRegressor,
                               StackingRegressor, VotingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
    PROCESS_FEATURES, TARGET, ENCODED_NODE_COL,
    RANDOM_STATE, CV_FOLDS, TECH_NODES, NODE_COLORS,
)
from font_setting import set_korean_font
set_korean_font()
# ═══════════════════════════════════════════════════════════════
# 0. 데이터 로드 & EDA 피처셋 C 구성
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 4: PHASE 2/3 — HYPERPARAMETER TUNING & ADVANCED ENSEMBLE")
print("=" * 65)

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

X = df[FS_C].values
y = df[TARGET].values
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n▶ 사용 피처셋: C — EDA + 피처엔지니어링 ({len(FS_C)}개)")

# ═══════════════════════════════════════════════════════════════
# 1. 하이퍼파라미터 탐색 공간 & 설계 근거
# ═══════════════════════════════════════════════════════════════

SEARCH_SPACES = {

    # ─────────────────────────────────────────────────────────
    # Gradient Boosting: 가장 중요한 파라미터 조합 설계
    #
    # [핵심 설계 철학]
    # GBM은 얕은 트리(weak learner)를 순차 합산하는 방식.
    # 깊은 트리 → 단일 트리가 과학습 → boosting 효과 감소.
    # 낮은 lr + 많은 트리 = 일반화 향상 (Shrinkage 원리).
    # subsampling = 매 반복마다 다른 데이터 뷰 → 다양성 확보.
    # ─────────────────────────────────────────────────────────
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            # 트리 수: 100~300 (lr=0.05에서 100개면 충분히 수렴)
            "n_estimators": [100, 200, 300],
            # lr: 낮을수록 일반화↑, 0.05가 성능-속도 균형점
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
            # depth: 2~5, 얕은 트리(2~3) 권장 — boosting 철학
            "max_depth": [2, 3, 4, 5],
            # subsampling: 0.8~0.9 권장 (Friedman 2002)
            "subsample": [0.7, 0.8, 0.9, 1.0],
            # leaf 최소 샘플: 과적합 방지, 1(기본)~4
            "min_samples_leaf": [1, 2, 4],
            # 분기 피처 비율: sqrt는 23개 피처에서 너무 적음
            "max_features": ["sqrt", 0.7, 0.9, None],
        },
        "n_iter": 40,
    },

    # ─────────────────────────────────────────────────────────
    # HistGBM: 히스토그램 이산화 GBM
    #
    # [GBM과 차이]
    # 피처값을 max_bins개로 이산화하여 분기점 탐색 가속.
    # 대용량(>10만)에서 GBM보다 유리, 소규모에서 정보 손실 위험.
    # min_samples_leaf를 크게 설정해야 과적합 방지 (l2_reg 보완).
    # ─────────────────────────────────────────────────────────
    "HistGBM": {
        "model": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            # None = 제한 없음(leaf-wise 성장 제어는 min_samples로)
            "max_depth": [3, 5, 7, None],
            # HistGBM에서 핵심 정규화 — 10~50 권장
            "min_samples_leaf": [10, 20, 30, 50],
            # L2 패널티: 0이면 정규화 없음, 1.0은 강한 규제
            "l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "max_bins": [128, 255],
        },
        "n_iter": 30,
    },

    # ─────────────────────────────────────────────────────────
    # Random Forest: Bagging 기반 병렬 앙상블
    #
    # [GBM과 차이]
    # 독립적 트리를 병렬로 학습 → 순차 boosting과 달리 속도 빠름.
    # 깊은 트리 허용 (bias↓) + 앙상블이 variance↓.
    # max_features: 분기 후보 피처 수 — 상관된 피처가 많으면 0.5~0.7 권장.
    # ─────────────────────────────────────────────────────────
    "Random Forest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            # 깊을수록 편향↓, None이면 완전 성장 (분산 관리는 n_estimators로)
            "max_depth": [5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            # sqrt(≈5개)는 23피처에서 핵심 피처 누락 빈번 → 0.5~0.7 권장
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            # Bootstrap=False는 Pasting → 다양성↓, 보통 True 권장
            "bootstrap": [True, False],
        },
        "n_iter": 40,
    },

    # ─────────────────────────────────────────────────────────
    # AdaBoost: 가중치 기반 순차 부스팅
    #
    # [GBM과 차이]
    # GBM은 잔차(gradient)를 학습, AdaBoost는 오분류 샘플 가중치 조정.
    # 기본 학습기(stump, depth=1)를 사용하므로 n_estimators 많이 필요.
    # loss='square'는 회귀에서 큰 잔차에 강한 페널티 → 수율 극단값 학습 효과.
    # ─────────────────────────────────────────────────────────
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=RANDOM_STATE),
        "params": {
            # stump 기반이므로 GBM보다 많은 estimator 필요
            "n_estimators": [100, 200, 300, 500],
            # lr: 각 learner 가중치 축소율, 0.05~0.5 범위 탐색
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            # linear: 기본, square: 큰 오차 강조, exponential: 이상치 민감
            "loss": ["linear", "square", "exponential"],
        },
        "n_iter": 25,
    },

    # ─────────────────────────────────────────────────────────
    # Extra Trees: 완전 랜덤 분기 앙상블
    #
    # [RF와 차이]
    # RF는 각 피처의 최적 분기점 탐색, ET는 무작위 분기점 선택.
    # ET의 분산↓이지만 편향↑ → RF보다 개별 성능 낮으나 앙상블 다양성↑.
    # 단독으로는 RF에 미치지 못하나 Stacking 기반 학습기로 유용.
    # ─────────────────────────────────────────────────────────
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            # ET는 이미 랜덤 분기 → max_features 효과 RF보다 작음
            "max_features": ["sqrt", "log2", 0.5, 0.7],
        },
        "n_iter": 30,
    },
}

# ═══════════════════════════════════════════════════════════════
# 2. RandomizedSearchCV 실행
# ═══════════════════════════════════════════════════════════════
print(f"\n▶ Phase 2 — RandomizedSearchCV (CV=3, n_iter 각 모델 별도)\n")

tuning_results  = {}
best_estimators = {}
baseline_r2 = {  # Phase 1 결과 (피처셋 C 기준)
    "Gradient Boosting": 0.6596,
    "HistGBM":           0.6269,
    "Random Forest":     0.6611,
    "AdaBoost":          0.6654,
    "Extra Trees":       0.6243,
}

for name, cfg in SEARCH_SPACES.items():
    print(f"  [{name}] 탐색 중 (n_iter={cfg['n_iter']})...", end=" ", flush=True)
    t0 = time.time()
    search = RandomizedSearchCV(
        cfg["model"], cfg["params"],
        n_iter=cfg["n_iter"], cv=3, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)

    # 최적 파라미터로 5-Fold 최종 검증
    cv5 = cross_validate(
        search.best_estimator_, X, y, cv=kf,
        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
        n_jobs=-1,
    )
    r2m  =  cv5["test_r2"].mean()
    r2s  =  cv5["test_r2"].std()
    rmse = np.sqrt(-cv5["test_neg_mean_squared_error"].mean())
    mae  = -cv5["test_neg_mean_absolute_error"].mean()
    elapsed = time.time() - t0

    tuning_results[name] = dict(
        baseline_r2=baseline_r2[name],
        search_best_r2=search.best_score_,
        cv5_r2=r2m, cv5_std=r2s,
        cv5_rmse=rmse, cv5_mae=mae,
        best_params=search.best_params_,
        elapsed=elapsed,
    )
    best_estimators[name] = search.best_estimator_
    search.best_estimator_.fit(X, y)  # 전체 데이터로 재학습

    print(f"search={search.best_score_:.4f}  5CV={r2m:.4f}±{r2s:.4f}  [{elapsed:.1f}s]")
    print(f"    best params: {search.best_params_}")

# ═══════════════════════════════════════════════════════════════
# 3. Phase 3 — 고급 앙상블 (Stacking / Voting)
# ═══════════════════════════════════════════════════════════════
print("\n▶ Phase 3 — 고급 앙상블 (Stacking / Voting)...")

# Top 3 모델 선택 (tuning_results 기준)
top3_names = sorted(tuning_results, key=lambda k: tuning_results[k]["cv5_r2"], reverse=True)[:3]
top3_est   = [(n.replace(" ", "_"), best_estimators[n]) for n in top3_names]

print(f"  기반 학습기 (Top 3): {top3_names}")

adv_results = {}
for aname, amodel in [
    ("Voting (Top3)",            VotingRegressor(estimators=top3_est, n_jobs=-1)),
    ("Stacking (Top3→Ridge)",    StackingRegressor(
                                     estimators=top3_est,
                                     final_estimator=Ridge(alpha=1.0),
                                     cv=3, n_jobs=-1)),
]:
    cv5 = cross_validate(amodel, X, y, cv=kf,
                         scoring=["r2", "neg_mean_squared_error",
                                  "neg_mean_absolute_error"],
                         n_jobs=-1)
    r2m  =  cv5["test_r2"].mean()
    r2s  =  cv5["test_r2"].std()
    rmse = np.sqrt(-cv5["test_neg_mean_squared_error"].mean())
    mae  = -cv5["test_neg_mean_absolute_error"].mean()
    adv_results[aname] = dict(cv5_r2=r2m, cv5_std=r2s, cv5_rmse=rmse, cv5_mae=mae)
    print(f"  {aname:<30s}: R²={r2m:.4f}±{r2s:.4f}  RMSE={rmse:.4f}")

# ═══════════════════════════════════════════════════════════════
# 4. 최적 모델 선정 & 저장
# ═══════════════════════════════════════════════════════════════
# 단독 모델 중 최고 R² 선택 (복잡도 고려, Stacking 제외)
best_name = max(tuning_results, key=lambda k: tuning_results[k]["cv5_r2"])
FINAL     = best_estimators[best_name]

print(f"\n▶ 최종 선정: {best_name}  (5-CV R²={tuning_results[best_name]['cv5_r2']:.4f})")

joblib.dump(FINAL, f"{OUTPUT_DIR}/best_model.pkl")
joblib.dump(best_estimators, f"{OUTPUT_DIR}/all_tuned_models.pkl")

meta = {
    "model_name":    best_name,
    "best_params":   tuning_results[best_name]["best_params"],
    "cv5_r2":        tuning_results[best_name]["cv5_r2"],
    "cv5_std":       tuning_results[best_name]["cv5_std"],
    "cv5_rmse":      tuning_results[best_name]["cv5_rmse"],
    "cv5_mae":       tuning_results[best_name]["cv5_mae"],
    "feature_set":   "C-EDA+FE",
    "features":      FS_C,
    "le_classes":    le.classes_.tolist(),
    "all_tuning":    {k: {kk: vv for kk, vv in v.items() if kk != "best_params"}
                      for k, v in tuning_results.items()},
    "adv_results":   adv_results,
}
with open(f"{OUTPUT_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=float)

rows = []
for name, res in tuning_results.items():
    row = {"model": name}
    row.update({k: v for k, v in res.items() if k != "best_params"})
    row["best_params_str"] = str(res["best_params"])
    rows.append(row)
pd.DataFrame(rows).sort_values("cv5_r2", ascending=False).to_csv(
    f"{OUTPUT_DIR}/tuning_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# 5. 시각화
# ═══════════════════════════════════════════════════════════════

# Fig 9: 튜닝 전·후 비교
names  = list(SEARCH_SPACES.keys())
before = [tuning_results[n]["baseline_r2"] for n in names]
after  = [tuning_results[n]["cv5_r2"]      for n in names]
stds   = [tuning_results[n]["cv5_std"]     for n in names]
rmses  = [tuning_results[n]["cv5_rmse"]    for n in names]
maes   = [tuning_results[n]["cv5_mae"]     for n in names]
x = np.arange(len(names)); w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig 9: Phase 2 — 하이퍼파라미터 튜닝 결과",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.bar(x - w/2, before, w, label="Baseline", color="#95a5a6", alpha=0.85)
ax.bar(x + w/2, after,  w, label="Tuned",    color="#e74c3c", alpha=0.85)
ax.errorbar(x + w/2, after, yerr=stds,
            fmt="none", color="black", capsize=4, linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax.set_ylabel("R² (5-Fold CV)"); ax.set_title("튜닝 전·후 R² 비교\n(오차막대=±1std)")
ax.legend(); ax.set_ylim(min(before + after) - 0.02, max(before + after) + 0.025)
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - w/2, b + 0.001, f"{b:.4f}", ha="center", fontsize=7, color="#555")
    ax.text(i + w/2, a + 0.001, f"{a:.4f}", ha="center", fontsize=7, fontweight="bold")

ax2 = axes[1]
ax2.bar(x, rmses, color="#e67e22", alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax2.set_ylabel("RMSE"); ax2.set_title("튜닝 후 RMSE 비교\n(↓낮을수록 우수)")
for bar, val in zip(ax2.patches, rmses):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
             f"{val:.4f}", ha="center", fontsize=8.5)

ax3 = axes[2]
ax3.bar(x, maes, color="#27ae60", alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax3.set_ylabel("MAE"); ax3.set_title("튜닝 후 MAE 비교\n(↓낮을수록 우수)")
for bar, val in zip(ax3.patches, maes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
             f"{val:.4f}", ha="center", fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_tuning.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 9] Tuning comparison 저장 완료")

# Fig 9b: Learning Curve (상위 3개)
top3_for_lc = sorted(tuning_results, key=lambda k: tuning_results[k]["cv5_r2"], reverse=True)[:3]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fig 9b: Learning Curve — 상위 3개 모델", fontsize=13, fontweight="bold")
lc_colors = ["#e74c3c", "#2980b9", "#27ae60"]

for ax, name, color in zip(axes, top3_for_lc, lc_colors):
    tr_sz, tr_sc, va_sc = learning_curve(
        best_estimators[name], X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="r2", n_jobs=-1,
    )
    tr_m = tr_sc.mean(axis=1); tr_s = tr_sc.std(axis=1)
    va_m = va_sc.mean(axis=1); va_s = va_sc.std(axis=1)
    ax.plot(tr_sz, tr_m, "o-", color=color, lw=2, label="Train")
    ax.fill_between(tr_sz, tr_m - tr_s, tr_m + tr_s, alpha=0.12, color=color)
    ax.plot(tr_sz, va_m, "s--", color=color, alpha=0.65, lw=2, label="Validation")
    ax.fill_between(tr_sz, va_m - va_s, va_m + va_s, alpha=0.07, color=color)
    ax.set_xlabel("Training Samples"); ax.set_ylabel("R²")
    ax.set_title(f"{name}\n5-CV R²={tuning_results[name]['cv5_r2']:.4f}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    gap = tr_m[-1] - va_m[-1]
    ax.text(0.05, 0.07, f"과적합 Gap={gap:.3f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9b_learning_curve.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 9b] Learning curve 저장 완료")

# Fig 9c: 전체 Leaderboard
all_scores = {}
for n, res in tuning_results.items():
    all_scores[f"{n} (Tuned)"] = dict(R2=res["cv5_r2"], std=res["cv5_std"], RMSE=res["cv5_rmse"])
for n, res in adv_results.items():
    all_scores[n] = dict(R2=res["cv5_r2"], std=res["cv5_std"], RMSE=res["cv5_rmse"])
all_scores["Linear Reg. (Baseline)"] = dict(R2=0.5837, std=0.0166, RMSE=0.1130)
all_scores["Bagging-DT (Baseline)"]  = dict(R2=0.6683, std=0.0271, RMSE=0.1008)

lb_df = (pd.DataFrame(all_scores).T.reset_index()
         .rename(columns={"index": "model"})
         .sort_values("R2", ascending=False).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Fig 9c: 전체 모델 Leaderboard — 최종 후보 종합 비교",
             fontsize=13, fontweight="bold")
bar_colors = []
for m in lb_df["model"]:
    if "Stacking" in m:   bar_colors.append("#9b59b6")
    elif "Voting" in m:   bar_colors.append("#8e44ad")
    elif "Gradient" in m or "HistGBM" in m: bar_colors.append("#e74c3c")
    elif "Tuned" in m:    bar_colors.append("#e67e22")
    else:                 bar_colors.append("#95a5a6")

bars = ax.barh(lb_df["model"], lb_df["R2"].astype(float), color=bar_colors, alpha=0.85)
ax.errorbar(lb_df["R2"].astype(float), range(len(lb_df)),
            xerr=lb_df["std"].astype(float),
            fmt="none", color="black", capsize=3, lw=1.5)
ax.invert_yaxis(); ax.set_xlabel("R² (5-Fold CV)", fontsize=12)
legend_elems = [
    mpatches.Patch(fc="#9b59b6", label="Stacking"),
    mpatches.Patch(fc="#e74c3c", label="최고 부스팅 (Tuned)"),
    mpatches.Patch(fc="#e67e22", label="기타 앙상블 (Tuned)"),
    mpatches.Patch(fc="#95a5a6", label="Baseline"),
]
ax.legend(handles=legend_elems, fontsize=9, loc="lower right")
for bar, val in zip(bars, lb_df["R2"].astype(float)):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9c_leaderboard.png", bbox_inches="tight", dpi=150)
plt.close()
print("[Fig 9c] Leaderboard 저장 완료")

# ═══════════════════════════════════════════════════════════════
# 6. 노드별 성능 분석
# ═══════════════════════════════════════════════════════════════
y_pred = FINAL.predict(X)
print(f"\n▶ 최종 모델 Train 성능:")
print(f"  R²={r2_score(y, y_pred):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(y, y_pred)):.4f}  "
      f"MAE={mean_absolute_error(y, y_pred):.4f}")

print("\n▶ Technology Node별 R²:")
for node in TECH_NODES:
    mask = df["technology_node"] == node
    r2_n = r2_score(y[mask], y_pred[mask])
    print(f"  {node}: R²={r2_n:.4f}  (n={mask.sum()})")

print("\n[STEP 4 완료] best_model.pkl · model_meta.json · tuning_results.csv · Fig 9/9b/9c 생성")
