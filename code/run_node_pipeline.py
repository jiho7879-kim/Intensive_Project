"""
run_node_pipeline.py  ─ 노드 집중 분석 파이프라인
══════════════════════════════════════════════════════════════════
지정한 Technology Node 데이터만으로 best model을 탐색·선정합니다.
run_full_pipeline.py의 §1~§7 흐름을 그대로 따르되,
"전체 노드 비교" 부분은 제외하고 노드 내부 최적화에 집중합니다.

【전체 파이프라인과의 차이점】
  - 데이터  : 지정 노드(예: 22nm) 데이터만 사용
  - 피처셋  : 노드 상수 컬럼(tech_encoded, is_advanced_node, node_x_cd) 제외
              → 노드 내부에서는 상수이므로 정보 없음
  - EDA     : 단일 노드 분포 분석 (이분화 비교 제외)
  - 진단    : 노드 내부 잔차·Q-Q·Learning Curve
  - §6      : 피처셋 A/B/C 비교는 노드 내 데이터 기준으로 수행
  - §7      : Phase별 추이 + 최종 파라미터 요약 (노드 간 비교 제외)

실행 예시:
    python run_node_pipeline.py --node 22nm
    python run_node_pipeline.py --node 7nm
    python run_node_pipeline.py --node 14nm --steps 1 2 3
    python run_node_pipeline.py --node 10nm --skip 1

출력:
    output_<node>/figures/   PNG Figure
    output_<node>/results/   JSON/CSV
    output_<node>/best_model_<node>.pkl
══════════════════════════════════════════════════════════════════
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns, joblib
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (cross_validate, cross_val_score,
    cross_val_predict, KFold, RandomizedSearchCV, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, BayesianRidge, TheilSenRegressor, RANSACRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
    HistGradientBoostingRegressor, StackingRegressor, VotingRegressor)
from font_setting import set_korean_font
set_korean_font()

# 1. 터미널 출력과 파일 출력을 동시에 수행하는 클래스
class Logger:
    def __init__(self, filename="pipeline.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 실시간 저장 보장

    def flush(self):
        # 파이썬의 표준 출력 요구사항을 맞추기 위한 빈 메서드
        pass
    
# ── XGBoost / LightGBM / CatBoost 동적 로드 ─────────────────
def _try_import():
    res = {}
    for key, name_real, name_alt, factory in [
        ('xgb', 'XGBoost', 'HistGBM(≈XGBoost)',
         lambda: __import__('xgboost').XGBRegressor(
             n_estimators=300, learning_rate=0.05, max_depth=4,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=42, n_jobs=-1)),
        ('lgb', 'LightGBM', 'HistGBM(≈LightGBM)',
         lambda: __import__('lightgbm').LGBMRegressor(
             n_estimators=300, learning_rate=0.05, num_leaves=31,
             subsample=0.8, colsample_bytree=0.8,
             random_state=42, n_jobs=-1, verbose=-1)),
        ('cb', 'CatBoost', 'HistGBM(≈CatBoost)',
         lambda: __import__('catboost').CatBoostRegressor(
             iterations=300, learning_rate=0.05, depth=6,
             l2_leaf_reg=3.0, random_seed=42, verbose=0)),
    ]:
        try:
            res[key] = factory(); res[f'{key}_name'] = name_real
        except ImportError:
            fallback = {
                'xgb': HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, max_depth=4,
                    l2_regularization=1.0, random_state=42),
                'lgb': HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, min_samples_leaf=10,
                    l2_regularization=0.01, random_state=42),
                'cb': HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.03, max_depth=6,
                    l2_regularization=3.0, min_samples_leaf=30, random_state=42),
            }[key]
            res[key] = fallback; res[f'{key}_name'] = name_alt
    return res

BOOST_LIBS = _try_import()

# ══════════════════════════════════════════════════════════════
# 전역 설정
# ══════════════════════════════════════════════════════════════
RANDOM_STATE  = 42
DATA_PATH     = "semiconductor_yield_forecasting_data.csv"
TARGET        = 'yield'
VALID_NODES   = ['7nm', '10nm', '14nm', '22nm', '28nm']
NODE_COLOR    = {'7nm':'#e74c3c','10nm':'#e67e22','14nm':'#f39c12',
                 '22nm':'#27ae60','28nm':'#2980b9'}
CAT_COLORS    = {'linear':'#3498db','robust':'#9b59b6','nonlinear':'#e67e22',
                 'tree':'#95a5a6','ensemble':'#e74c3c','boosting':'#c0392b'}

PROCESS_FEATURES = [
    'etch_rate','pressure','temperature','exposure_time','focus_offset','dose',
    'deposition_rate','thickness_uniformity','implant_energy','tilt_angle',
    'critical_dimension','oxide_thickness','resistivity','defect_count',
    'defect_density','vth','leakage_current','resistance',
]

# ── 노드 내에서 상수가 되는 컬럼 (피처셋에서 제외) ──────────
NODE_CONST_COLS = ['tech_encoded', 'is_advanced_node', 'node_x_cd']

# ── 설정은 parse 후 초기화 ────────────────────────────────────
OUTPUT_DIR = None
FIG_DIR    = None
RES_DIR    = None
kf         = None

def init_dirs(node: str):
    global OUTPUT_DIR, FIG_DIR, RES_DIR, kf
    OUTPUT_DIR = f"output_{node.replace('/', '_')}"
    FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
    RES_DIR    = os.path.join(OUTPUT_DIR, "results")
    for d in [OUTPUT_DIR, FIG_DIR, RES_DIR]:
        os.makedirs(d, exist_ok=True)
    # 노드 샘플 수에 맞게 Fold 수 결정 (최소 2, 최대 5)
    # 실제 샘플 수는 데이터 로드 후 업데이트
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ── 헬퍼 ──────────────────────────────────────────────────────
def banner(s): print(f"\n{'═'*65}\n  {s}\n{'═'*65}")
def norm(a):   r = a - a.min(); return r / (r.max() + 1e-10)

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → {name}.png")

def save_json(obj, name):
    with open(os.path.join(RES_DIR, f"{name}.json"), 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  → {name}.json")

def save_csv(df_, name):
    df_.to_csv(os.path.join(RES_DIR, f"{name}.csv"), index=False)
    print(f"  → {name}.csv")

def load_json(name):
    p = os.path.join(RES_DIR, f"{name}.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

# ══════════════════════════════════════════════════════════════
# 데이터 로드 — 지정 노드만 필터링
# ══════════════════════════════════════════════════════════════
def load_node_data(node: str):
    """
    전체 데이터를 로드한 뒤 지정 노드만 슬라이싱.
    피처셋 A/B/C/D를 구성하되, 노드 내 상수 컬럼(tech_encoded 등)은 제외.
    """
    df_all = pd.read_csv(DATA_PATH)

    # 노드 필터링
    df = df_all[df_all['technology_node'] == node].copy().reset_index(drop=True)
    n_samples = len(df)
    print(f"\n  노드 [{node}] 데이터: {n_samples}개")
    if n_samples < 20:
        raise ValueError(f"[{node}] 샘플 수 부족 ({n_samples}개). 최소 20개 필요.")

    # KFold 조정 (샘플 수 기반)
    global kf
    n_splits = min(5, max(2, n_samples // 15))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    print(f"  CV Folds: {n_splits} (샘플 {n_samples}개 기준)")

    # 파생 변수 생성 (전체 데이터 기준 → 노드 데이터로 제한)
    df['defect_composite']    = (df['defect_count'] * df['defect_density']).apply(np.log1p)
    df['cd_vth_product']      = df['critical_dimension'] * df['vth']
    df['uniformity_x_defect'] = df['thickness_uniformity'] * df['defect_count']

    # 추가 파생 (D용)
    df['process_stability']    = (df['etch_rate']       / (df['etch_rate'].std()       + 1e-10)
                                 + df['deposition_rate'] / (df['deposition_rate'].std() + 1e-10)
                                 + df['implant_energy']  / (df['implant_energy'].std()  + 1e-10)) / 3
    df['electrical_composite'] = df['vth'] * df['resistance'] / df['leakage_current'].clip(1e-10)
    df['cd_squared']           = df['critical_dimension'] ** 2

    # 피처셋 구성 (노드 상수 컬럼 제외)
    # A: 원본 (defect_density 포함)
    FS_A = [f for f in PROCESS_FEATURES]

    # B: 다중공선성 제거 (defect_density 제거)
    FS_B = [f for f in PROCESS_FEATURES if f != 'defect_density']

    # C: EDA + FE (노드 내 상수 제외)
    #    note: node_x_cd, is_advanced_node, tech_encoded은 노드 내에서 상수
    FS_C = FS_B + ['defect_composite', 'cd_vth_product', 'uniformity_x_defect']

    # D: C + 추가 파생
    FS_D = FS_C + ['process_stability', 'electrical_composite', 'cd_squared']

    print(f"  피처셋: A={len(FS_A)}개  B={len(FS_B)}개  C={len(FS_C)}개  D={len(FS_D)}개")
    print(f"  ※ 제외된 노드 상수 컬럼: {NODE_CONST_COLS}")

    return df, {'A': FS_A, 'B': FS_B, 'C': FS_C, 'D': FS_D}

# ══════════════════════════════════════════════════════════════
# §1  EDA — 노드 내부
# ══════════════════════════════════════════════════════════════
def section1(df, FS, node):
    banner(f"§1  EDA — [{node}] 노드 내부 분석")
    y = df[TARGET].values
    n = len(df)

    # 기초 통계
    print(f"\n  수율 통계: μ={y.mean():.4f}  σ={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}  IQR={np.percentile(y,75)-np.percentile(y,25):.4f}")

    # Shapiro-Wilk 정규성
    sw_stat, sw_p = stats.shapiro(y[:500])
    print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} "
          f"({'정규분포 가능' if sw_p>0.05 else '비정규'} α=0.05)")

    # 피처셋별 probe R²
    probe = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                       max_depth=2, random_state=RANDOM_STATE)
    fs_r2 = {}
    for nm, fset in FS.items():
        r2 = float(cross_val_score(probe, df[fset].values, y, cv=kf, scoring='r2').mean())
        fs_r2[nm] = r2
        print(f"  FS-{nm}: R²={r2:.4f}")

    # 다중공선성 (노드 내)
    corr_m = df[PROCESS_FEATURES].corr()
    high_corr = [(PROCESS_FEATURES[i], PROCESS_FEATURES[j], float(corr_m.iloc[i, j]))
                 for i in range(len(PROCESS_FEATURES))
                 for j in range(i+1, len(PROCESS_FEATURES))
                 if abs(corr_m.iloc[i, j]) > 0.8]
    print(f"  다중공선성 쌍 (|r|>0.8): {len(high_corr)}개")

    outlier_pct = {
        f: float(((df[f] < df[f].quantile(.25) - 1.5*(df[f].quantile(.75)-df[f].quantile(.25))) |
                   (df[f] > df[f].quantile(.75) + 1.5*(df[f].quantile(.75)-df[f].quantile(.25)))).mean()*100)
        for f in PROCESS_FEATURES
    }

    # ── Figure §1 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    nc = NODE_COLOR.get(node, '#3498db')
    fig.suptitle(f'§1. EDA — [{node}] 노드 내부 분석 (n={n})',
                 fontsize=15, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # A: 수율 분포
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(y, bins=min(30, n//4), color=nc, edgecolor='white', alpha=0.8, density=True)
    xr = np.linspace(y.min(), y.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, y.mean(), y.std()), 'r-', lw=2, label='Normal fit')
    ax.axvline(y.mean(), color='darkred', linestyle='--', lw=1.5, label=f'μ={y.mean():.3f}')
    ax.set_xlabel('Yield'); ax.set_ylabel('Density')
    ax.set_title(f'A. [{node}] 수율 분포\nμ={y.mean():.4f}  σ={y.std():.4f}', fontweight='bold')
    ax.legend(fontsize=8)

    # B: 피처-수율 상관 (Top 10)
    ax2 = fig.add_subplot(gs[0, 1])
    feat_corrs = {f: float(abs(stats.pearsonr(df[f], y)[0])) for f in FS['C']}
    top_feats = sorted(feat_corrs, key=feat_corrs.get, reverse=True)[:10]
    colors_c = ['#e74c3c' if feat_corrs[f] > 0.3 else '#3498db' for f in top_feats]
    ax2.barh(top_feats, [feat_corrs[f] for f in top_feats], color=colors_c, alpha=0.85)
    ax2.set_xlabel('|Pearson r|')
    ax2.set_title(f'B. 피처-Yield 상관 Top10\n(노드 [{node}] 내부)', fontweight='bold')
    ax2.invert_yaxis()
    for i, f in enumerate(top_feats):
        ax2.text(feat_corrs[f] + 0.005, i, f'{feat_corrs[f]:.3f}', va='center', fontsize=8.5)

    # C: 피처셋별 R²
    ax3 = fig.add_subplot(gs[0, 2])
    fls = list(fs_r2.keys()); fvs = list(fs_r2.values())
    bars = ax3.bar([f'FS-{k}' for k in fls], fvs,
                   color=['#95a5a6', '#3498db', '#e74c3c', '#e67e22'], alpha=0.85)
    ax3.set_ylabel('R² (CV)')
    ax3.set_title('C. 피처셋별 R²\n(노드 내 데이터 기준)', fontweight='bold')
    ax3.axhline(fvs[0], color='gray', linestyle=':', lw=1.5, label='Baseline')
    ax3.legend(fontsize=8)
    for bar, val in zip(bars, fvs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

    # D: 다중공선성 히트맵 (핵심 컬럼)
    ax4 = fig.add_subplot(gs[1, 0:2])
    key_cols = ['thickness_uniformity', 'defect_count', 'defect_density',
                'critical_dimension', 'vth', 'oxide_thickness', 'resistance', TARGET]
    key_cols = [c for c in key_cols if c in df.columns]
    mask_ = np.triu(np.ones((len(key_cols), len(key_cols)), dtype=bool), k=1)
    sns.heatmap(df[key_cols].corr(), ax=ax4, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, annot_kws={'size': 8}, mask=mask_)
    ax4.set_title(f'D. 다중공선성 진단 [{node}]', fontweight='bold')
    ax4.tick_params(axis='x', rotation=35, labelsize=8)

    # E: 이상치 비율
    ax5 = fig.add_subplot(gs[1, 2])
    out_s = pd.Series(outlier_pct).sort_values(ascending=False).head(12)
    ax5.barh(out_s.index, out_s.values,
             color=['#e74c3c' if v > 3 else '#f39c12' if v > 1 else '#2ecc71'
                    for v in out_s.values], alpha=0.85)
    ax5.axvline(3, color='red', linestyle='--', lw=1, label='3%')
    ax5.set_xlabel('이상치(%)'); ax5.set_title('E. 피처별 이상치 비율', fontweight='bold')
    ax5.legend(fontsize=8)

    save_fig(fig, f'sec1_eda_{node}')
    save_json({'node': node, 'n_samples': n, 'yield_mean': float(y.mean()),
               'yield_std': float(y.std()), 'fs_r2': fs_r2,
               'high_corr_pairs': high_corr, 'shapiro_p': float(sw_p)},
              'sec1_eda')
    return fs_r2

# ══════════════════════════════════════════════════════════════
# §2  Phase 1 — 베이스라인 스크리닝
# ══════════════════════════════════════════════════════════════
def section2(df, FS, node):
    banner(f"§2  Phase 1 — [{node}] 베이스라인 스크리닝 (23개 모델)")
    X = df[FS['C']].values; y = df[TARGET].values
    Xs = StandardScaler().fit_transform(X)

    ransac = RANSACRegressor(estimator=Ridge(alpha=1.0),
                             min_samples=0.5, random_state=RANDOM_STATE)
    xgb_name = BOOST_LIBS['xgb_name']
    lgb_name  = BOOST_LIBS['lgb_name']
    cb_name   = BOOST_LIBS['cb_name']

    MODELS = {
        # ── 선형 (5) ──────────────────────────────────────────
        'Linear Regression': (LinearRegression(),                                True,  'linear'),
        'Ridge':             (Ridge(alpha=1.0),                                   True,  'linear'),
        'Lasso':             (Lasso(alpha=0.001, max_iter=5000),                  True,  'linear'),
        'ElasticNet':        (ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), True,'linear'),
        'PLS (n=5)':         (PLSRegression(n_components=min(5, X.shape[1])),     True,  'linear'),
        # ── 이상치 강건 (5) ───────────────────────────────────
        'Huber':             (HuberRegressor(epsilon=1.35, max_iter=300),         True,  'robust'),
        'Bayesian Ridge':    (BayesianRidge(),                                    True,  'robust'),
        'TheilSen':          (TheilSenRegressor(
                                  random_state=RANDOM_STATE, max_subpopulation=500), True,'robust'),
        'Kernel Ridge':      (KernelRidge(alpha=1.0, kernel='rbf'),               True,  'robust'),
        'RANSAC+Ridge':      (ransac,                                             True,  'robust'),
        # ── 비선형 (4) ────────────────────────────────────────
        'SVR (RBF)':         (SVR(kernel='rbf', C=1.0),                           True,  'nonlinear'),
        'NuSVR':             (NuSVR(kernel='rbf', nu=0.5),                        True,  'nonlinear'),
        'KNN (k=5)':         (KNeighborsRegressor(n_neighbors=min(5, len(y)//4)), True,  'nonlinear'),
        'MLP (128-64)':      (MLPRegressor(hidden_layer_sizes=(128, 64),
                                           max_iter=500, random_state=RANDOM_STATE), True,'nonlinear'),
        # ── 단일 트리 (1) ─────────────────────────────────────
        'Decision Tree':     (DecisionTreeRegressor(max_depth=6,
                                                    random_state=RANDOM_STATE),   False, 'tree'),
        # ── 앙상블 (4) ────────────────────────────────────────
        'Bagging (DT)':      (BaggingRegressor(
                                  estimator=DecisionTreeRegressor(max_depth=5),
                                  n_estimators=50, random_state=RANDOM_STATE,
                                  n_jobs=-1),                                     False, 'ensemble'),
        'Random Forest':     (RandomForestRegressor(n_estimators=100,
                                  random_state=RANDOM_STATE, n_jobs=-1),          False, 'ensemble'),
        'Extra Trees':       (ExtraTreesRegressor(n_estimators=100,
                                  random_state=RANDOM_STATE, n_jobs=-1),          False, 'ensemble'),
        'AdaBoost':          (AdaBoostRegressor(n_estimators=100,
                                  random_state=RANDOM_STATE),                     False, 'ensemble'),
        # ── 부스팅 (4) ────────────────────────────────────────
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100,
                                  random_state=RANDOM_STATE),                     False, 'boosting'),
        xgb_name:            (BOOST_LIBS['xgb'],                                 False, 'boosting'),
        lgb_name:            (BOOST_LIBS['lgb'],                                 False, 'boosting'),
        cb_name:             (BOOST_LIBS['cb'],                                  False, 'boosting'),
    }

    print(f"\n  {'Model':<28s}  {'R² mean':>8s}  {'R² std':>7s}  {'RMSE':>7s}")
    print("  " + "─"*58)

    p1 = {}
    for name, (model, use_std, cat) in MODELS.items():
        Xu = Xs if use_std else X
        try:
            cv = cross_validate(model, Xu, y, cv=kf,
                                scoring=['r2', 'neg_mean_squared_error',
                                         'neg_mean_absolute_error'], n_jobs=-1)
            r2m  = float(cv['test_r2'].mean()); r2s = float(cv['test_r2'].std())
            rmse = float(np.sqrt(-cv['test_neg_mean_squared_error'].mean()))
            mae  = float(-cv['test_neg_mean_absolute_error'].mean())
        except Exception as e:
            print(f"  ⚠️  {name}: {e}")
            r2m = r2s = rmse = mae = 0.0
        p1[name] = dict(R2_mean=r2m, R2_std=r2s, RMSE=rmse, MAE=mae,
                        category=cat, use_std=use_std)
        print(f"  {name:<28s}  {r2m:>8.4f}  {r2s:>7.4f}  {rmse:>7.4f}")

    p1_df = (pd.DataFrame(p1).T.reset_index().rename(columns={'index': 'model'})
             .sort_values('R2_mean', ascending=False).reset_index(drop=True))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, len(p1)//2)))
    fig.suptitle(f'§2. Phase 1 — [{node}] 베이스라인 스크리닝 ({kf.n_splits}-Fold CV)',
                 fontsize=13, fontweight='bold')
    nc = NODE_COLOR.get(node, '#3498db')

    colors_p1 = [CAT_COLORS.get(p1_df.loc[p1_df['model']==m, 'category'].values[0], '#95a5a6')
                 for m in p1_df['model']]
    bars = axes[0].barh(p1_df['model'], p1_df['R2_mean'].astype(float),
                        color=colors_p1, alpha=0.85)
    axes[0].errorbar(p1_df['R2_mean'].astype(float), range(len(p1_df)),
                     xerr=p1_df['R2_std'].astype(float),
                     fmt='none', color='black', capsize=3, lw=1.5)
    axes[0].invert_yaxis()
    legend_elems = [mpatches.Patch(fc=v, label=k) for k, v in CAT_COLORS.items()]
    axes[0].legend(handles=legend_elems, fontsize=8, loc='lower right')
    axes[0].set_xlabel('R²'); axes[0].set_title('R² 비교', fontweight='bold')
    for bar, val in zip(bars, p1_df['R2_mean'].astype(float)):
        axes[0].text(val + .001, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=7.5)

    cat_avg = (p1_df.groupby('category')['R2_mean']
               .apply(lambda s: s.astype(float).mean())
               .sort_values(ascending=False))
    bars2 = axes[1].bar(cat_avg.index, cat_avg.values,
                        color=[CAT_COLORS.get(c, '#95a5a6') for c in cat_avg.index], alpha=0.85)
    axes[1].set_ylabel('평균 R²'); axes[1].set_title('카테고리별 평균 R²', fontweight='bold')
    for bar, val in zip(bars2, cat_avg.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

    save_fig(fig, f'sec2_baseline_{node}')
    save_csv(p1_df, 'phase1_results')
    save_json({'results': p1, 'cat_avg': cat_avg.to_dict(),
               'node': node, 'n_samples': len(y)}, 'sec2_phase1')
    return p1

# ══════════════════════════════════════════════════════════════
# §3  Phase 2 — 하이퍼파라미터 튜닝
# ══════════════════════════════════════════════════════════════
def section3(df, FS, p1_results, node):
    banner(f"§3  Phase 2 — [{node}] 하이퍼파라미터 튜닝")
    X = df[FS['C']].values; y = df[TARGET].values
    n = len(y)

    candidates = ['Gradient Boosting', BOOST_LIBS['xgb_name'],
                  BOOST_LIBS['lgb_name'], BOOST_LIBS['cb_name'],
                  'Random Forest', 'AdaBoost']
    baseline_r2 = {k: p1_results[k]['R2_mean']
                   for k in candidates if k in p1_results}
    print(f"\n  Baseline R²:")
    for k, v in sorted(baseline_r2.items(), key=lambda x: -x[1]):
        print(f"    {k:<32s}: {v:.4f}")

    # 소규모 샘플에 맞게 n_iter 조정
    n_iter_scale = max(0.3, min(1.0, n / 200))

    SEARCH_SPACES = {
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [2, 3, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 0.7, 0.9],
            },
            'n_iter': max(10, int(30 * n_iter_scale))
        },
    }

    # XGB/LGB/CB 대안 (HistGBM)
    for key, extra_p, n_it in [
        ('xgb', {'max_iter': [100, 200], 'learning_rate': [0.05, 0.1, 0.15],
                 'max_depth': [3, 5, None], 'min_samples_leaf': [10, 20, 30],
                 'l2_regularization': [0.0, 0.1, 1.0]}, 15),
        ('lgb', {'max_iter': [150, 300], 'learning_rate': [0.03, 0.05, 0.08],
                 'max_depth': [None, 5], 'min_samples_leaf': [5, 10, 20],
                 'l2_regularization': [0.0, 0.01]}, 12),
        ('cb',  {'max_iter': [200, 300], 'learning_rate': [0.01, 0.03, 0.05],
                 'max_depth': [4, 6], 'min_samples_leaf': [20, 30],
                 'l2_regularization': [1.0, 3.0]}, 10),
    ]:
        nm = BOOST_LIBS[f'{key}_name']
        if nm in p1_results:
            if '≈' in nm or '대안' in nm or 'Hist' in nm:
                SEARCH_SPACES[nm] = {
                    'model': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                    'params': extra_p,
                    'n_iter': max(8, int(n_it * n_iter_scale))
                }

    if 'Random Forest' in p1_results:
        SEARCH_SPACES['Random Forest'] = {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 8, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2', 0.7],
            },
            'n_iter': max(10, int(25 * n_iter_scale))
        }

    tuning_results = {}; best_estimators = {}
    for name, cfg in SEARCH_SPACES.items():
        bl = baseline_r2.get(name, 0.0)
        print(f"\n  [{name}] Baseline={bl:.4f} (n_iter={cfg['n_iter']})...",
              end=' ', flush=True)
        t0 = time.time()
        # 소규모 데이터: RandomizedSearchCV의 inner CV도 조정
        inner_cv = min(3, kf.n_splits)
        search = RandomizedSearchCV(cfg['model'], cfg['params'],
                                    n_iter=cfg['n_iter'], cv=inner_cv,
                                    scoring='r2', random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X, y)
        cv5 = cross_validate(search.best_estimator_, X, y, cv=kf,
                             scoring=['r2', 'neg_mean_squared_error',
                                      'neg_mean_absolute_error'], n_jobs=-1)
        r2m  = float(cv5['test_r2'].mean()); r2s = float(cv5['test_r2'].std())
        rmse = float(np.sqrt(-cv5['test_neg_mean_squared_error'].mean()))
        mae  = float(-cv5['test_neg_mean_absolute_error'].mean())
        elapsed = time.time() - t0
        tuning_results[name] = dict(
            baseline_r2=bl, search_best_r2=float(search.best_score_),
            cv5_r2=r2m, cv5_std=r2s, cv5_rmse=rmse, cv5_mae=mae,
            best_params=search.best_params_, elapsed=elapsed
        )
        best_estimators[name] = search.best_estimator_
        search.best_estimator_.fit(X, y)
        print(f"search={search.best_score_:.4f}  5CV={r2m:.4f}±{r2s:.4f}  [{elapsed:.0f}s]")
        print(f"    best_params: {search.best_params_}")

    save_json(tuning_results, 'sec3_tuning_results')
    joblib.dump(best_estimators, os.path.join(OUTPUT_DIR, 'all_tuned_models.pkl'))

    # Figure 튜닝 전후 비교
    names_ = list(SEARCH_SPACES.keys())
    before = [tuning_results[n]['baseline_r2'] for n in names_]
    after  = [tuning_results[n]['cv5_r2']      for n in names_]
    stds_  = [tuning_results[n]['cv5_std']      for n in names_]
    x = np.arange(len(names_)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'§3. [{node}] Phase 2 — 하이퍼파라미터 튜닝 결과', fontsize=13, fontweight='bold')
    axes[0].bar(x - w/2, before, w, label='Baseline', color='#95a5a6', alpha=0.85)
    axes[0].bar(x + w/2, after,  w, label='Tuned',    color='#e74c3c', alpha=0.85)
    axes[0].errorbar(x + w/2, after, yerr=stds_,
                     fmt='none', color='black', capsize=4, lw=1.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(' ', '\n') for n in names_], fontsize=7)
    axes[0].set_ylabel('R²'); axes[0].set_title('튜닝 전·후 R² 비교'); axes[0].legend()
    if before and after:
        axes[0].set_ylim(min(before + after) - 0.03, max(before + after) + 0.03)
    for i, (b, a) in enumerate(zip(before, after)):
        axes[0].text(i - w/2, b + 0.001, f'{b:.4f}', ha='center', fontsize=6.5, color='#555')
        axes[0].text(i + w/2, a + 0.001, f'{a:.4f}', ha='center', fontsize=6.5, fontweight='bold')

    # Learning Curve (1위 모델)
    top1 = max(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'])
    try:
        tr_sz, tr_sc, va_sc = learning_curve(
            best_estimators[top1], X, y,
            train_sizes=np.linspace(0.2, 1.0, min(6, kf.n_splits)),
            cv=kf, scoring='r2', n_jobs=-1)
        tr_m = tr_sc.mean(axis=1); tr_s = tr_sc.std(axis=1)
        va_m = va_sc.mean(axis=1); va_s = va_sc.std(axis=1)
        axes[1].plot(tr_sz, tr_m, 'o-', color='#e74c3c', lw=2, label='Train')
        axes[1].fill_between(tr_sz, tr_m - tr_s, tr_m + tr_s, alpha=0.12, color='#e74c3c')
        axes[1].plot(tr_sz, va_m, 's--', color='#2980b9', lw=2, label='Validation')
        axes[1].fill_between(tr_sz, va_m - va_s, va_m + va_s, alpha=0.07, color='#2980b9')
        axes[1].set_xlabel('Training Samples'); axes[1].set_ylabel('R²')
        axes[1].set_title(f'Learning Curve — {top1}', fontweight='bold')
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
        if len(tr_m) > 0:
            gap = tr_m[-1] - va_m[-1]
            axes[1].text(0.05, 0.07, f'과적합 Gap={gap:.3f}',
                         transform=axes[1].transAxes, fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Learning Curve 생성 실패\n{e}',
                     ha='center', va='center', transform=axes[1].transAxes)

    save_fig(fig, f'sec3_tuning_{node}')
    return tuning_results, best_estimators

# ══════════════════════════════════════════════════════════════
# §4  Phase 3 — 앙상블
# ══════════════════════════════════════════════════════════════
def section4(df, FS, tuning_results, best_estimators, node):
    banner(f"§4  Phase 3 — [{node}] 고급 앙상블")
    X = df[FS['C']].values; y = df[TARGET].values
    top3 = sorted(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'], reverse=True)[:3]
    top3_est = [(n.replace(' ', '_').replace('(', '').replace(')', '')
                  .replace('≈', '').replace('/', '')[:18],
                 best_estimators[n]) for n in top3]
    print(f"  기반 학습기 Top3: {top3}")

    adv = {}
    for aname, amodel in [
        ('Voting (Top3)',
         VotingRegressor(estimators=top3_est, n_jobs=-1)),
        ('Stacking (Top3→Ridge)',
         StackingRegressor(estimators=top3_est,
                           final_estimator=Ridge(alpha=1.0),
                           cv=min(3, kf.n_splits), n_jobs=-1)),
    ]:
        try:
            cv5 = cross_validate(amodel, X, y, cv=kf,
                                 scoring=['r2', 'neg_mean_squared_error',
                                          'neg_mean_absolute_error'], n_jobs=-1)
            r2m = float(cv5['test_r2'].mean())
            adv[aname] = dict(
                cv5_r2=r2m,
                cv5_std=float(cv5['test_r2'].std()),
                cv5_rmse=float(np.sqrt(-cv5['test_neg_mean_squared_error'].mean())),
                cv5_mae=float(-cv5['test_neg_mean_absolute_error'].mean())
            )
            print(f"  {aname:<30s}: R²={r2m:.4f}±{adv[aname]['cv5_std']:.4f}")
        except Exception as e:
            print(f"  ⚠️  {aname}: {e}")

    # Leaderboard
    all_sc = {}
    for n, res in tuning_results.items():
        all_sc[f'{n} (Tuned)'] = dict(R2=res['cv5_r2'], std=res['cv5_std'], RMSE=res['cv5_rmse'])
    for n, res in adv.items():
        all_sc[n] = dict(R2=res['cv5_r2'], std=res['cv5_std'], RMSE=res['cv5_rmse'])
    lb = (pd.DataFrame(all_sc).T.reset_index().rename(columns={'index': 'model'})
          .sort_values('R2', ascending=False))

    fig, ax = plt.subplots(figsize=(10, max(5, len(lb)//2+2)))
    fig.suptitle(f'§4. [{node}] 전체 Leaderboard', fontsize=13, fontweight='bold')
    bar_colors = []
    for m in lb['model']:
        if 'Stacking' in m or 'Voting' in m: bar_colors.append('#9b59b6')
        elif 'Gradient Boosting' in m:        bar_colors.append('#e74c3c')
        elif '≈' in m or 'Hist' in m:        bar_colors.append('#c0392b')
        elif 'Tuned' in m:                    bar_colors.append('#e67e22')
        else:                                  bar_colors.append('#95a5a6')
    bars = ax.barh(lb['model'], lb['R2'].astype(float), color=bar_colors, alpha=0.85)
    ax.errorbar(lb['R2'].astype(float), range(len(lb)),
                xerr=lb['std'].astype(float),
                fmt='none', color='black', capsize=3, lw=1.5)
    ax.invert_yaxis(); ax.set_xlabel(f'R² ({kf.n_splits}-Fold CV)', fontsize=11)
    for bar, val in zip(bars, lb['R2'].astype(float)):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8.5, fontweight='bold')

    save_fig(fig, f'sec4_leaderboard_{node}')
    save_json(adv, 'sec4_adv_results')
    save_csv(lb, 'sec4_leaderboard')
    return adv

# ══════════════════════════════════════════════════════════════
# §5  최종 모델 진단
# ══════════════════════════════════════════════════════════════
def section5(df, FS, tuning_results, best_estimators, adv_results, node):
    banner(f"§5  최종 모델 진단 — [{node}]")
    X = df[FS['C']].values; y = df[TARGET].values
    nc = NODE_COLOR.get(node, '#3498db')

    best_name = max(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'])
    FINAL = best_estimators[best_name]
    FINAL.fit(X, y)
    y_pred    = FINAL.predict(X)
    residuals = y - y_pred
    train_r2  = r2_score(y, y_pred)
    print(f"  최종 모델: {best_name}")
    print(f"  Train R²={train_r2:.4f}  CV R²={tuning_results[best_name]['cv5_r2']:.4f}")

    joblib.dump(FINAL, os.path.join(OUTPUT_DIR, f'best_model_{node}.pkl'))

    # 잔차 분석
    _, p_norm = stats.shapiro(residuals[:500])
    print(f"  잔차: μ={residuals.mean():.4f}  σ={residuals.std():.4f}  "
          f"Shapiro p={p_norm:.4f}")

    # Figure §5 — 노드 내 진단 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'§5. 최종 모델 진단 — [{node}] | {best_name}',
                 fontsize=13, fontweight='bold')

    # A: Pred vs Actual
    ax = axes[0, 0]
    ax.scatter(y, y_pred, alpha=0.5, s=20, color=nc, label=node)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.5, label='Perfect')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'A. Pred vs Actual\n(Train R²={train_r2:.3f}  CV R²={tuning_results[best_name]["cv5_r2"]:.3f})',
                 fontweight='bold')
    ax.legend(fontsize=8)

    # B: 잔차 분석
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, s=15, color=nc)
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.axhline(1.96 * residuals.std(), color='orange', linestyle=':', lw=1, label='±1.96σ')
    ax.axhline(-1.96 * residuals.std(), color='orange', linestyle=':', lw=1)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residual')
    ax.set_title('B. 잔차 분석', fontweight='bold'); ax.legend(fontsize=8)

    # C: Q-Q Plot
    ax = axes[1, 0]
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title('C. Q-Q Plot', fontweight='bold')
    ax.get_lines()[0].set(markersize=4, alpha=0.6)

    # D: 잔차 분포
    ax = axes[1, 1]
    ax.hist(residuals, bins=min(30, len(residuals)//4),
            color=nc, edgecolor='white', alpha=0.75, density=True)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()), 'r-', lw=2)
    ax.set_xlabel('Residual'); ax.set_ylabel('Density')
    ax.set_title(f'D. 잔차 분포\nμ={residuals.mean():.4f}  σ={residuals.std():.4f}',
                 fontweight='bold')
    ax.text(0.03, 0.90, f'Shapiro-Wilk p={p_norm:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    save_fig(fig, f'sec5_diagnostics_{node}')
    return FINAL, best_name

# ══════════════════════════════════════════════════════════════
# §6  EDA 피처셋 비교 (노드 내)
# ══════════════════════════════════════════════════════════════
def section6(df, FS, tuning_results, best_estimators, node):
    banner(f"§6  EDA 피처셋 반영 효과 비교 — [{node}]")
    y = df[TARGET].values
    best_name = max(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'])
    bp_raw = tuning_results[best_name]['best_params']

    if isinstance(best_estimators[best_name], GradientBoostingRegressor):
        cmp_model = GradientBoostingRegressor(**bp_raw, random_state=RANDOM_STATE)
    else:
        cmp_model = HistGradientBoostingRegressor(**bp_raw, random_state=RANDOM_STATE)

    COMP = {
        'Best Model': cmp_model,
        'RF': RandomForestRegressor(n_estimators=100, max_depth=8,
                                    random_state=RANDOM_STATE, n_jobs=-1),
    }

    comp_table = []
    for mname, model in COMP.items():
        row = {'Model': mname}
        for fsn, fset in [('A-원본', FS['A']), ('B-EDA제거', FS['B']), ('C-EDA+FE★', FS['C'])]:
            r2 = float(cross_val_score(model, df[fset].values, y, cv=kf, scoring='r2').mean())
            row[fsn] = r2
        comp_table.append(row)
    comp_df = pd.DataFrame(comp_table).set_index('Model')

    print(f"\n  피처셋별 R² [{node}]:")
    print(comp_df.to_string())

    delta_B = comp_df.iloc[:, 1] - comp_df.iloc[:, 0]
    delta_C = comp_df.iloc[:, 2] - comp_df.iloc[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'§6. [{node}] EDA 피처셋 반영 효과', fontsize=13, fontweight='bold')

    sns.heatmap(comp_df.astype(float), ax=axes[0], annot=True, fmt='.4f',
                cmap='RdYlGn', linewidths=0.5,
                annot_kws={'size': 11, 'fontweight': 'bold'},
                vmin=comp_df.values.astype(float).min() - 0.005,
                vmax=comp_df.values.astype(float).max() + 0.005)
    axes[0].set_title('R² (CV)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15, labelsize=9)

    x_d = np.arange(len(comp_df)); wd = 0.3
    axes[1].bar(x_d - wd/2, delta_B.values.astype(float) * 1000, wd,
                label='B-A', color='#3498db', alpha=0.85)
    axes[1].bar(x_d + wd/2, delta_C.values.astype(float) * 1000, wd,
                label='C-A', color='#e74c3c', alpha=0.85)
    axes[1].set_xticks(x_d); axes[1].set_xticklabels(comp_df.index, fontsize=10)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('R² 개선량 (×10⁻³)')
    axes[1].set_title('Baseline(A) 대비 개선량', fontweight='bold')
    axes[1].legend(fontsize=9)

    save_fig(fig, f'sec6_feature_compare_{node}')

# ══════════════════════════════════════════════════════════════
# §7  최종 요약
# ══════════════════════════════════════════════════════════════
def section7(df, FS, tuning_results, best_estimators, adv_results, FINAL, node):
    banner(f"§7  최종 선정 요약 — [{node}]")
    X = df[FS['C']].values; y = df[TARGET].values
    nc = NODE_COLOR.get(node, '#3498db')

    best_name = max(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'])
    tr = tuning_results[best_name]

    print(f"\n  ★ Best Model: {best_name}")
    print(f"  CV R²={tr['cv5_r2']:.4f}  std={tr['cv5_std']:.4f}")
    print(f"  RMSE={tr['cv5_rmse']:.4f}  MAE={tr['cv5_mae']:.4f}")
    print(f"  best_params: {tr['best_params']}")

    # XAI — Permutation + Pearson (노드 내)
    FINAL.fit(X, y)
    print("\n  XAI: Permutation Importance 계산 중...", end=' ', flush=True)
    perm = permutation_importance(FINAL, X, y, n_repeats=10,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    perm_n = norm(np.clip(perm.importances_mean, 0, None))
    pear_n = norm(np.array([abs(stats.pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]))
    print("완료")

    has_fi = hasattr(FINAL, 'feature_importances_')
    if has_fi:
        fi_n = norm(FINAL.feature_importances_)
        ens = (fi_n + perm_n + pear_n) / 3
    else:
        ens = (perm_n + pear_n) / 2

    xai_df = (pd.DataFrame({'feature': FS['C'], 'Permutation': perm_n,
                             'Pearson': pear_n, 'Ensemble': ens})
              .sort_values('Ensemble', ascending=False).reset_index(drop=True))
    if has_fi:
        xai_df.insert(1, 'BuiltIn', fi_n[np.argsort(-ens)])
    xai_df['rank'] = range(1, len(xai_df) + 1)

    print(f"\n  [{node}] XAI 상위 피처:")
    for _, row in xai_df.head(8).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:<24s}  {row['Ensemble']:.4f}  "
              f"{'█' * int(row['Ensemble'] * 25)}")

    # Phase 추이 (노드 내)
    d_ph1 = load_json('sec2_phase1')
    p1_best = max((v.get('R2_mean', 0) for v in
                   (d_ph1.get('results', {}) if d_ph1 else {}).values()), default=0)
    p2_best = tr['cv5_r2']
    p3_best = max((r['cv5_r2'] for r in adv_results.values()), default=p2_best) if adv_results else p2_best

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'§7. [{node}] Best Model 선정 — 전 과정 요약',
                 fontsize=14, fontweight='bold', y=1.01)
    gs7 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.42)

    # A: Phase 추이
    ax = fig.add_subplot(gs7[0, 0])
    phases = ['Phase1\nBaseline', 'Phase2\nTuned', 'Phase3\nAdvanced']
    p_best = [p1_best, p2_best, p3_best]
    ax.plot(phases, p_best, 'o-', color=nc, lw=2.5, markersize=12, zorder=3)
    ax.fill_between(phases, p_best, [v - 0.01 for v in p_best], alpha=0.1, color=nc)
    for ph, val in zip(phases, p_best):
        ax.annotate(f'R²={val:.4f}', (ph, val),
                    textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=nc, alpha=0.9))
    if p_best:
        ax.set_ylim(min(p_best) - 0.04, max(p_best) + 0.06)
    ax.set_ylabel('R²'); ax.grid(alpha=0.3)
    ax.set_title(f'A. Phase별 최고 성능 [{node}]', fontweight='bold')

    # B: 파라미터 요약
    ax2 = fig.add_subplot(gs7[0, 1:])
    ax2.axis('off')
    param_text = '\n'.join([f"  {k}: {v}" for k, v in tr['best_params'].items()])
    ax2.text(0.05, 0.5,
             f"★ Best Model: {best_name}\n"
             f"노드: [{node}]  (n={len(y)})\n\n"
             f"최적 파라미터:\n{param_text}\n\n"
             f"CV R²  = {tr['cv5_r2']:.4f} ±{tr['cv5_std']:.4f}\n"
             f"CV RMSE= {tr['cv5_rmse']:.4f}\n"
             f"CV MAE = {tr['cv5_mae']:.4f}",
             transform=ax2.transAxes, fontsize=10.5, va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax2.set_title('B. 최종 파라미터 요약', fontweight='bold')

    # C: XAI 상위 피처
    ax3 = fig.add_subplot(gs7[1, :2])
    top_n = min(12, len(xai_df))
    xai_top = xai_df.head(top_n)
    bars = ax3.barh(xai_top['feature'], xai_top['Ensemble'],
                    color=[nc if i < 3 else '#95a5a6' for i in range(top_n)], alpha=0.85)
    ax3.invert_yaxis(); ax3.set_xlabel('Ensemble Score')
    ax3.set_title(f'C. [{node}] XAI 유의인자 순위 (Perm + Pearson)', fontweight='bold')
    for bar, val in zip(bars, xai_top['Ensemble']):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8.5)

    # D: Leaderboard Top8
    ax4 = fig.add_subplot(gs7[1, 2])
    ax4.axis('off')
    lb_data = [(n, f"{res['cv5_r2']:.4f}", f"±{res['cv5_std']:.4f}")
               for n, res in sorted(tuning_results.items(),
                                    key=lambda x: -x[1]['cv5_r2'])[:8]]
    tbl = ax4.table(cellText=[[i+1, nm[:18], r2, std] for i, (nm, r2, std) in enumerate(lb_data)],
                    colLabels=['#', 'Model', 'R²', 'std'],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.3)
    # 1위 강조
    for j in range(4):
        tbl[1, j].set_facecolor('#FFF3CD')
    ax4.set_title('D. Leaderboard Top8', fontweight='bold')

    save_fig(fig, f'sec7_summary_{node}')
    save_csv(xai_df, f'xai_results_{node}')
    save_json({'node': node, 'n_samples': len(y),
               'model_name': best_name,
               'cv5_r2': tr['cv5_r2'], 'cv5_std': tr['cv5_std'],
               'cv5_rmse': tr['cv5_rmse'], 'cv5_mae': tr['cv5_mae'],
               'best_params': tr['best_params'],
               'features': FS['C'],
               'xai_top5': xai_df.head(5)[['feature', 'Ensemble']].to_dict('records'),
               'phase_r2': {'phase1': p1_best, 'phase2': p2_best, 'phase3': p3_best}},
              'model_meta')
    return xai_df

# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
def main():
    global DATA_PATH
    parser = argparse.ArgumentParser(
        description='노드 집중 분석 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_node_pipeline.py --node 22nm
  python run_node_pipeline.py --node 7nm  --steps 1 2 3
  python run_node_pipeline.py --node 14nm --skip 1
  python run_node_pipeline.py --list-nodes
        """
    )
    parser.add_argument('--node',   type=str, help=f'분석할 노드 ({", ".join(VALID_NODES)})')
    parser.add_argument('--steps',  nargs='+', type=int, help='실행할 단계 번호')
    parser.add_argument('--skip',   nargs='+', type=int, help='건너뛸 단계 번호')
    parser.add_argument('--list-nodes', action='store_true', help='사용 가능한 노드 목록')
    parser.add_argument('--data',   type=str, default=DATA_PATH, help='데이터 CSV 경로')
    args = parser.parse_args()

    if args.list_nodes:
        print(f"사용 가능한 노드: {', '.join(VALID_NODES)}")
        sys.exit(0)

    if not args.node:
        parser.error("--node 인자가 필요합니다. (예: --node 22nm)")

    node = args.node.strip()
    if node not in VALID_NODES:
        parser.error(f"'{node}'는 유효하지 않은 노드입니다. 사용 가능: {VALID_NODES}")

    DATA_PATH = args.data
    sys.stdout = Logger(f"execution_log_{node}.txt")
    to_run = sorted(args.steps if args.steps else [1, 2, 3, 4, 5, 6, 7])
    if args.skip: to_run = [s for s in to_run if s not in args.skip]

    desc = {
        1: '§1 EDA (노드 내부)',
        2: '§2 Baseline (23개)',
        3: '§3 Tuning',
        4: '§4 Ensemble',
        5: '§5 진단',
        6: '§6 EDA 비교',
        7: '§7 요약+XAI',
    }

    print("═" * 65)
    print(f"  노드 집중 분석 파이프라인  —  [{node}]")
    print(f"  XGBoost : {BOOST_LIBS['xgb_name']}")
    print(f"  LightGBM: {BOOST_LIBS['lgb_name']}")
    print(f"  CatBoost: {BOOST_LIBS['cb_name']}")
    print("═" * 65)
    print(f"실행 단계: {[desc.get(s, str(s)) for s in to_run]}")

    init_dirs(node)

    # 데이터 로드
    print(f"\n▶ 데이터 로드 [{node}]...")
    df, FS = load_node_data(node)
    print(f"  사용 피처셋: A={len(FS['A'])}  B={len(FS['B'])}  C={len(FS['C'])}  D={len(FS['D'])}")

    # 중간 결과 복원 (독립 실행 지원)
    p1_results      = None
    tuning_results  = None
    best_estimators = None
    adv_results     = None
    FINAL           = None

    if 2 not in to_run:
        d = load_json('sec2_phase1')
        if d: p1_results = d.get('results')
    if 3 not in to_run:
        d = load_json('sec3_tuning_results')
        if d: tuning_results = d
    if 4 not in to_run:
        d = load_json('sec4_adv_results')
        if d: adv_results = d

    mp = os.path.join(OUTPUT_DIR, 'all_tuned_models.pkl')
    if 3 not in to_run and os.path.exists(mp):
        best_estimators = joblib.load(mp)

    bp = os.path.join(OUTPUT_DIR, f'best_model_{node}.pkl')
    if 5 not in to_run and os.path.exists(bp):
        FINAL = joblib.load(bp)

    t_total = time.time()
    for step in to_run:
        t0 = time.time()
        try:
            if   step == 1:
                section1(df, FS, node)
            elif step == 2:
                p1_results = section2(df, FS, node)
            elif step == 3:
                if not p1_results:
                    print("⚠️  Step2 결과 없음. --steps 2 3 으로 실행하세요.")
                    break
                tuning_results, best_estimators = section3(df, FS, p1_results, node)
            elif step == 4:
                if not (tuning_results and best_estimators):
                    print("⚠️  Step3 결과 없음."); break
                adv_results = section4(df, FS, tuning_results, best_estimators, node)
            elif step == 5:
                if not (tuning_results and best_estimators):
                    print("⚠️  Step3 결과 없음."); break
                FINAL, _ = section5(df, FS, tuning_results, best_estimators,
                                    adv_results or {}, node)
            elif step == 6:
                if not (tuning_results and best_estimators):
                    print("⚠️  Step3 결과 없음."); break
                section6(df, FS, tuning_results, best_estimators, node)
            elif step == 7:
                if not FINAL:
                    if os.path.exists(bp): FINAL = joblib.load(bp)
                    else:
                        print("⚠️  best_model 없음. Step5 먼저 실행하세요.")
                        break
                if not tuning_results:
                    print("⚠️  튜닝 결과 없음. Step3 먼저 실행하세요.")
                    break
                section7(df, FS, tuning_results, best_estimators or {},
                         adv_results or {}, FINAL, node)

            print(f"  ✅ Step {step} 완료 ({time.time()-t0:.0f}s)")

        except Exception as e:
            print(f"  ❌ Step {step} 오류: {e}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'═'*65}")
    print(f"  [{node}] 분석 완료 ({elapsed:.0f}s)")
    print(f"  출력 디렉토리: {OUTPUT_DIR}/")
    print(f"  figures : {len(os.listdir(FIG_DIR))}개")
    print(f"  results : {len(os.listdir(RES_DIR))}개")
    if os.path.exists(bp):
        print(f"  best_model_{node}.pkl : ✅")
    print(f"{'═'*65}")

if __name__ == '__main__':
    main()
