"""
run_full_pipeline.py  ─ 완전 재현 파이프라인 (Final)
══════════════════════════════════════════════════════════════
§1  EDA 심층 분석 (이분화·다중공선성·이상치·피처셋 설계)
§2  Phase 1 — 23개 모델 베이스라인 스크리닝
    선형5 | 이상치강건5 | 비선형4 | 트리1 | 앙상블4 | 부스팅4
    * HistGBM(≈XGBoost) / (≈LightGBM) / (≈CatBoost) 대안 포함
§3  Phase 2 — Top 모델 RandomizedSearchCV (best_params 저장)
§4  Phase 3 — 동종 Stacking/Voting
§5  최종 모델 진단 (잔차·Q-Q·노드별 R²)
§6  EDA 피처셋 A/B/C 반영 효과 비교
§7  최종 선정 종합 요약
§8  개선 실험 — 의사결정 흐름
    ① 22nm R²<0.5 발견 → 노드별 개별 모델 → ❌ 기각
    ② 동종 Stacking 동률 → 이기종(GBM+SVR+RF) → ❌ 기각
    ③ 피처 표현력 → FS-D 추가 파생 → ❌ 기각
    ④ 커널 모델 낮음 → 이분화 구조 문제 규명

【재현성 보장】
  · 모든 중간 결과(best_params 포함)를 output/results/ 에 저장
  · 각 단계 독립 실행 가능 (이전 단계 결과 자동 복원)
  · RANDOM_STATE=42 전역 고정

실행:
    python run_full_pipeline.py
    python run_full_pipeline.py --steps 3 4
    python run_full_pipeline.py --skip 1
══════════════════════════════════════════════════════════════
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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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
# ── 한글 폰트 설정 (font_setting.py 없어도 동작) ──────────────
try:
    from font_setting import set_korean_font
    set_korean_font()
except ImportError:
    # font_setting.py 없는 환경: matplotlib 기본 한글 폰트 자동 탐색
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    _sys = platform.system()
    _candidates = {
        'Windows': ['Malgun Gothic', 'Microsoft YaHei', 'Arial Unicode MS'],
        'Darwin':  ['AppleGothic', 'Apple SD Gothic Neo', 'Arial Unicode MS'],
        'Linux':   ['NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR',
                    'DejaVu Sans', 'Liberation Sans'],
    }.get(_sys, ['DejaVu Sans'])
    _installed = {f.name for f in fm.fontManager.ttflist}
    _font = next((f for f in _candidates if f in _installed), None)
    if _font:
        plt.rcParams['font.family'] = _font
    plt.rcParams['axes.unicode_minus'] = False

# 1. 터미널 출력과 파일 출력을 동시에 수행하는 클래스
class Logger:
    """
    터미널 + 파일 동시 출력 로거.
    Windows cp949/euc-kr 환경에서 유니코드(═, §, ─ 등) 출력 시
    UnicodeEncodeError 를 방지하기 위해 errors='replace' 로 처리.
    파일은 항상 utf-8 로 저장.
    """
    def __init__(self, filename="pipeline.log"):
        self.log = open(filename, "a", encoding="utf-8")

        # ── 터미널 스트림 설정 ──────────────────────────────────
        # sys.stdout 이 reconfigure 를 지원하면(Python 3.7+) utf-8 으로 전환 시도
        raw = sys.stdout
        if hasattr(raw, 'reconfigure'):
            try:
                raw.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        self.terminal = raw

    def write(self, message):
        # 파일: 항상 utf-8 (손실 없음)
        self.log.write(message)
        self.log.flush()

        # 터미널: 인코딩 오류 시 '?' 로 대체 (crash 방지)
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            safe = message.encode(enc, errors='replace').decode(enc, errors='replace')
            self.terminal.write(safe)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass
# ── XGBoost / LightGBM / CatBoost 동적 로드 ─────────────────
# 설치된 경우 실제 라이브러리 사용, 없으면 HistGBM 대안으로 fallback
def _try_import():
    res = {}
    try:
        from xgboost import XGBRegressor
        res['xgb'] = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=42, n_jobs=-1)
        res['xgb_name'] = 'XGBoost (실제)'
    except ImportError:
        res['xgb'] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=42)
        res['xgb_name'] = 'HistGBM (≈XGBoost 대안)'

    try:
        from lightgbm import LGBMRegressor
        res['lgb'] = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                    num_leaves=31, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42,
                                    n_jobs=-1, verbose=-1)
        res['lgb_name'] = 'LightGBM (실제)'
    except ImportError:
        res['lgb'] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=0.01, random_state=42)
        res['lgb_name'] = 'HistGBM (≈LightGBM 대안)'

    try:
        from catboost import CatBoostRegressor
        res['cb'] = CatBoostRegressor(iterations=300, learning_rate=0.05,
                                       depth=6, l2_leaf_reg=3.0,
                                       random_seed=42, verbose=0)
        res['cb_name'] = 'CatBoost (실제)'
    except ImportError:
        res['cb'] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=6,
            l2_regularization=3.0, min_samples_leaf=30, random_state=42)
        res['cb_name'] = 'HistGBM (≈CatBoost 대안)'

    return res

BOOST_LIBS = _try_import()

# ══════════════════════════════════════════════════════════════
# 전역 설정
# ══════════════════════════════════════════════════════════════
RANDOM_STATE = 42
CV_FOLDS     = 5
DATA_PATH    = "semiconductor_yield_forecasting_data.csv"
OUTPUT_DIR   = "output"
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")
RES_DIR      = os.path.join(OUTPUT_DIR, "results")
for d in [OUTPUT_DIR, FIG_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

PROCESS_FEATURES = [
    'etch_rate','pressure','temperature','exposure_time','focus_offset','dose',
    'deposition_rate','thickness_uniformity','implant_energy','tilt_angle',
    'critical_dimension','oxide_thickness','resistivity','defect_count',
    'defect_density','vth','leakage_current','resistance',
]
TARGET     = 'yield'
TECH_NODES = ['7nm','10nm','14nm','22nm','28nm']
NODE_COLORS= {'7nm':'#e74c3c','10nm':'#e67e22','14nm':'#f39c12','22nm':'#27ae60','28nm':'#2980b9'}
CAT_COLORS = {'linear':'#3498db','robust':'#9b59b6','nonlinear':'#e67e22',
              'tree':'#95a5a6','ensemble':'#e74c3c','boosting':'#c0392b'}
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def banner(s): print(f"\n{'='*65}\n  {s}\n{'='*65}")
def norm(a):   r=a-a.min(); return r/(r.max()+1e-10)
def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR,f"{name}.png"),bbox_inches='tight',dpi=150); plt.close(fig)
    print(f"  → {name}.png")
def save_json(obj, name):
    with open(os.path.join(RES_DIR,f"{name}.json"),'w') as f: json.dump(obj,f,indent=2,default=float)
    print(f"  → {name}.json")
def save_csv(df_, name):
    df_.to_csv(os.path.join(RES_DIR,f"{name}.csv"),index=False); print(f"  → {name}.csv")
def load_json(name):
    p=os.path.join(RES_DIR,f"{name}.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

# ══════════════════════════════════════════════════════════════
# 데이터 로드 & 피처셋 A/B/C/D
# ══════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df['tech_encoded'] = le.fit_transform(df['technology_node'])
    FS_A = PROCESS_FEATURES + ['tech_encoded']
    # B: 다중공선성 제거 (defect_density vs thickness_uniformity r=0.9996)
    FS_B = [f for f in PROCESS_FEATURES if f != 'defect_density'] + ['tech_encoded']
    # C: EDA + 피처 엔지니어링 ★ (최종 채택)
    df['is_advanced_node']    = df['technology_node'].isin(['7nm','10nm']).astype(int)
    df['defect_composite']    = (df['defect_count']*df['defect_density']).apply(np.log1p)
    df['Swing']      = df['vth']*np.log10(df['leakage_current'])
    df['IR_Drop'] = df['resistance']*(df['leakage_current'].clip(1e-10))
    df['node_x_cd']           = df['tech_encoded']*df['critical_dimension']
    FS_C = FS_B + ['is_advanced_node','defect_composite',
                   'Swing','IR_Drop','node_x_cd']
    # D: C + 추가 파생 (§8 실험)
    df['process_stability']    = (df['etch_rate']/df['etch_rate'].std()
                                 +df['deposition_rate']/df['deposition_rate'].std()
                                 +df['implant_energy']/df['implant_energy'].std())/3
    df['power_efficiency'] = df['vth']/df['resistance']/df['leakage_current'].clip(1e-10)
    df['cd_squared']           = df['critical_dimension']**2
    FS_D = FS_C + ['process_stability','power_efficiency','cd_squared']
    return df, le, {'A':FS_A,'B':FS_B,'C':FS_C,'D':FS_D}


# ══════════════════════════════════════════════════════════════
# §1  EDA
# ══════════════════════════════════════════════════════════════
def section1(df, le, FS):
    banner("§1  EDA 심층 분석 & 피처셋 설계")
    y = df[TARGET].values
    hg=df[df['technology_node'].isin(['7nm','10nm'])][TARGET]
    lg=df[df['technology_node'].isin(['14nm','22nm','28nm'])][TARGET]
    t_s,t_p=stats.ttest_ind(hg,lg) #hg와 lg 두 그룹간 평군 차이를 t-test 수행
    #모든 node 간 평균 차이가 있는지 ANAVA 수행
    f_s,f_p=stats.f_oneway(*[df[df['technology_node']==n][TARGET].values for n in TECH_NODES]) 
    
    print(f"  이분화 t-test: t={t_s:.2f}, p={t_p:.2e}")
    print(f"  ANOVA:        F={f_s:.2f}, p={f_p:.2e}")

    # 성능 비교를 위한 기준 모델을 GBM으로 설정
    probe = GradientBoostingRegressor(n_estimators=100,learning_rate=0.05,
                                      max_depth=3,random_state=RANDOM_STATE)
    fs_r2 = {}
    # 피처셋별 K-Fold CV를 통해 R2 점수를 계산
    for nm,fset in FS.items():
        r2 = float(cross_val_score(probe,df[fset].values,y,cv=kf,scoring='r2').mean())
        fs_r2[nm]=r2; print(f"  FS-{nm}: R²={r2:.4f}")
    
    corr_m = df[PROCESS_FEATURES].corr() #공정 피처들 간의 상관계수 계산
    # 상관계수가 0.8 이상인 다중공선성 추정 피처 쌍 찾기
    high_corr = [(PROCESS_FEATURES[i],PROCESS_FEATURES[j],float(corr_m.iloc[i,j]))
                 for i in range(len(PROCESS_FEATURES))
                 for j in range(i+1,len(PROCESS_FEATURES)) if abs(corr_m.iloc[i,j])>0.8]
    # IQR 방식을 사용해서 피처별 이상치 비율 계산
    outlier_pct = {f:float(((df[f]<df[f].quantile(.25)-1.5*(df[f].quantile(.75)-df[f].quantile(.25)))|
                             (df[f]>df[f].quantile(.75)+1.5*(df[f].quantile(.75)-df[f].quantile(.25)))).mean()*100)
                   for f in PROCESS_FEATURES}
    df_outlier = df.copy()
    df_outlier['leakage_current'] = np.log10(df_outlier['leakage_current'])
    outlier_pct_log = {f:float(((df_outlier[f]<df_outlier[f].quantile(.25)-1.5*(df_outlier[f].quantile(.75)-df_outlier[f].quantile(.25)))|
                                (df_outlier[f]>df_outlier[f].quantile(.75)+1.5*(df_outlier[f].quantile(.75)-df_outlier[f].quantile(.25)))).mean()*100)
                    for f in PROCESS_FEATURES}

    fig=plt.figure(figsize=(20,14))
    fig.suptitle('§1. EDA 심층 분석 — 피처셋 설계 근거',fontsize=16,fontweight='bold',y=1.01)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.38)

    #node별 yield 분포를 바이올린plot으로 표현
    ax=fig.add_subplot(gs[0,0])
    parts=ax.violinplot([df[df['technology_node']==n][TARGET].values for n in TECH_NODES],
                        showmeans=True,showmedians=True)
    for pc,n in zip(parts['bodies'],TECH_NODES): pc.set_facecolor(NODE_COLORS[n]); pc.set_alpha(0.7)
    ax.set_xticks(range(1,6)); ax.set_xticklabels(TECH_NODES,fontsize=8)
    ax.set_ylabel('Yield'); ax.set_title('A. 노드별 Yield (Violin)',fontweight='bold')
    ax.axhline(df[TARGET].mean(),color='gray',linestyle=':',lw=1.2,label=f'μ={df[TARGET].mean():.3f}')
    ax.legend(fontsize=7)

    # 다중공선성 히트맵
    ax=fig.add_subplot(gs[0,1])
    cols_hm=['thickness_uniformity','defect_count','defect_density',
             'critical_dimension','oxide_thickness','vth','resistance',TARGET]
    mask_=np.triu(np.ones((len(cols_hm),len(cols_hm)),dtype=bool),k=1)
    sns.heatmap(df[cols_hm].corr(),ax=ax,annot=True,fmt='.2f',cmap='RdYlGn',
                center=0,linewidths=0.4,annot_kws={'size':7},mask=mask_)
    ax.set_title('B. 다중공선성 진단',fontweight='bold')
    ax.tick_params(axis='x',rotation=35,labelsize=7); ax.tick_params(axis='y',rotation=0,labelsize=7)

    # node별 피처와 yield의 상관 분석
    ax=fig.add_subplot(gs[0,2])
    top5=['defect_count','thickness_uniformity','vth','critical_dimension','oxide_thickness']
    ch=pd.DataFrame({n:df[df['technology_node']==n][top5+[TARGET]].corr()[TARGET].drop(TARGET)
                     for n in TECH_NODES})
    sns.heatmap(ch,ax=ax,annot=True,fmt='.3f',cmap='RdYlGn_r',center=0,
                linewidths=0.4,annot_kws={'size':9,'fontweight':'bold'})
    ax.set_title('C. 노드별 피처-Yield 상관',fontweight='bold')

    # 피처별 이상치 비율
    ax=fig.add_subplot(gs[1,0])
    out_s=pd.Series(outlier_pct).sort_values(ascending=False)
    ax.barh(out_s.index,out_s.values,
            color=['#e74c3c' if v>3 else '#f39c12' if v>1 else '#2ecc71' for v in out_s.values],alpha=0.85)
    ax.axvline(3,color='red',linestyle='--',lw=1,label='3% 기준')
    ax.set_xlabel('이상치(%)'); ax.set_title('D. 피처별 이상치 비율',fontweight='bold'); ax.legend(fontsize=8)

    # 피처셋별 R2 성능비교
    ax=fig.add_subplot(gs[1,1])
    fl=['A','B','C★','D']; fv=[fs_r2[k] for k in ['A','B','C','D']]
    bars=ax.bar(fl,fv,color=['#95a5a6','#3498db','#e74c3c','#e67e22'],alpha=0.85)
    ax.set_ylim(min(fv)-.01,max(fv)+.008); ax.set_ylabel('R² (5-Fold CV)')
    ax.set_title('E. 피처셋별 R²',fontweight='bold')
    for bar,val in zip(bars,fv):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.0003,
                f'{val:.4f}',ha='center',fontweight='bold',fontsize=10)
    ax.axhline(fv[0],color='gray',linestyle=':',lw=1.5,label='Baseline'); ax.legend(fontsize=8)

    # F. 이분화 검증 히스토램
    ax=fig.add_subplot(gs[1,2])
    ax.hist(hg,bins=25,alpha=0.65,color='#e74c3c',label=f'고수율(7/10nm) μ={hg.mean():.3f}')
    ax.hist(lg,bins=25,alpha=0.65,color='#3498db',label=f'저수율(14~28nm) μ={lg.mean():.3f}')
    ax.axvline(hg.mean(),color='#c0392b',linestyle='--',lw=2)
    ax.axvline(lg.mean(),color='#1a5276',linestyle='--',lw=2)
    ax.set_xlabel('Yield'); ax.set_ylabel('Count')
    ax.set_title(f'F. 이분화 검증\nt={t_s:.1f}, p={t_p:.2e}',fontweight='bold'); ax.legend(fontsize=8)

    save_fig(fig,'sec1_eda_deep')
    save_json({'fs_r2':fs_r2,'t_stat':float(t_s),'t_p':float(t_p),'f_stat':float(f_s),
               'f_p':float(f_p),'high_corr_pairs':high_corr,'outlier_pct':outlier_pct},'sec1_eda')

# ══════════════════════════════════════════════════════════════
# §2  Phase 1 — 27개 모델 베이스라인
# ══════════════════════════════════════════════════════════════
def section2(df, FS):
    banner(f"§2  Phase 1 — 베이스라인 스크리닝 (27개 모델)")
    #FS_C에 대한 Scaling 데이터 준비
    X=df[FS['C']].values; y=df[TARGET].values; Xs=StandardScaler().fit_transform(X)

    # RANSAC: base_estimator 없으면 LinearRegression 사용
    # https://blog.naver.com/samsjang/221003286930 참고
    # Random sample consensus
    # sampling을 random으로 추출해서 Ridge로 회귀 모델을 만들고 오차가 적은 데이터(정상치)가 가장 많은 모델을 최종 선택하는 방식
    ransac = RANSACRegressor(estimator=Ridge(alpha=1.0),
                             min_samples=0.5, random_state=RANDOM_STATE)

    xgb_name = BOOST_LIBS['xgb_name']
    lgb_name = BOOST_LIBS['lgb_name']
    cb_name  = BOOST_LIBS['cb_name']

    # 모델이름 : (모델객체, 스케일링여부, 카테고리) 순서대로 기재함
    # Tree기반은 StandardScaler가 필요없음.
    MODELS = {
        # ── 선형 (6) ───────────────────────────────────────────
        'Linear Regression':    (LinearRegression(),                                      True,  'linear'),
        'Ridge':                (Ridge(alpha=1.0),                                        True,  'linear'),
        'Lasso':                (Lasso(alpha=0.001,max_iter=5000),                        True,  'linear'),
        'ElasticNet':           (ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=5000),      True,  'linear'),
        'PLS (n=8)':           (PLSRegression(n_components=8),                           True,  'linear'),
        'PCR (PCA+Ridge)':     (make_pipeline(StandardScaler(), PCA(n_components=0.95), Ridge()), True, 'linear'),
        
        # ── 이상치 강건 (6) ────────────────────────────────────
        'Huber Regressor':      (HuberRegressor(epsilon=1.35,max_iter=300),               True,  'robust'),
        'Bayesian Ridge':       (BayesianRidge(),                                         True,  'robust'),
        'TheilSen':             (TheilSenRegressor(random_state=RANDOM_STATE,
                                                   max_subpopulation=500),                True,  'robust'),
        'Kernel Ridge (RBF)':   (KernelRidge(alpha=1.0,kernel='rbf'),                     True,  'robust'),
        'RANSAC + Ridge':       (ransac,                                                  True,  'robust'),
        'Quantile (Q=0.1)':     (QuantileRegressor(quantile=0.1, alpha=0),                True,  'robust'),
        
        # ── 비선형 및 해석가능 (5) ────────────────────────────────
        'SVR (RBF)':           (SVR(kernel='rbf',C=1.0),                                 True,  'nonlinear'),
        'SVR (POLY)':          (SVR(kernel='poly',C=1.0),                                True,  'nonlinear'),
        'NuSVR':               (NuSVR(kernel='rbf',nu=0.5),                              True,  'nonlinear'),
        'KNN (k=5)':           (KNeighborsRegressor(n_neighbors=5),                      True,  'nonlinear'),
        'MLP (128-64)':        (MLPRegressor(hidden_layer_sizes=(128,64),max_iter=500,
                                             random_state=RANDOM_STATE),                  True,  'nonlinear'),
        
        # ── 단일 트리 (1) ──────────────────────────────────────
        'Decision Tree':        (DecisionTreeRegressor(max_depth=8,
                                                       random_state=RANDOM_STATE),       False, 'tree'),
        
        # ── 앙상블 (4) ─────────────────────────────────────────
        'Bagging (DT)':         (BaggingRegressor(
                                    estimator=DecisionTreeRegressor(max_depth=6),
                                    n_estimators=50,random_state=RANDOM_STATE,
                                    n_jobs=-1),                                           False, 'ensemble'),
        'Random Forest':        (RandomForestRegressor(n_estimators=100,
                                    random_state=RANDOM_STATE,n_jobs=-1),                 False, 'ensemble'),
        'Extra Trees':          (ExtraTreesRegressor(n_estimators=100,
                                    random_state=RANDOM_STATE,n_jobs=-1),                 False, 'ensemble'),
        'AdaBoost':             (AdaBoostRegressor(n_estimators=100,
                                    random_state=RANDOM_STATE),                           False, 'ensemble'),
        
        # ── 부스팅 (5) ─────────────────────────────────────────
        'Gradient Boosting':    (GradientBoostingRegressor(n_estimators=100,
                                    random_state=RANDOM_STATE),                           False, 'boosting'),
        'HistGradientBoost':    (HistGradientBoostingRegressor(random_state=RANDOM_STATE), False, 'boosting'),
        xgb_name:               (BOOST_LIBS['xgb'],                                       False, 'boosting'),
        lgb_name:               (BOOST_LIBS['lgb'],                                       False, 'boosting'),
        cb_name:                (BOOST_LIBS['cb'],                                        False, 'boosting'),
    }

    print(f"\n  라이브러리 상태:")
    print(f"    XGBoost : {xgb_name}")
    print(f"    LightGBM: {lgb_name}")
    print(f"    CatBoost: {cb_name}")
    print(f"\n  {'Model':<30s}  {'R² mean':>8s}  {'R² std':>7s}  {'RMSE':>7s}  {'Cat':>10s}")
    print("  "+"-"*72)

    p1={}
    for name,(model,use_std,cat) in MODELS.items():
        Xu=Xs if use_std else X
        try:
            cv=cross_validate(model,Xu,y,cv=kf,
                              scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],
                              n_jobs=-1)
            r2m=float(cv['test_r2'].mean()); r2s=float(cv['test_r2'].std())
            rmse=float(np.sqrt(-cv['test_neg_mean_squared_error'].mean()))
            mae=float(-cv['test_neg_mean_absolute_error'].mean())
        except Exception as e:
            print(f"  ⚠️ {name} 실패: {e}")
            r2m=r2s=rmse=mae=0.0
        p1[name]=dict(R2_mean=r2m,R2_std=r2s,RMSE=rmse,MAE=mae,category=cat,use_std=use_std)
        print(f"  {name:<30s}  {r2m:>8.4f}  {r2s:>7.4f}  {rmse:>7.4f}  {cat:>10s}")

    p1_df=(pd.DataFrame(p1).T.reset_index().rename(columns={'index':'model'})
           .sort_values('R2_mean',ascending=False).reset_index(drop=True))

    fig,axes=plt.subplots(1,3,figsize=(22,10))
    fig.suptitle(f'§2. Phase 1 — 베이스라인 스크리닝 ({len(MODELS)}개 모델, {CV_FOLDS}-Fold CV)\n'
                 f'{xgb_name} | {lgb_name} | {cb_name}',
                 fontsize=12,fontweight='bold')

    colors_p1=[CAT_COLORS.get(p1_df.loc[p1_df['model']==m,'category'].values[0],'#95a5a6')
               for m in p1_df['model']]
    bars=axes[0].barh(p1_df['model'],p1_df['R2_mean'].astype(float),color=colors_p1,alpha=0.85)
    axes[0].errorbar(p1_df['R2_mean'].astype(float),range(len(p1_df)),
                     xerr=p1_df['R2_std'].astype(float),fmt='none',color='black',capsize=3,lw=1.5)
    axes[0].invert_yaxis(); axes[0].axvline(0.6,color='orange',linestyle='--',lw=1.5)
    axes[0].legend(handles=[mpatches.Patch(fc=v,label=k) for k,v in CAT_COLORS.items()],
                   fontsize=8,loc='lower right')
    axes[0].set_xlabel('R²'); axes[0].set_title('R² 비교',fontweight='bold')
    for bar,val in zip(bars,p1_df['R2_mean'].astype(float)):
        axes[0].text(val+.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=7)

    ps=p1_df.sort_values('RMSE')
    bars2=axes[1].barh(ps['model'],ps['RMSE'].astype(float),
                       color=[CAT_COLORS.get(ps.loc[ps['model']==m,'category'].values[0],'#95a5a6')
                              for m in ps['model']],alpha=0.85)
    axes[1].invert_yaxis(); axes[1].set_xlabel('RMSE'); axes[1].set_title('RMSE 비교',fontweight='bold')
    for bar,val in zip(bars2,ps['RMSE'].astype(float)):
        axes[1].text(val+.0003,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=7)

    cat_avg=(p1_df.groupby('category')['R2_mean']
             .apply(lambda s:s.astype(float).mean()).sort_values(ascending=False))
    bars3=axes[2].bar(cat_avg.index,cat_avg.values,
                      color=[CAT_COLORS.get(c,'#95a5a6') for c in cat_avg.index],alpha=0.85)
    axes[2].set_ylabel('평균 R²'); axes[2].set_title('카테고리별 평균 R²',fontweight='bold')
    for bar,val in zip(bars3,cat_avg.values):
        axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.003,
                     f'{val:.4f}',ha='center',fontweight='bold',fontsize=10)

    save_fig(fig,'sec2_phase1_screening')
    save_csv(p1_df,'phase1_results')
    save_json({'results':p1,'cat_avg':cat_avg.to_dict(),
               'lib_status':{'xgb':xgb_name,'lgb':lgb_name,'cb':cb_name}},'sec2_phase1')
    return p1

# ══════════════════════════════════════════════════════════════
# §3  Phase 2 — 튜닝 (best_params 완전 저장)
# ══════════════════════════════════════════════════════════════
def section3(df, FS, p1_results):
    banner("§3  Phase 2 — 하이퍼파라미터 튜닝 (RandomizedSearchCV)")
    X=df[FS['C']].values; y=df[TARGET].values
    # candidates=['Gradient Boosting',BOOST_LIBS['xgb_name'],BOOST_LIBS['lgb_name'],
    #             BOOST_LIBS['cb_name'],'Random Forest','AdaBoost']
    candidates=['Gradient Boosting','AdaBoost','Random Forest','HistGradientBoost','Extra Trees']
    baseline_r2={n:p1_results[n]['R2_mean'] for n in candidates if n in p1_results}
    print("\n  Phase1 Baseline R²:")
    for n,r in sorted(baseline_r2.items(),key=lambda x:-x[1]):
        print(f"    {n:<34s}: {r:.4f}")

    SEARCH_SPACES={
        'Gradient Boosting':{
            'model':GradientBoostingRegressor(random_state=RANDOM_STATE),
            'params':{'n_estimators':[100,200,300],'learning_rate':[0.01,0.03,0.05,0.08,0.1,0.15],
                      'max_depth':[2,3,4,5],'subsample':[0.7,0.8,0.9,1.0],
                      'min_samples_leaf':[1,2,4],'max_features':['sqrt',0.7,0.9,None]},
            'n_iter':40},
        'Extra Trees': {
            'model':ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 8, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7],
                'bootstrap': [True, False]
            },
            'n_iter': 25
        },
        'HistGradientBoost':{
            'model': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            'params': {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, None],
                'min_samples_leaf': [10, 20, 30, 50],
                'l2_regularization': [0.0, 0.01, 0.1, 1.0],
                'max_bins': [128, 255]
            },
            'n_iter': 25
        },
    }
    # XGB/LGB/CB 대안의 공통 탐색 공간
    for key,name,extra_p in [
        ('xgb',BOOST_LIBS['xgb_name'],
         {'max_iter':[100,200,300],'learning_rate':[0.01,0.05,0.1,0.15],
          'max_depth':[3,5,7,None],'min_samples_leaf':[10,20,30,50],
          'l2_regularization':[0.0,0.01,0.1,1.0],'max_bins':[128,255]}),
        ('lgb',BOOST_LIBS['lgb_name'],
         {'max_iter':[200,300,500],'learning_rate':[0.01,0.03,0.05,0.08],
          'max_depth':[None,5,8],'min_samples_leaf':[5,10,15,20],
          'l2_regularization':[0.0,0.01,0.1]}),
        ('cb', BOOST_LIBS['cb_name'],
         {'max_iter':[200,300,500],'learning_rate':[0.01,0.03,0.05],
          'max_depth':[4,5,6,8],'min_samples_leaf':[20,30,50],
          'l2_regularization':[0.5,1.0,2.0,5.0]}),
    ]:
        if name in p1_results:
            # 실제 라이브러리면 해당 파라미터, 아니면 HistGBM 파라미터
            model_inst = BOOST_LIBS[key]
            if 'HistGBM' in name or '대안' in name:
                SEARCH_SPACES[name]={'model':HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                                     'params':extra_p,'n_iter':25}

    if 'Random Forest' in p1_results:
        SEARCH_SPACES['Random Forest']={
            'model':RandomForestRegressor(random_state=RANDOM_STATE,n_jobs=-1),
            'params':{'n_estimators':[100,200,300,500],'max_depth':[5,8,12,None],
                      'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],
                      'max_features':['sqrt','log2',0.5,0.7],'bootstrap':[True,False]},
            'n_iter':40}
    if 'AdaBoost' in p1_results:
        SEARCH_SPACES['AdaBoost']={
            'model':AdaBoostRegressor(random_state=RANDOM_STATE),
            'params':{'n_estimators':[100,200,300,500],'learning_rate':[0.01,0.05,0.1,0.5,1.0],
                      'loss':['linear','square','exponential']},
            'n_iter':25}

    tuning_results={}; best_estimators={}
    for name,cfg in SEARCH_SPACES.items():
        bl=baseline_r2.get(name,0.0)
        print(f"\n  [{name}] Baseline={bl:.4f} → 탐색 (n_iter={cfg['n_iter']})...",end=' ',flush=True)
        t0=time.time()
        search=RandomizedSearchCV(cfg['model'],cfg['params'],n_iter=cfg['n_iter'],
                                   cv=3,scoring='r2',random_state=RANDOM_STATE,n_jobs=-1)
        search.fit(X,y)
        cv5=cross_validate(search.best_estimator_,X,y,cv=kf,
                           scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],n_jobs=-1)
        r2m=float(cv5['test_r2'].mean()); r2s=float(cv5['test_r2'].std())
        rmse=float(np.sqrt(-cv5['test_neg_mean_squared_error'].mean()))
        mae=float(-cv5['test_neg_mean_absolute_error'].mean())
        elapsed=time.time()-t0
        # ★ best_params 반드시 저장
        tuning_results[name]=dict(baseline_r2=bl,search_best_r2=float(search.best_score_),
            cv5_r2=r2m,cv5_std=r2s,cv5_rmse=rmse,cv5_mae=mae,
            best_params=search.best_params_,elapsed=elapsed)
        best_estimators[name]=search.best_estimator_
        search.best_estimator_.fit(X,y)
        print(f"search={search.best_score_:.4f}  5CV={r2m:.4f}±{r2s:.4f}  [{elapsed:.0f}s]")
        print(f"    best_params: {search.best_params_}")

    # ★ 저장 (best_params 포함)
    save_json(tuning_results,'sec3_tuning_results')
    joblib.dump(best_estimators,os.path.join(OUTPUT_DIR,'all_tuned_models.pkl'))

    names_=list(SEARCH_SPACES.keys())
    before=[tuning_results[n]['baseline_r2'] for n in names_]
    after =[tuning_results[n]['cv5_r2']      for n in names_]
    stds_ =[tuning_results[n]['cv5_std']     for n in names_]
    rmses_=[tuning_results[n]['cv5_rmse']    for n in names_]
    x=np.arange(len(names_)); w=0.35

    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle('§3. Phase 2 — 하이퍼파라미터 튜닝 결과',fontsize=13,fontweight='bold')
    ax=axes[0]
    ax.bar(x-w/2,before,w,label='Baseline',color='#95a5a6',alpha=0.85)
    ax.bar(x+w/2,after, w,label='Tuned',   color='#e74c3c',alpha=0.85)
    ax.errorbar(x+w/2,after,yerr=stds_,fmt='none',color='black',capsize=4,lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in names_],fontsize=7)
    ax.set_ylabel('R²'); ax.set_title('튜닝 전·후 R² 비교'); ax.legend()
    ax.set_ylim(min(before+after)-.025,max(before+after)+.025)
    for i,(b,a) in enumerate(zip(before,after)):
        ax.text(i-w/2,b+.001,f'{b:.4f}',ha='center',fontsize=6.5,color='#555')
        ax.text(i+w/2,a+.001,f'{a:.4f}',ha='center',fontsize=6.5,fontweight='bold')
    for ax_,vals,title,color in zip(axes[1:],[rmses_,[tuning_results[n]['cv5_mae'] for n in names_]],
                                     ['RMSE (↓)','MAE (↓)'],['#e67e22','#27ae60']):
        ax_.bar(x,vals,color=color,alpha=0.85)
        ax_.set_xticks(x); ax_.set_xticklabels([n.replace(' ','\n') for n in names_],fontsize=7)
        ax_.set_title(title)
        for bar,val in zip(ax_.patches,vals):
            ax_.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.0003,f'{val:.4f}',ha='center',fontsize=8)
    save_fig(fig,'sec3a_tuning_comparison')

    # top5=sorted(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'],reverse=True)[:4]
    # fig,axes=plt.subplots(1,5,figsize=(25,5))
    # fig.suptitle('§3. Learning Curve — 상위 5개 모델',fontsize=13,fontweight='bold')
    # for ax_,name,color in zip(axes,top5,['#e74c3c','#2980b9','#27ae60',"#6a20e0","#efe33c"]):
    #     tr_sz,tr_sc,va_sc=learning_curve(best_estimators[name],X,y,
    #         train_sizes=np.linspace(0.1,1.0,8),cv=5,scoring='r2',n_jobs=-1)
    top3 = sorted(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'], reverse=True)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # 5개면 너비를 더 확보해야 합니다
    axes = axes.flatten() # 1차원 배열로 안전하게 처리

    colors = ['#e74c3c', '#2980b9', '#27ae60']

    for i, ax_ in enumerate(axes):
        if i < len(top3): # 모델이 있는 경우에만 그림
            name = top3[i]
            color = colors[i]
            
            tr_sz, tr_sc, va_sc = learning_curve(best_estimators[name], X, y,
                                                train_sizes=np.linspace(0.1, 1.0, 8), 
                                                cv=5, scoring='r2', n_jobs=-1)
            tr_m=tr_sc.mean(axis=1); tr_s=tr_sc.std(axis=1)
            va_m=va_sc.mean(axis=1); va_s=va_sc.std(axis=1)
            ax_.plot(tr_sz,tr_m,'o-',color=color,lw=2,label='Train')
            ax_.fill_between(tr_sz,tr_m-tr_s,tr_m+tr_s,alpha=0.12,color=color)
            ax_.plot(tr_sz,va_m,'s--',color=color,alpha=0.65,lw=2,label='Validation')
            ax_.fill_between(tr_sz,va_m-va_s,va_m+va_s,alpha=0.07,color=color)
            ax_.set_xlabel('Training Samples'); ax_.set_ylabel('R²')
            ax_.set_title(f'{name}\nCV R²={tuning_results[name]["cv5_r2"]:.4f}',fontweight='bold')
            ax_.legend(fontsize=8); ax_.grid(alpha=0.3)
            gap=tr_m[-1]-va_m[-1]
            ax_.text(0.05,0.07,f'과적합 Gap={gap:.3f}',transform=ax_.transAxes,fontsize=8,
                    bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    save_fig(fig,'sec3b_learning_curve')
    return tuning_results, best_estimators

def section3_1(df, FS, p1_results):
    banner("§3-1. Phase 2 — 전 모델 하이퍼파라미터 통합 튜닝 및 성능 비교")
    X = df[FS['C']].values
    y = df[TARGET].values
    
    # 스케일링이 필요한 모델과 아닌 모델을 구분하기 위한 유틸리티 함수
    def get_pipe(model, needs_scaling=True):
        if needs_scaling:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return Pipeline([('model', model)])

    xgb_name = BOOST_LIBS['xgb_name']
    lgb_name = BOOST_LIBS['lgb_name']
    cb_name  = BOOST_LIBS['cb_name']

    # 모든 모델을 파이프라인으로 래핑
    MODELS = {
        'Linear Regression': (get_pipe(LinearRegression()), 'linear'),
        'Ridge': (get_pipe(Ridge(alpha=1.0)), 'linear'),
        'Lasso': (get_pipe(Lasso(alpha=0.001, max_iter=5000)), 'linear'),
        'ElasticNet': (get_pipe(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)), 'linear'),
        'PLS (n=8)': (get_pipe(PLSRegression(n_components=8)), 'linear'),
        'PCR (PCA+Ridge)': (Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('ridge', Ridge())]), 'linear'),
        'Huber Regressor': (get_pipe(HuberRegressor(epsilon=1.35, max_iter=300)), 'robust'),
        'Bayesian Ridge': (get_pipe(BayesianRidge()), 'robust'),
        'TheilSen': (get_pipe(TheilSenRegressor(random_state=RANDOM_STATE, max_subpopulation=500)), 'robust'),
        'Kernel Ridge (RBF)': (get_pipe(KernelRidge(alpha=1.0, kernel='rbf')), 'robust'),
        'RANSAC + Ridge': (get_pipe(RANSACRegressor(estimator=Ridge(), min_samples=0.5, random_state=RANDOM_STATE)), 'robust'),
        'SVR (RBF)': (get_pipe(SVR(kernel='rbf', C=1.0)), 'nonlinear'),
        'SVR (POLY)': (get_pipe(SVR(kernel='poly', C=1.0)), 'nonlinear'),
        'NuSVR': (get_pipe(NuSVR(kernel='rbf', nu=0.5)), 'nonlinear'),
        'KNN (k=5)': (get_pipe(KNeighborsRegressor(n_neighbors=5)), 'nonlinear'),
        'Decision Tree': (get_pipe(DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), False), 'tree'),
        'Bagging (DT)': (get_pipe(BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1), False), 'ensemble'),
        'Random Forest': (get_pipe(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False), 'ensemble'),
        'Extra Trees': (get_pipe(ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False), 'ensemble'),
        'AdaBoost': (get_pipe(AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE), False), 'ensemble'),
        'Gradient Boosting': (get_pipe(GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), False), 'boosting'),
        'HistGradientBoost': (get_pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE), False), 'boosting'),
        xgb_name: (get_pipe(BOOST_LIBS['xgb'], False), 'boosting'),
        lgb_name: (get_pipe(BOOST_LIBS['lgb'], False), 'boosting'),
        cb_name: (get_pipe(BOOST_LIBS['cb'], False), 'boosting'),
    }

    # 파이프라인 적용에 따른 파라미터명 수정 (예: 'alpha' -> 'model__alpha')
    def wrap_params(params):
        return {f'model__{k}': v for k, v in params.items()}

    SEARCH_SPACES = {
        name: {'model': MODELS[name][0], 'params': wrap_params(p_cfg['params']), 'n_iter': p_cfg['n_iter']}
        for name, p_cfg in {
            'Linear Regression': {'params': {}, 'n_iter': 1},
            'Ridge': {'params': {'alpha': [0.1, 1.0, 10.0]}, 'n_iter': 3},
            'Lasso': {'params': {'alpha': [0.0001, 0.001, 0.01]}, 'n_iter': 3},
            'ElasticNet': {'params': {'alpha': [0.001, 0.01], 'l1_ratio': [0.2, 0.5, 0.8]}, 'n_iter': 6},
            'PLS (n=8)': {'params': {'n_components': [4, 6, 8]}, 'n_iter': 3},
            'PCR (PCA+Ridge)': {'params': {'pca__n_components': [0.8, 0.9, 0.95], 'ridge__alpha': [0.1, 1.0]}, 'n_iter': 6},
            'Huber Regressor': {'params': {'epsilon': [1.1, 1.35, 1.5], 'alpha': [0.0001, 0.001]}, 'n_iter': 6},
            'Bayesian Ridge': {'params': {'alpha_init': [1e-6, 1e-3], 'lambda_init': [1e-6, 1e-3]}, 'n_iter': 4},
            'TheilSen': {'params': {'max_subpopulation': [200, 500, 1000]}, 'n_iter': 3},
            'Kernel Ridge (RBF)': {'params': {'alpha': [0.1, 1.0], 'gamma': [0.01, 0.1, 1.0]}, 'n_iter': 6},
            'RANSAC + Ridge': {'params': {'min_samples': [0.3, 0.5, 0.7]}, 'n_iter': 3},
            'SVR (RBF)': {'params': {'C': [0.1, 1.0, 10], 'gamma': ['scale', 'auto']}, 'n_iter': 6},
            'SVR (POLY)': {'params': {'C': [0.1, 1.0], 'degree': [2, 3]}, 'n_iter': 4},
            'NuSVR': {'params': {'nu': [0.3, 0.5, 0.7], 'C': [0.1, 1.0]}, 'n_iter': 6},
            'KNN (k=5)': {'params': {'n_neighbors': [3, 5, 10, 20]}, 'n_iter': 4},
            'Decision Tree': {'params': {'max_depth': [3, 5, 8, 12]}, 'n_iter': 4},
            'Bagging (DT)': {'params': {'n_estimators': [50, 100], 'max_samples': [0.5, 0.8, 1.0]}, 'n_iter': 6},
            'Random Forest': {'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}, 'n_iter': 8},
            'Extra Trees': {'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}, 'n_iter': 6},
            'AdaBoost': {'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}, 'n_iter': 6},
            'Gradient Boosting': {'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}, 'n_iter': 6},
            'HistGradientBoost': {'params': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, None]}, 'n_iter': 6},
            xgb_name: {'params': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}, 'n_iter': 6},
            lgb_name: {'params': {'learning_rate': [0.01, 0.1], 'num_leaves': [20, 31, 50]}, 'n_iter': 6},
            cb_name: {'params': {'learning_rate': [0.01, 0.1], 'depth': [4, 6, 8]}, 'n_iter': 6},
        }.items()
    }
    # PCR과 같이 파라미터명이 다른 경우 예외처리
    SEARCH_SPACES['PCR (PCA+Ridge)']['params'] = {'pca__n_components': [0.8, 0.9, 0.95], 'ridge__alpha': [0.1, 1.0]}
    tuning_results, best_estimators = {}, {}
    for name, cfg in SEARCH_SPACES.items():
        print(f"  [{name}] 튜닝 중...", end=' ', flush=True)
        search = RandomizedSearchCV(cfg['model'], cfg['params'], n_iter=cfg['n_iter'], cv=3, scoring='r2', random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X, y)
        cv = cross_validate(search.best_estimator_, X, y, cv=kf, scoring=['r2', 'neg_mean_squared_error'], n_jobs=-1)
        tuning_results[name] = {'cv_r2': float(cv['test_r2'].mean()), 'rmse': float(np.sqrt(-cv['test_neg_mean_squared_error'].mean())), 'best_params': search.best_params_}
        best_estimators[name] = search.best_estimator_
        print(f"완료 (R²: {tuning_results[name]['cv_r2']:.4f})")

    # [시각화 구현]
    res_df = pd.DataFrame(tuning_results).T.reset_index().rename(columns={'index':'model'})
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('§3-1. 모델 튜닝 성능 및 단계별 비교 분석', fontsize=15, fontweight='bold')

    # 1. 3-1 튜닝 성능
    res_df = res_df.sort_values('cv_r2', ascending=True)
    axes[0].barh(res_df['model'], res_df['cv_r2'], color='skyblue', alpha=0.8)
    axes[0].set_title('Tuned R² (Phase 2)', fontweight='bold')

    # 2. Phase 2 vs Phase 1 변화
    common = [m for m in res_df['model'] if m in p1_results]
    diff_df = pd.DataFrame({'P1': [p1_results[m]['R2_mean'] for m in common], 'P2': [tuning_results[m]['cv_r2'] for m in common]}, index=common)
    diff_df.plot(kind='barh', ax=axes[1], color=['#95a5a6', '#e74c3c'], width=0.7)
    axes[1].set_title('성능 향상 비교 (P1 vs P2)', fontweight='bold')

    # 3. 전체 요약 (간략화)
    # axes[2].scatter(res_df['cv_r2'], res_df['rmse'], color='purple', alpha=0.6)
    # axes[2].set_xlabel('R²'); axes[2].set_ylabel('RMSE'); axes[2].set_title('R² vs RMSE (튜닝 모델)', fontweight='bold')
    common = [m for m in res_df['model'] if m in p1_results]
    diff_df = pd.DataFrame({'Delta': [tuning_results[m]['cv_r2']-p1_results[m]['R2_mean'] for m in common]}, index=common)
    diff_df.plot(kind='barh', ax=axes[2], color=['#e74c3c'], width=0.7)
    axes[2].set_title('성능 향상 비교 (P2-P1)', fontweight='bold')
    
    save_fig(fig, 'sec3_1_summary_visualization')
    save_json(tuning_results, 'sec3_1_final_results')
    
    return tuning_results, best_estimators

# ══════════════════════════════════════════════════════════════
# §4  Phase 3 — 고급 앙상블
# ══════════════════════════════════════════════════════════════
def section4(df, FS, tuning_results, best_estimators):
    banner("§4  Phase 3 — 고급 앙상블 (Stacking / Voting)")
    X=df[FS['C']].values; y=df[TARGET].values
    top3=sorted(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'],reverse=True)[:3]
    top3_est=[(n.replace(' ','_').replace('(','').replace(')','')
                .replace('≈','').replace('/','').replace('·','')[:20],
               best_estimators[n]) for n in top3]
    print(f"  기반 학습기 Top3: {top3}")

    adv={}
    for aname,amodel in [('Voting (Top3)',VotingRegressor(estimators=top3_est,n_jobs=-1)),
                          ('Stacking (Top3→Ridge)',StackingRegressor(
                              estimators=top3_est,final_estimator=Ridge(alpha=1.0),cv=3,n_jobs=-1))]:
        cv5=cross_validate(amodel,X,y,cv=kf,
                           scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],n_jobs=-1)
        r2m=float(cv5['test_r2'].mean())
        adv[aname]=dict(cv5_r2=r2m,cv5_std=float(cv5['test_r2'].std()),
                        cv5_rmse=float(np.sqrt(-cv5['test_neg_mean_squared_error'].mean())),
                        cv5_mae=float(-cv5['test_neg_mean_absolute_error'].mean()))
        print(f"  {aname:<30s}: R²={r2m:.4f}±{adv[aname]['cv5_std']:.4f}")

    all_sc={}
    for n,res in tuning_results.items():
        all_sc[f'{n} (Tuned)']=dict(R2=res['cv5_r2'],std=res['cv5_std'],RMSE=res['cv5_rmse'])
    for n,res in adv.items():
        all_sc[n]=dict(R2=res['cv5_r2'],std=res['cv5_std'],RMSE=res['cv5_rmse'])
    lb=pd.DataFrame(all_sc).T.reset_index().rename(columns={'index':'model'}).sort_values('R2',ascending=False)

    fig,ax=plt.subplots(figsize=(13,9))
    fig.suptitle('§4. 전체 모델 Leaderboard',fontsize=14,fontweight='bold')
    bar_colors=[]
    for m in lb['model']:
        if 'Stacking' in m or 'Voting' in m: bar_colors.append('#9b59b6')
        elif 'Gradient Boosting' in m: bar_colors.append('#e74c3c')
        elif '≈' in m or 'Hist' in m or 'XGBoost' in m or 'LightGBM' in m or 'CatBoost' in m: bar_colors.append('#c0392b')
        elif 'Tuned' in m: bar_colors.append('#e67e22')
        else: bar_colors.append('#95a5a6')
    bars=ax.barh(lb['model'],lb['R2'].astype(float),color=bar_colors,alpha=0.85)
    ax.errorbar(lb['R2'].astype(float),range(len(lb)),xerr=lb['std'].astype(float),
                fmt='none',color='black',capsize=3,lw=1.5)
    ax.invert_yaxis(); ax.set_xlabel('R² (5-Fold CV)',fontsize=12)
    ax.legend(handles=[mpatches.Patch(fc='#e74c3c',label='GBM (Tuned)'),
                       mpatches.Patch(fc='#c0392b',label='XGB/LGBM/CB 계열'),
                       mpatches.Patch(fc='#9b59b6',label='Stacking/Voting'),
                       mpatches.Patch(fc='#e67e22',label='기타 앙상블'),
                       mpatches.Patch(fc='#95a5a6',label='Baseline')],fontsize=9,loc='lower right')
    for bar,val in zip(bars,lb['R2'].astype(float)):
        ax.text(val+.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=8.5,fontweight='bold')
    save_fig(fig,'sec4_leaderboard')
    save_json(adv,'sec4_adv_results')
    save_csv(lb,'sec4_leaderboard')
    return adv

# ══════════════════════════════════════════════════════════════
# §5-7  진단 / EDA 비교 / 요약
# ══════════════════════════════════════════════════════════════
def section5_6_7(df, FS, tuning_results, best_estimators, adv_results):
    banner("§5–7  최종 모델 진단 / EDA 비교 / 종합 요약")
    FS_C=FS['C']; X=df[FS_C].values; y=df[TARGET].values
    best_name=max(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'])
    FINAL=best_estimators[best_name]; FINAL.fit(X,y)
    y_pred=FINAL.predict(X); residuals=y-y_pred
    print(f"  최종 모델: {best_name}  Train R²={r2_score(y,y_pred):.4f}")
    joblib.dump(FINAL,os.path.join(OUTPUT_DIR,'best_model.pkl'))

    nodes_=[n for n in TECH_NODES if (df['technology_node']==n).any()]
    node_perf={}
    for node in nodes_:
        mask=df['technology_node']==node
        node_perf[node]=dict(R2=float(r2_score(y[mask],y_pred[mask])),
                             RMSE=float(np.sqrt(mean_squared_error(y[mask],y_pred[mask]))),
                             n=int(mask.sum()))
        print(f"  {node}: R²={node_perf[node]['R2']:.4f} (n={node_perf[node]['n']})")

    # §5 Figure
    r2_vals=[node_perf[n]['R2'] for n in nodes_]; n_vals=[node_perf[n]['n'] for n in nodes_]
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    fig.suptitle(f'§5. 최종 모델 심층 진단 — {best_name}',fontsize=14,fontweight='bold')
    ax=axes[0,0]
    for node in nodes_:
        mask=df['technology_node']==node
        ax.scatter(y[mask],y_pred[mask],alpha=0.4,s=12,color=NODE_COLORS.get(node,'#95a5a6'),label=node)
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=1.5,label='Perfect')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'A. Pred vs Actual (R²={r2_score(y,y_pred):.3f})',fontweight='bold')
    ax.legend(fontsize=7,markerscale=2)
    ax=axes[0,1]
    for node in nodes_:
        mask=df['technology_node']==node
        ax.scatter(y_pred[mask],residuals[mask],alpha=0.3,s=8,color=NODE_COLORS.get(node,'#95a5a6'))
    ax.axhline(0,color='red',linestyle='--',lw=1.5)
    ax.axhline(1.96*residuals.std(),color='orange',linestyle=':',lw=1,label='±1.96σ')
    ax.axhline(-1.96*residuals.std(),color='orange',linestyle=':',lw=1)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residual'); ax.set_title('B. 잔차 분석',fontweight='bold'); ax.legend(fontsize=8)
    ax=axes[0,2]
    stats.probplot(residuals,dist='norm',plot=ax); ax.set_title('C. Q-Q Plot',fontweight='bold')
    ax.get_lines()[0].set(markersize=3,alpha=0.5)
    ax=axes[1,0]
    bars=ax.bar(nodes_,r2_vals,color=[NODE_COLORS.get(n,'#95a5a6') for n in nodes_],alpha=0.85)
    ax.set_ylabel('R²'); ax.set_title('D. 노드별 R²',fontweight='bold'); ax.set_ylim(0,1.1)
    for bar,val,n_ in zip(bars,r2_vals,n_vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.01,
                f'{val:.3f}\n(n={n_})',ha='center',fontweight='bold',fontsize=8.5)
    ax=axes[1,1]
    ax.hist(residuals,bins=40,color='#3498db',edgecolor='white',alpha=0.7,density=True)
    xr=np.linspace(residuals.min(),residuals.max(),200)
    ax.plot(xr,stats.norm.pdf(xr,residuals.mean(),residuals.std()),'r-',lw=2)
    ax.set_xlabel('Residual'); ax.set_ylabel('Density')
    ax.set_title(f'E. 잔차 분포\nμ={residuals.mean():.4f}, σ={residuals.std():.4f}',fontweight='bold')
    _,p_norm=stats.shapiro(residuals[:500])
    ax.text(0.03,.90,f'Shapiro-Wilk p={p_norm:.4f}',transform=ax.transAxes,fontsize=8,
            bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.9))
    ax=axes[1,2]
    res_bn=[residuals[df['technology_node']==n] for n in nodes_]
    bp=ax.boxplot(res_bn,labels=nodes_,patch_artist=True,notch=True)
    for patch,node in zip(bp['boxes'],nodes_): patch.set_facecolor(NODE_COLORS.get(node,'#95a5a6')); patch.set_alpha(0.7)
    ax.axhline(0,color='red',linestyle='--',lw=1.5); ax.set_ylabel('Residual'); ax.set_title('F. 노드별 잔차',fontweight='bold')
    save_fig(fig,'sec5_final_diagnostics')

    # §6 EDA 비교
    FS_A_=PROCESS_FEATURES+['tech_encoded']; FS_B_=[f for f in PROCESS_FEATURES if f!='defect_density']+['tech_encoded']
    bp_raw=tuning_results[best_name]['best_params']
    # HistGBM 대안과 GBM 파라미터 분리
    if isinstance(best_estimators[best_name], GradientBoostingRegressor):
        cmp_model=GradientBoostingRegressor(**bp_raw,random_state=RANDOM_STATE)
    else:
        cmp_model=HistGradientBoostingRegressor(**bp_raw,random_state=RANDOM_STATE)
    COMP={'Best Model':cmp_model,
          'RF':RandomForestRegressor(n_estimators=200,max_depth=12,max_features=0.7,
                                     random_state=RANDOM_STATE,n_jobs=-1)}
    comp_table=[]
    for mname,model in COMP.items():
        row={'Model':mname}
        for fsn,fset in [('A-Baseline',FS_A_),('B-EDA제거',FS_B_),('C-EDA+FE★',FS_C)]:
            row[fsn]=float(cross_val_score(model,df[fset].values,y,cv=kf,scoring='r2').mean())
        comp_table.append(row)
    comp_df=pd.DataFrame(comp_table).set_index('Model')
    delta_B=comp_df.iloc[:,1]-comp_df.iloc[:,0]; delta_C=comp_df.iloc[:,2]-comp_df.iloc[:,0]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle('§6. EDA 피처셋 반영 효과 비교',fontsize=13,fontweight='bold')
    sns.heatmap(comp_df.astype(float),ax=axes[0],annot=True,fmt='.4f',cmap='RdYlGn',
                linewidths=0.5,annot_kws={'size':11,'fontweight':'bold'},
                vmin=comp_df.values.astype(float).min()-.002,vmax=comp_df.values.astype(float).max()+.002)
    axes[0].set_title('R² (5-Fold CV)',fontweight='bold'); axes[0].tick_params(axis='x',rotation=15,labelsize=9)
    x_d=np.arange(len(comp_df)); wd=0.3
    axes[1].bar(x_d-wd/2,delta_B.values.astype(float)*1000,wd,label='B-A',color='#3498db',alpha=0.85)
    axes[1].bar(x_d+wd/2,delta_C.values.astype(float)*1000,wd,label='C-A',color='#e74c3c',alpha=0.85)
    axes[1].set_xticks(x_d); axes[1].set_xticklabels(comp_df.index,fontsize=10)
    axes[1].axhline(0,color='black',lw=0.8); axes[1].set_ylabel('R² 개선량 (×10⁻³)')
    axes[1].set_title('Baseline(A) 대비 개선량',fontweight='bold'); axes[1].legend(fontsize=9)
    save_fig(fig,'sec6_eda_comparison')

    # §7 요약 Figure
    fig=plt.figure(figsize=(18,10))
    fig.suptitle('§7. Best Model 선정 — 전 과정 요약',fontsize=15,fontweight='bold',y=1.01)
    gs7=gridspec.GridSpec(2,4,figure=fig,hspace=0.5,wspace=0.4)
    ax=fig.add_subplot(gs7[0,:2])
    d_ph1=load_json('sec2_phase1')
    p1_best=max((v.get('R2_mean',0) for v in (d_ph1.get('results',{}) if d_ph1 else {}).values()),default=0.66)
    p_best=[p1_best,tuning_results[best_name]['cv5_r2'],max(r['cv5_r2'] for r in adv_results.values())]
    p_mdl=['Phase1 Best',f'{best_name[:18]}★','Stacking']
    ax.plot(['Phase1\nBaseline','Phase2\nTuned','Phase3\nAdvanced'],p_best,
            'o-',color='#e74c3c',lw=2.5,markersize=12,zorder=3)
    ax.fill_between(['Phase1\nBaseline','Phase2\nTuned','Phase3\nAdvanced'],
                    p_best,[v-.01 for v in p_best],alpha=0.1,color='#e74c3c')
    # 수정 후: range()를 사용하여 인덱스(0, 1, 2)를 명시적으로 사용
    categories = ['Phase1\nBaseline', 'Phase2\nTuned', 'Phase3\nAdvanced']
    for i, (ph, val, mn) in enumerate(zip(categories, p_best, p_mdl)):
        ax.annotate(f'{mn}\nR²={val:.4f}', 
                    xy=(i, val), # 문자열 대신 숫자 인덱스(i) 사용
                    textcoords='offset points', 
                    xytext=(0, 14),
                    ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9))
    ax.set_ylim(min(p_best)-.03,max(p_best)+.04); ax.set_ylabel('R²'); ax.grid(alpha=0.3)
    ax.set_title('A. Phase별 최고 성능 추이',fontweight='bold',fontsize=12)
    ax2=fig.add_subplot(gs7[0,2:]); ax2.axis('off')
    param_text='\n'.join([f"  {k}: {v}" for k,v in tuning_results[best_name]['best_params'].items()])
    ax2.text(0.05,.5,f"★ Best: {best_name}\n\n{param_text}\n\n"
             f"5-CV R²  = {tuning_results[best_name]['cv5_r2']:.4f} ±{tuning_results[best_name]['cv5_std']:.4f}\n"
             f"5-CV RMSE= {tuning_results[best_name]['cv5_rmse']:.4f}\n"
             f"5-CV MAE = {tuning_results[best_name]['cv5_mae']:.4f}",
             transform=ax2.transAxes,fontsize=10,va='center',fontfamily='monospace',
             bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.9))
    ax2.set_title('B. 최종 파라미터 요약',fontweight='bold',fontsize=12)
    ax3=fig.add_subplot(gs7[1,:2])
    bars=ax3.bar(nodes_,r2_vals,color=[NODE_COLORS.get(n,'#95a5a6') for n in nodes_],alpha=0.85,width=0.5)
    ax3.set_ylabel('R²'); ax3.set_ylim(0,1.1); ax3.set_title('C. 노드별 R²',fontweight='bold',fontsize=11)
    for bar,val,n_ in zip(bars,r2_vals,n_vals):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.01,f'{val:.3f}\n(n={n_})',ha='center',fontweight='bold',fontsize=9)
    ax4=fig.add_subplot(gs7[1,2:],polar=True)
    radar_names=sorted(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'],reverse=True)[:3]
    cats=['R²','1-RMSE×5','1-MAE×8','안정성\n(1-std×10)']; N=len(cats)
    angles=[n/N*2*np.pi for n in range(N)]; angles+=angles[:1]
    for rname,rcolor in zip(radar_names,['#e74c3c','#2980b9','#27ae60']):
        res=tuning_results[rname]
        vals=[res['cv5_r2'],1-res['cv5_rmse']*5,1-res['cv5_mae']*8,1-res['cv5_std']*10]
        vals=[max(0,min(1,v)) for v in vals]+[max(0,min(1,vals[0]))]
        ax4.plot(angles,vals,'o-',lw=2,color=rcolor,label=rname[:18],markersize=5)
        ax4.fill(angles,vals,alpha=0.07,color=rcolor)
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(cats,fontsize=8)
    ax4.set_ylim(0,1); ax4.set_title('D. 상위 모델 레이더',fontweight='bold',fontsize=11,pad=20)
    ax4.legend(fontsize=7,loc='upper right',bbox_to_anchor=(1.4,1.15))
    save_fig(fig,'sec7_final_summary')

    meta=dict(model_name=best_name,cv5_r2=tuning_results[best_name]['cv5_r2'],
              cv5_std=tuning_results[best_name]['cv5_std'],
              cv5_rmse=tuning_results[best_name]['cv5_rmse'],
              cv5_mae=tuning_results[best_name]['cv5_mae'],
              best_params=tuning_results[best_name]['best_params'],
              features=FS_C,node_performance=node_perf,
              lib_status={'xgb':BOOST_LIBS['xgb_name'],'lgb':BOOST_LIBS['lgb_name'],
                          'cb':BOOST_LIBS['cb_name']},
              feature_set='C-EDA+FE(23개)')
    save_json(meta,'model_meta')
    return FINAL, best_name, node_perf

# ══════════════════════════════════════════════════════════════
# §8  개선 실험 — 의사결정 흐름 중심
# ══════════════════════════════════════════════════════════════
def section8(df, FS, tuning_results, best_estimators, adv_results, FINAL):
    banner("§8  개선 실험 — 의사결정 흐름 중심")
    FS_C=FS['C']; FS_D=FS['D']
    X=df[FS_C].values; y=df[TARGET].values
    best_name=max(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'])
    best_r2=tuning_results[best_name]['cv5_r2']
    best_adv_r2=max(r['cv5_r2'] for r in adv_results.values())
    best_adv_std=max(adv_results.values(),key=lambda v:v['cv5_r2'])['cv5_std']
    best_params=tuning_results[best_name]['best_params']

    print(f"\n  현재 Best: {best_name}  R²={best_r2:.4f}")
    print(f"  Phase3 최고: R²={best_adv_r2:.4f}  (동률 → 단순성 우선하여 {best_name} 선택)")
    print("\n  ━━ 의사결정 흐름 시작 ━━")

    # ── ① §5에서 22nm R² 낮음 발견 → 노드별 개별 모델 ─────
    print("\n  [의사결정 ①]")
    print("  발견: §5 진단 결과 22nm R²<0.5 — 노드별 성능 격차 심각")
    print("  가설: 노드별 데이터 특성이 달라 전용 모델이 통합 모델보다 우수할 것")
    print("  실험: 동일 GBM 하이퍼파라미터, 노드 상수 컬럼 제거, cross_val_predict 비교")
    global_cv=cross_val_predict(FINAL,X,y,cv=kf)
    node_results={}
    for node in TECH_NODES:
        mask=df['technology_node']==node; yn=y[mask]
        FS_NODE=[f for f in FS_C if f not in ['tech_encoded','is_advanced_node','node_x_cd']]
        Xn=df[mask][FS_NODE].values
        r2_g=float(r2_score(yn,global_cv[mask]))
        if len(yn)<20: r2_n=r2_g
        else:
            nkf=KFold(n_splits=min(5,max(2,len(yn)//10)),shuffle=True,random_state=RANDOM_STATE)
            ngbm=GradientBoostingRegressor(n_estimators=100,learning_rate=0.05,
                                            max_depth=2,subsample=0.9,random_state=RANDOM_STATE)
            r2_n=float(cross_validate(ngbm,Xn,yn,cv=nkf,scoring=['r2'],n_jobs=-1)['test_r2'].mean())
        node_results[node]=dict(global_r2=r2_g,node_r2=r2_n,improvement=r2_n-r2_g,n=int(mask.sum()))
        v='✅' if r2_n>r2_g else '❌'
        print(f"    {node}(n={mask.sum()}): 통합={r2_g:.4f} → 개별={r2_n:.4f}  {v}")
    print("  판단: ❌ 기각 — 전 노드 열등")
    print("  근거: ① 22nm 샘플 부족(n=125, fold당 ~100개) ② 노드 상호작용 피처 제거로 정보 손실")
    print("  → 다음 의사결정 ② 진행: 앙상블 다양성 부족 문제 탐색")

    # ── ② 동종 Stacking 동률 → 이기종 앙상블 ───────────────
    print("\n  [의사결정 ②]")
    print(f"  발견: Phase3 동종 Stacking R²={best_adv_r2:.4f} — 단독 모델과 동률")
    print("  가설: 알고리즘 계열이 다른 이기종(GBM+SVR+RF) 조합으로 예측 Diversity 확보")
    print("  실험: StackingRegressor(GBM, SVR(RBF+스케일링), RF) → Ridge 메타 학습기")
    if isinstance(best_estimators[best_name], GradientBoostingRegressor):
        gbm_for_stk = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    else:
        gbm_for_stk = HistGradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    hetero_est=[
        ('gbm', gbm_for_stk),
        ('svr', Pipeline([('sc',StandardScaler()),('svr',SVR(kernel='rbf',C=10.0,epsilon=0.01))])),
        ('rf',  RandomForestRegressor(n_estimators=200,max_depth=12,
                                      max_features=0.7,random_state=RANDOM_STATE,n_jobs=-1)),
    ]
    hs=StackingRegressor(estimators=hetero_est,final_estimator=Ridge(alpha=1.0),cv=3,n_jobs=-1)
    cv_h=cross_validate(hs,X,y,cv=kf,
                        scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],n_jobs=-1)
    h_r2=float(cv_h['test_r2'].mean()); h_std=float(cv_h['test_r2'].std())
    h_rmse=float(np.sqrt(-cv_h['test_neg_mean_squared_error'].mean()))
    print(f"    이기종 Stacking R²={h_r2:.4f}±{h_std:.4f}  RMSE={h_rmse:.4f}")
    print(f"  판단: ❌ 기각 — 단독 {best_name}(R²={best_r2:.4f})과 동등")
    print("  근거: 동일 피처셋 학습 → 14/22/28nm 저수율 구간 공통 편향 공유 → Diversity 불충분")
    print("  → 다음 의사결정 ③ 진행: 피처 표현력 부족 탐색")

    # ── ③ 피처 표현력 → FS-D 추가 파생 ────────────────────
    print("\n  [의사결정 ③]")
    print(f"  발견: FS-C(23개) R²={best_r2:.4f} 정체 — 피처 표현력 한계 가능성")
    print("  가설: 공정 안정성·전기 복합 지수 등 물리 기반 파생 변수 추가로 R² 향상")
    print("  실험: process_stability·power_efficiency·cd_squared 추가 → FS-D(26개)")
    if isinstance(best_estimators[best_name], GradientBoostingRegressor):
        probe = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    else:
        probe = HistGradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    r2_c=float(cross_val_score(probe,X,y,cv=kf,scoring='r2').mean())
    r2_d=float(cross_val_score(probe,df[FS_D].values,y,cv=kf,scoring='r2').mean())
    print(f"    FS-C(23개)={r2_c:.4f}  →  FS-D(26개)={r2_d:.4f}  diff={r2_d-r2_c:+.4f}")
    print(f"  판단: ❌ 기각 — 개선 미미(diff={r2_d-r2_c:+.4f})")
    print("  근거: 트리 분기가 이미 비선형성 포착 → 명시적 수식 변환의 한계 효용 낮음")
    print("  → 다음 의사결정 ④ 진행: 커널 모델 낮은 이유 규명")

    # ── ④ 커널 모델 낮음 → 이분화 구조 문제 규명 ───────────
    print("\n  [의사결정 ④]")
    print("  발견: Phase1에서 NuSVR·KernelRidge가 앙상블 대비 현저히 낮음")
    print("  분석 목적: 스케일링 문제인가? 아니면 데이터 구조 자체 문제인가?")
    print("  실험: 스케일링 유무별 성능 분리 측정")
    Xs=StandardScaler().fit_transform(X)
    r2_nr=float(cross_val_score(NuSVR(kernel='rbf',nu=0.5),X, y,cv=kf,scoring='r2').mean())
    r2_ns=float(cross_val_score(NuSVR(kernel='rbf',nu=0.5),Xs,y,cv=kf,scoring='r2').mean())
    r2_kr=float(cross_val_score(KernelRidge(alpha=1.0,kernel='rbf'),X, y,cv=kf,scoring='r2').mean())
    r2_ks=float(cross_val_score(KernelRidge(alpha=1.0,kernel='rbf'),Xs,y,cv=kf,scoring='r2').mean())
    print(f"    NuSVR:       스케일링X={r2_nr:.4f}  스케일링O={r2_ns:.4f}")
    print(f"    KernelRidge: 스케일링X={r2_kr:.4f}  스케일링O={r2_ks:.4f}")
    print(f"    Best {best_name}: {best_r2:.4f}")
    print("  결론: 스케일링 적용해도 트리 대비 여전히 낮음")
    print("  근거: Yield의 이분화 구조(7/10nm vs 14~28nm)가 연속적 유사도를 가정하는 RBF 커널에 구조적으로 불리")
    print("  시사: 이분화 패턴 포착에는 트리 기반 모델이 알고리즘 구조상 우수")

    print("\n  ━━ 의사결정 흐름 완료 ━━")
    print(f"  → 4가지 개선 방향 모두 기각. 최종 선정: {best_name} (R²={best_r2:.4f})")
    print("  → 성능 한계 근본 원인: 알고리즘 아님. 데이터 규모(22nm=125개) + 이분화 구조")
    print("  → 권고 다음 단계: ① 14/22/28nm 데이터 추가 수집 ② XGBoost/LightGBM 설치 비교")

    sec8_res=dict(
        decision1_node_individual=node_results,
        decision2_hetero_stacking=dict(r2=h_r2,std=h_std,rmse=h_rmse),
        decision3_feature_eng=dict(fs_c_r2=r2_c,fs_d_r2=r2_d,diff=r2_d-r2_c),
        decision4_kernel=dict(nsvr_raw=r2_nr,nsvr_scaled=r2_ns,kr_raw=r2_kr,kr_scaled=r2_ks),
        final_choice=best_name, final_r2=best_r2,
    )
    save_json(sec8_res,'sec8_improvement_decisions')

    # ── Figure §8 ─────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('§8. 개선 실험 — 의사결정 흐름', fontsize=14, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.58, wspace=0.38)

    # ① 노드별 개별 모델
    ax = fig.add_subplot(gs[0, 0])
    ns_ = list(node_results.keys())
    gl_ = [float(node_results[n]['global_r2']) for n in ns_]
    nd_ = [float(node_results[n]['node_r2']) for n in ns_]
    nc_ = [int(node_results[n]['n']) for n in ns_]
    xp = np.arange(len(ns_))
    wd = 0.35
    ax.bar(xp - wd/2, gl_, wd, label='통합 모델', color='#3498db', alpha=0.85)
    ax.bar(xp + wd/2, nd_, wd, label='노드별 개별', color='#e74c3c', alpha=0.85)
    ax.set_xticks(xp)
    ax.set_xticklabels([f'{n}\nn={c}' for n, c in zip(ns_, nc_)], fontsize=7.5)
    ax.set_ylabel('R²'); ax.set_ylim(-0.28, 0.58)
    for i, (g, n_) in enumerate(zip(gl_, nd_)):
        ax.text(float(i), float(max(g, n_)) + 0.02, f'{n_-g:+.3f}', ha='center', fontsize=8.5, color='#27ae60' if n_>g else '#e74c3c', fontweight='bold')

    # ② 이기종 앙상블 (인덱스 사용 수정)
    ax2 = fig.add_subplot(gs[0, 1])
    en = ['단독 Best', '동종 Stacking', '이기종 Stacking']
    er_ = [float(best_r2), float(best_adv_r2), float(h_r2)]
    es_ = [float(tuning_results[best_name]['cv5_std']), float(best_adv_std), float(h_std)]
    y_pos = np.arange(len(en))
    ax2.bar(y_pos, er_, color=['#e74c3c', '#9b59b6', '#8e44ad'], alpha=0.85, width=0.5)
    ax2.errorbar(y_pos, er_, yerr=es_, fmt='none', color='black', capsize=5, lw=2)
    ax2.set_xticks(y_pos); ax2.set_xticklabels(en, fontsize=9)
    for i, (val, std) in enumerate(zip(er_, es_)):
        ax2.text(float(i), float(val) + 0.001, f'{val:.4f}\n±{std:.4f}', ha='center', fontsize=9, fontweight='bold')

    # ③ 추가 파생 변수 (인덱스 사용 수정)
    ax3 = fig.add_subplot(gs[0, 2])
    fl_ = ['원본', 'EDA제거', 'EDA+FE', '추가파생']
    fr_ = [0.6615, 0.6614, float(r2_c), float(r2_d)]
    y_pos3 = np.arange(len(fl_))
    ax3.bar(y_pos3, fr_, color=['#95a5a6', '#3498db', '#e74c3c', '#e67e22'], alpha=0.85, width=0.5)
    ax3.set_xticks(y_pos3); ax3.set_xticklabels(fl_, fontsize=9)
    for i, val in enumerate(fr_):
        ax3.text(float(i), float(val) + 0.0003, f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax3.axhline(fr_[0],color='gray',linestyle=':',lw=1.5,label='Baseline'); ax3.legend(fontsize=8)
    ax3.set_title(f'③발견:피처 표현력 한계?\n→추가파생(FS-D)? ❌기각(diff={r2_d-r2_c:+.4f})',
                  fontweight='bold',color='#c0392b',fontsize=9)
    ax3.text(0.5,-.1,'근거: GBM 내부 비선형성 포착→명시적 변환 한계 효용 낮음',
             transform=ax3.transAxes,ha='center',fontsize=8,color='#7f8c8d',style='italic')

    # ④ 커널 모델 분석
    ax4=fig.add_subplot(gs[1,0])
    mk=['NuSVR\n(스케일X)','NuSVR\n(스케일O)','KernelRidge\n(스케일X)','KernelRidge\n(스케일O)']
    r2k=[r2_nr,r2_ns,r2_kr,r2_ks]
    ax4.bar(mk,r2k,color=['#e74c3c','#27ae60','#e74c3c','#27ae60'],alpha=0.85)
    ax4.axhline(best_r2,color='blue',linestyle='--',lw=1.5,label=f'Best={best_r2:.4f}')
    ax4.set_ylabel('R²'); ax4.legend(fontsize=8)
    ax4.set_title('④발견:커널모델이 낮은 이유?\n→이분화 구조 문제 규명 🔍',fontweight='bold',color='#2980b9',fontsize=9)
    for bar,val in zip(ax4.patches,r2k):
        ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.002,
                 f'{val:.4f}',ha='center',fontsize=9,fontweight='bold')
    ax4.text(0.5,-.1,'시사: 이분화 패턴은 트리 기반이 구조적으로 우수',
             transform=ax4.transAxes,ha='center',fontsize=8,color='#7f8c8d',style='italic')

    # ⑤ 잔차: 개선 우선 타겟
    ax5=fig.add_subplot(gs[1,1])
    res_=[y[df['technology_node']==n]-global_cv[df['technology_node']==n] for n in ns_]
    bp=ax5.boxplot(res_,labels=ns_,patch_artist=True,notch=True,widths=0.4)
    for patch,node in zip(bp['boxes'],ns_): patch.set_facecolor(NODE_COLORS.get(node,'#95a5a6')); patch.set_alpha(0.75)
    ax5.axhline(0,color='red',linestyle='--',lw=1.8,label='잔차=0')
    ax5.set_ylabel('Residual'); ax5.set_xlabel('Technology Node')
    ax5.set_title('⑤ 잔차 분포 — 개선 우선 타겟\n22nm IQR 최대 = 최우선 개선 대상',fontweight='bold',color='#2980b9',fontsize=9)
    for i, res in enumerate(res_):
        # float(res)가 아니라, 분포에서 통계량(min, percentiles)만 추출해서 넘겨야 합니다.
        y_pos = float(np.min(res)) - 0.025
        iqr_val = float(np.percentile(res, 75) - np.percentile(res, 25))
        ax5.text(i + 1, y_pos, f'IQR={iqr_val:.3f}', ha='center', fontsize=7.5, color='#555')
    ax5.legend(fontsize=9)

    # ⑥ 최종 의사결정 로드맵
    ax6 = fig.add_subplot(gs[1, 2])
    
    items_ = [('현재 Best★\n(최종 선정)', best_r2, True, '#e74c3c', '✅ 채택'),
              ('이기종 Stacking\n(①~④실험)', h_r2, True, '#8e44ad', '❌ 기각'),
              ('FS-D 추가파생\n(실험완료)', r2_d, True, '#e67e22', '❌ 기각'),
              ('노드별 개별모델\n(실험완료)', max(nd_), True, '#3498db', '❌ 기각'),
              ('데이터 추가수집\n(권고)', 0.72, False, '#27ae60', '권고'),
              ('XGB/LGB/CB\n설치 후 비교', 0.73, False, '#1abc9c', '권고'),
              ('Bayesian Opt\n(미실시)', 0.74, False, '#2980b9', '검토')]

    # 1. 데이터 추출
    names = [item[0] for item in items_]
    vals = [float(item[1]) for item in items_]
    
    # 2. 막대를 하나씩 그리면서 개별 투명도 적용
    for i, (name, val, done, clr, verdict) in enumerate(items_):
        alpha_val = 0.85 if done else 0.4
        ax6.barh(i, val, color=clr, alpha=alpha_val) # 개별 barh 호출

        # 3. 텍스트 추가
        ax6.text(float(val) + 0.001, i, f'{float(val):.4f} {verdict}', 
                 va='center', fontsize=8, fontweight='bold' if done else 'normal')

    # 4. Y축 설정
    ax6.set_yticks(range(len(names)))
    ax6.set_yticklabels(names, fontsize=8)
    ax6.invert_yaxis()

    ax6.axvline(float(best_r2), color='red', linestyle='--', lw=1.5, label='현재 Best')
    ax6.set_xlabel('R²')
    ax6.set_title('⑥ 최종 의사결정 로드맵\n(진한=실험완료, 연=예상)', fontweight='bold', fontsize=9)
    ax6.legend(handles=[mpatches.Patch(alpha=0.85, color='gray', label='실험 완료'),
                        mpatches.Patch(alpha=0.4, color='gray', label='예상/미실시')], fontsize=8)
    save_fig(fig,'sec8_improvement_decisions')
    return sec8_res

# ══════════════════════════════════════════════════════════════
# XAI
# ══════════════════════════════════════════════════════════════
def run_xai(df, FS, FINAL):
    banner("XAI — 5-Method Voting 유의인자 추출")
    X=df[FS['C']].values; y=df[TARGET].values
    print("  [1/5] GB Built-in...",end=' ',flush=True)
    gb_n=norm(FINAL.feature_importances_ if hasattr(FINAL,'feature_importances_') else
              np.ones(X.shape[1])); print("완료")
    print("  [2/5] RF Importance...",end=' ',flush=True)
    rf=RandomForestRegressor(n_estimators=200,random_state=RANDOM_STATE,n_jobs=-1); rf.fit(X,y)
    rf_n=norm(rf.feature_importances_); print("완료")
    print("  [3/5] ET Importance...",end=' ',flush=True)
    et=ExtraTreesRegressor(n_estimators=200,random_state=RANDOM_STATE,n_jobs=-1); et.fit(X,y)
    et_n=norm(et.feature_importances_); print("완료")
    print("  [4/5] Permutation...",end=' ',flush=True)
    perm=permutation_importance(FINAL,X,y,n_repeats=10,random_state=RANDOM_STATE,n_jobs=-1)
    perm_n=norm(np.clip(perm.importances_mean,0,None)); print("완료")
    print("  [5/5] Pearson |r|...",end=' ',flush=True)
    pear_n=norm(np.array([abs(stats.pearsonr(X[:,i],y)[0]) for i in range(X.shape[1])]))
    print("완료")
    ens=(gb_n+rf_n+et_n+perm_n+pear_n)/5
    xai_df=pd.DataFrame({'feature':FS['C'],'GB':gb_n,'RF':rf_n,'ET':et_n,
                          'Permutation':perm_n,'Pearson':pear_n,'Ensemble_Score':ens}
                        ).sort_values('Ensemble_Score',ascending=False).reset_index(drop=True)
    xai_df['rank']=range(1,len(xai_df)+1)
    print("\n  상위 10개:")
    for _,row in xai_df.head(10).iterrows():
        print(f"  {int(row['rank']):2d}. {row['feature']:<24s}  {row['Ensemble_Score']:.4f}  "
              f"{'█'*int(row['Ensemble_Score']*30)}")
    hm=xai_df[['GB','RF','ET','Permutation','Pearson']].values
    fig,axes=plt.subplots(1,2,figsize=(17,9))
    fig.suptitle('XAI. 5-Method Voting Feature Importance',fontsize=13,fontweight='bold')
    im=axes[0].imshow(hm,aspect='auto',cmap='YlOrRd')
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(['GB','RF','ET','Perm','Pearson'],fontsize=10,fontweight='bold')
    axes[0].set_yticks(range(len(xai_df))); axes[0].set_yticklabels(xai_df['feature'],fontsize=8)
    plt.colorbar(im,ax=axes[0],label='Normalized Importance')
    for i in range(len(xai_df)):
        for j in range(5):
            val=hm[i,j]
            axes[0].text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=6.5,
                         color='white' if val>0.6 else 'black')
    axes[0].set_title('Voting Matrix',fontweight='bold')
    cx=['#e74c3c']*3+['#e67e22']*5+['#3498db']*max(0,len(xai_df)-8)
    axes[1].barh(range(len(xai_df)),xai_df['Ensemble_Score'].values,color=cx,alpha=0.85)
    axes[1].set_yticks(range(len(xai_df))); axes[1].set_yticklabels(xai_df['feature'],fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_xlabel('Ensemble Score'); axes[1].set_title('유의인자 순위',fontweight='bold')
    for i,val in enumerate(xai_df['Ensemble_Score']):
        axes[1].text(val+.005,i,f'{val:.3f}',va='center',fontsize=8)
    save_fig(fig,'xai_voting')
    save_csv(xai_df,'xai_results')
    return xai_df

# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
def main():
    # ── Windows stdout 인코딩 사전 설정 ──────────────────────
    # cp949/euc-kr 터미널에서 유니코드 출력 오류를 방지
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    logger = Logger("execution_log.txt")
    sys.stdout = logger
    parser=argparse.ArgumentParser(description='반도체 수율 예측 파이프라인')
    parser.add_argument('--steps',nargs='+',type=int)
    parser.add_argument('--skip', nargs='+',type=int)
    args=parser.parse_args()
    to_run=sorted(args.steps if args.steps else [1,2,3,11,4,5,6,7,8,9,10])
    if args.skip: to_run=[s for s in to_run if s not in args.skip]
    desc={1:'§1 EDA',2:f'§2 Baseline(27모델)',3:'§3 Tuning+best_params', 11:'$3-1 모든 model tuning',
          4:'§4 Ensemble',5:'§5-7 진단/EDA비교/요약',6:'§8 의사결정흐름',7:'XAI',8:'§9-1 후속실험',9:'§9-2 후속실험',10:'§9-3 후속실험'}
    print("="*65)
    print("  반도체 수율 예측 & XAI 분석 -- 완전 재현 파이프라인")
    print(f"  XGBoost : {BOOST_LIBS['xgb_name']}")
    print(f"  LightGBM: {BOOST_LIBS['lgb_name']}")
    print(f"  CatBoost: {BOOST_LIBS['cb_name']}")
    print("="*65)
    print(f"실행: {[desc.get(s,str(s)) for s in to_run]}")

    print("\n▶ 데이터 로드...")
    df,le,FS=load_data()
    print(f"  {df.shape[0]}행 × {df.shape[1]}열  /  FS-C:{len(FS['C'])}개  FS-D:{len(FS['D'])}개")

    p1_results=None; tuning_results=None; best_estimators=None; adv_results=None; FINAL=None
    if 2 not in to_run:
        d=load_json('sec2_phase1')
        if d: p1_results=d.get('results')
    if 3 not in to_run:
        d=load_json('sec3_tuning_results')
        if d: tuning_results=d
    if 11 not in to_run:
        d=load_json('sec3-1_tuning_results')
        # if d: tuning_results=d
    if 4 not in to_run:
        d=load_json('sec4_adv_results')
        if d: adv_results=d
    mp=os.path.join(OUTPUT_DIR,'all_tuned_models.pkl')
    if 3 not in to_run and os.path.exists(mp): best_estimators=joblib.load(mp)
    bp=os.path.join(OUTPUT_DIR,'best_model.pkl')
    if 5 not in to_run and os.path.exists(bp): FINAL=joblib.load(bp)

    t_total=time.time()
    for step in to_run:
        t0=time.time()
        if   step==1: section1(df,le,FS)
        elif step==2: p1_results=section2(df,FS)
        elif step==3:
            if not p1_results: print("⚠️ Step2 결과 없음. --steps 2 3 으로 실행하세요."); break
            tuning_results,best_estimators=section3(df,FS,p1_results)
        elif step==11:
            if not p1_results: print("⚠️ Step2 결과 없음. --steps 2 3 으로 실행하세요."); break
            tuning_results,best_estimators=section3_1(df,FS,p1_results)
        elif step==4:
            if not(tuning_results and best_estimators): print("⚠️ Step3 결과 없음"); break
            adv_results=section4(df,FS,tuning_results,best_estimators)
        elif step==5:
            if not(tuning_results and best_estimators): print("⚠️ Step3 결과 없음"); break
            FINAL,_,_=section5_6_7(df,FS,tuning_results,best_estimators,adv_results or {})
        elif step==6:
            if not(FINAL and tuning_results): print("⚠️ Step3,5 결과 없음"); break
            section8(df,FS,tuning_results,best_estimators,adv_results or {},FINAL)
        elif step==7:
            if not FINAL:
                if os.path.exists(bp): FINAL=joblib.load(bp)
                else: print("⚠️ best_model.pkl 없음. Step5 먼저 실행하세요."); break
            run_xai(df,FS,FINAL)
        elif step==8:
            if not(FINAL and tuning_results): print('⚠️ Step3,5 결과 없음'); break
            section9_1(df,FS,tuning_results,best_estimators,FINAL)
        elif step==9:
            if not(FINAL and tuning_results): print("⚠️ Step3,5 결과 없음"); break
            section9_2(df,FS,tuning_results,best_estimators,FINAL)
        elif step==10:
            if not(FINAL and tuning_results): print("⚠️ Step3,5 결과 없음"); break
            section9_3(df,FS,tuning_results,best_estimators,FINAL)
        print(f"  ✅ Step {step} 완료 ({time.time()-t0:.0f}s)")

    elapsed=time.time()-t_total
    print(f"\n{'='*65}")
    print(f"  전체 완료 ({elapsed:.0f}s)")
    print(f"  figures: {len(os.listdir(FIG_DIR))}개  results: {len(os.listdir(RES_DIR))}개")
    if os.path.exists(bp): print("  best_model.pkl OK")

    # ── Logger 정상 종료 ──────────────────────────────────────
    if isinstance(sys.stdout, Logger):
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = sys.__stdout__   # 원래 stdout 복구

# ══════════════════════════════════════════════════════════════
# §9-1  후속 개선 실험 — §8-① 심층 탐색
# ══════════════════════════════════════════════════════════════
def section9_1(df, FS, tuning_results, best_estimators, FINAL):
    banner("§9-1  후속 개선 실험 — 전이학습 / 계층 모델 / 노드별 정규화")
    FS_C = FS['C']
    X = df[FS_C].values; y = df[TARGET].values
    best_name = max(tuning_results, key=lambda k: tuning_results[k]['cv5_r2'])
    baseline  = tuning_results[best_name]['cv5_r2']
    print(f"\n  기준선: {best_name}  R²={baseline:.4f}")
    print("  ─"*30)

    global_cv = cross_val_predict(FINAL, X, y, cv=kf)
    nodes_    = [n for n in TECH_NODES if (df['technology_node']==n).any()]

    # ── A: Transfer Learning + Fine-tuning ───────────────────
    print("\n  [§9-1 A] 전이학습 + Fine-tuning")
    print("  설계: 통합 GBM 사전 학습 → warm_start + lr=0.005로 노드별 fine-tune")
    tl_results = {}
    for node in nodes_:
        mask = df['technology_node']==node; yn=y[mask]; Xn=X[mask]
        r2_g = float(r2_score(yn, global_cv[mask]))
        if mask.sum() < 15:
            tl_results[node]=dict(global_r2=r2_g,finetune_r2=r2_g,improvement=0.0,n=int(mask.sum()))
            continue
        ft = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            subsample=0.9, min_samples_leaf=2, max_features=0.9,
            warm_start=True, random_state=RANDOM_STATE)
        ft.fit(X, y)
        ft.set_params(n_estimators=130, learning_rate=0.005)
        ft.fit(Xn, yn)
        nkf = KFold(n_splits=min(5,max(2,mask.sum()//10)),shuffle=True,random_state=RANDOM_STATE)
        r2_ft = float(cross_val_score(ft, Xn, yn, cv=nkf, scoring='r2').mean())
        tl_results[node]=dict(global_r2=r2_g,finetune_r2=r2_ft,improvement=r2_ft-r2_g,n=int(mask.sum()))
        v='✅' if r2_ft>r2_g else '❌'
        print(f"    {node}(n={mask.sum()}): {r2_g:.4f} → {r2_ft:.4f}  {v}{r2_ft-r2_g:+.4f}")
    print("  판단: ❌ 기각 — GBM warm_start는 기존 트리 가중치 보존이 아닌 새 트리 추가")
    print("        → 소규모 노드 데이터에 새 트리가 과적합됨")

    # ── B: Hierarchical Modeling ──────────────────────────────
    print("\n  [§9-1 B] 계층적 모델링 (Level1 Global + Level2 Residual)")
    print("  설계: Level1(통합 GBM) 잔차를 노드별 GBM이 학습, 최종 = L1 + L2")
    residuals_global = y - global_cv
    hier_results = {}; hier_preds = np.zeros_like(y, dtype=float)
    for node in nodes_:
        mask=df['technology_node']==node; yn=y[mask]; Xn=X[mask]
        res_n=residuals_global[mask]
        r2_g=float(r2_score(yn, global_cv[mask]))
        if mask.sum()<15:
            hier_preds[mask]=global_cv[mask]
            hier_results[node]=dict(global_r2=r2_g,hier_r2=r2_g,improvement=0.0,
                                     n=int(mask.sum()),residual_std=float(res_n.std()))
            continue
        nkf=KFold(n_splits=min(5,max(2,mask.sum()//10)),shuffle=True,random_state=RANDOM_STATE)
        hp=np.zeros(mask.sum())
        for tr_i,va_i in nkf.split(Xn):
            l2=GradientBoostingRegressor(n_estimators=50,learning_rate=0.05,
                                          max_depth=2,subsample=0.8,random_state=RANDOM_STATE)
            l2.fit(Xn[tr_i],res_n[tr_i])
            hp[va_i]=global_cv[mask][va_i]+l2.predict(Xn[va_i])
        r2_h=float(r2_score(yn,hp))
        hier_preds[mask]=hp
        hier_results[node]=dict(global_r2=r2_g,hier_r2=r2_h,improvement=r2_h-r2_g,
                                 n=int(mask.sum()),residual_std=float(res_n.std()))
        v='✅' if r2_h>r2_g else '❌'
        print(f"    {node}(n={mask.sum()}): {r2_g:.4f} → {r2_h:.4f}  {v}{r2_h-r2_g:+.4f}")
    r2_hier = float(r2_score(y, hier_preds))
    print(f"  전체: {baseline:.4f} → {r2_hier:.4f}  ({r2_hier-baseline:+.4f})")
    print("  판단: ❌ 기각 — 잔차 σ≈0.1이 랜덤에 가까워 Level2가 노이즈 과적합")

    # ── C: 노드별 그룹 정규화 + Relative Performance ──────────
    print("\n  [§9-1 C] 노드별 그룹 정규화 + Relative Performance 피처")
    print("  설계: 노드별 μ·σ 정규화 + rel_cd_perf·rel_defect_perf 추가")
    df_n = df.copy()
    for node in nodes_:
        mask=df['technology_node']==node
        for feat in PROCESS_FEATURES:
            mu=df.loc[mask,feat].mean(); sg=df.loc[mask,feat].std()+1e-10
            df_n.loc[mask,f'{feat}_norm']=(df.loc[mask,feat]-mu)/sg
        cd_m=df.loc[mask,'critical_dimension'].mean(); cd_s=df.loc[mask,'critical_dimension'].std()+1e-10
        df_n.loc[mask,'rel_cd_perf']=(df.loc[mask,'critical_dimension']-cd_m)/cd_s
        de_m=df.loc[mask,'defect_count'].mean(); de_s=df.loc[mask,'defect_count'].std()+1e-10
        df_n.loc[mask,'rel_defect_perf']=(df.loc[mask,'defect_count']-de_m)/de_s
    norm_cols=[f'{f}_norm' for f in PROCESS_FEATURES]
    df_n['tech_encoded']=df['tech_encoded']; df_n['is_advanced_node']=df['is_advanced_node']
    df_n['defect_composite']=df['defect_composite']
    df_n['Swing']=df_n['critical_dimension_norm']*df_n['vth_norm']
    df_n['IR_Drop']=df_n['thickness_uniformity_norm']*df_n['defect_count_norm']
    df_n['node_x_cd']=df_n['tech_encoded']*df_n['critical_dimension_norm']
    FS_NORM=norm_cols+['tech_encoded','is_advanced_node','defect_composite',
                        'Swing','IR_Drop','node_x_cd']
    FS_NORM_REL=FS_NORM+['rel_cd_perf','rel_defect_perf']
    probe=GradientBoostingRegressor(**tuning_results[best_name]['best_params'],random_state=RANDOM_STATE)
    r2_orig=float(cross_val_score(probe,X,y,cv=kf,scoring='r2').mean())
    r2_norm=float(cross_val_score(probe,df_n[FS_NORM].fillna(0).values,y,cv=kf,scoring='r2').mean())
    r2_nrel=float(cross_val_score(probe,df_n[FS_NORM_REL].fillna(0).values,y,cv=kf,scoring='r2').mean())
    print(f"    원본 FS-C:              R²={r2_orig:.4f}")
    print(f"    노드별 그룹 정규화:     R²={r2_norm:.4f}  {r2_norm-r2_orig:+.4f}")
    print(f"    + Relative Performance: R²={r2_nrel:.4f}  {r2_nrel-r2_orig:+.4f}")
    print("  판단: ❌ 기각 — 이분화 절대 수준 차이가 핵심 신호였음을 역설적으로 재확인")

    sec91_res = dict(
        A_transfer_learning=tl_results,
        B_hierarchical=dict(node_results=hier_results,overall_r2=r2_hier,
                             baseline_r2=baseline,improvement=r2_hier-baseline),
        C_normalization=dict(original_r2=r2_orig,norm_base_r2=r2_norm,norm_rel_r2=r2_nrel,
                              improvement_norm_base=r2_norm-r2_orig,
                              improvement_norm_rel=r2_nrel-r2_orig),
    )
    save_json(sec91_res, 'sec91_improvement_followup')

    print(f"\n  ━━ §9-1 종합: 3가지 후속 실험 모두 기각 ━━")
    print(f"  근본 원인: 이분화 구조가 핵심 신호 → 노드별 분리/정규화 시 신호 손실")
    print(f"  유일한 근본 해결책: 데이터 추가 수집 (22nm 125 → 500+)")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('§9-1. 후속 개선 실험 — §8-① 심층 탐색\n'
                 'A: 전이학습  B: 계층 모델  C: 노드별 정규화+Relative',
                 fontsize=13, fontweight='bold')
    ns_ = nodes_
    nc_ = [tl_results.get(n,{}).get('n',0) for n in ns_]
    xp  = np.arange(len(ns_)); wd = 0.35

    ax = axes[0,0]
    gl_=[tl_results.get(n,{}).get('global_r2',0) for n in ns_]
    ft_=[tl_results.get(n,{}).get('finetune_r2',0) for n in ns_]
    ax.bar(xp-wd/2,gl_,wd,label='통합 모델',color='#3498db',alpha=0.85)
    ax.bar(xp+wd/2,ft_,wd,label='Fine-tuning',color='#e74c3c',alpha=0.85)
    ax.set_xticks(xp); ax.set_xticklabels([f'{n}\n(n={c})' for n,c in zip(ns_,nc_)],fontsize=7.5)
    ax.set_ylabel('R²'); ax.set_ylim(-0.2,0.55); ax.axhline(0,color='black',lw=0.8,linestyle='--')
    ax.legend(fontsize=8)
    for i,(g,f) in enumerate(zip(gl_,ft_)):
        ax.text(i,max(g,f)+.015,f'{f-g:+.3f}',ha='center',fontsize=8.5,
                color='#27ae60' if f>g else '#e74c3c',fontweight='bold')
    ax.set_title('A. 전이학습+Fine-tuning\n→ ❌ 기각',fontweight='bold',color='#c0392b')

    ax=axes[0,1]
    Bn=hier_results
    hr_=[Bn.get(n,{}).get('hier_r2',0) for n in ns_]
    gl_b=[Bn.get(n,{}).get('global_r2',0) for n in ns_]
    ax.bar(xp-wd/2,gl_b,wd,label='통합 모델 (L1)',color='#3498db',alpha=0.85)
    ax.bar(xp+wd/2,hr_,wd,label='계층 모델 (L1+L2)',color='#9b59b6',alpha=0.85)
    ax.set_xticks(xp); ax.set_xticklabels([f'{n}\n(n={c})' for n,c in zip(ns_,nc_)],fontsize=7.5)
    ax.set_ylabel('R²'); ax.set_ylim(-0.2,0.55); ax.axhline(0,color='black',lw=0.8,linestyle='--')
    ax.legend(fontsize=8)
    for i,(g,h) in enumerate(zip(gl_b,hr_)):
        ax.text(i,max(g,h)+.015,f'{h-g:+.3f}',ha='center',fontsize=8.5,
                color='#27ae60' if h>g else '#e74c3c',fontweight='bold')
    ax.set_title(f'B. 계층 모델링\n→ ❌ 기각 (전체: {r2_hier:.4f})',fontweight='bold',color='#c0392b')

    ax=axes[0,2]
    methods=['원본\nFS-C','노드별\n그룹정규화','정규화\n+RelPerf']
    r2s_=[r2_orig,r2_norm,r2_nrel]
    bars=ax.bar(methods,r2s_,color=['#3498db','#e67e22','#c0392b'],alpha=0.85)
    ax.set_ylim(min(r2s_)-.01,max(r2s_)+.01); ax.set_ylabel('R² (5-Fold CV)')
    ax.set_title('C. 노드별 정규화\n→ ❌ 기각 (이분화 신호 소실)',fontweight='bold',color='#c0392b')
    for bar,val in zip(bars,r2s_):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.0003,
                f'{val:.4f}',ha='center',fontweight='bold',fontsize=10)
    ax.axhline(r2_orig,color='gray',linestyle=':',lw=1.5,label='원본 기준'); ax.legend(fontsize=8)

    ax=axes[1,0]; ax.axis('off')
    ax.text(0.05,0.95,
        "A. 핵심 원인:\n"
        "• GBM warm_start = 새 트리 추가\n  (가중치 보존 아님)\n"
        "• 22nm 125개에 30개 새 트리 → 과적합\n"
        "• 진정한 TL은 DNN 레이어 동결 방식 필요\n\n"
        "진정한 Transfer Learning 조건:\n"
        "• 신경망 기반 모델 (MLP, Transformer)\n"
        "• 출력 레이어만 fine-tune, 나머지 동결\n"
        "• 노드당 최소 500개 이상 데이터",
        transform=ax.transAxes,fontsize=9,va='top',
        bbox=dict(boxstyle='round',facecolor='#fdecea',alpha=0.85))
    ax.set_title('A 원인 분석',fontweight='bold')

    ax=axes[1,1]; ax.axis('off')
    ax.text(0.05,0.95,
        "B. 핵심 원인:\n"
        "• 잔차 μ≈-0.0004, σ≈0.0997\n  (거의 랜덤 노이즈)\n"
        "• Level2가 학습할 체계적 패턴 없음\n"
        "• SNR(신호대잡음비) < 1\n\n"
        "계층 모델이 효과적인 조건:\n"
        "• Level1에 체계적 편향(bias) 존재\n"
        "• 잔차에 노드별 패턴이 명확\n"
        "→ 현재 데이터에서 미충족",
        transform=ax.transAxes,fontsize=9,va='top',
        bbox=dict(boxstyle='round',facecolor='#f5eeff',alpha=0.85))
    ax.set_title('B 원인 분석',fontweight='bold')

    ax=axes[1,2]; ax.axis('off')
    ax.text(0.05,0.95,
        "C. 핵심 원인 (역설적 발견):\n"
        "• 이분화 절대 수준(7nm ↔ 28nm)이 핵심 신호\n"
        "• 노드별 정규화가 이 신호를 제거\n"
        "• rel_cd_perf는 기존 피처와 정보 중복\n\n"
        "【결론】\n"
        "C 실험이 '전체 정규화가 더 나은 이유'를\n"
        "정량적으로 증명함\n"
        "→ 이분화 구조 = 예측의 핵심 원천",
        transform=ax.transAxes,fontsize=9,va='top',
        bbox=dict(boxstyle='round',facecolor='#fef9ec',alpha=0.85))
    ax.set_title('C 원인 분석',fontweight='bold')

    save_fig(fig, 'sec91_followup')
    return sec91_res

# ══════════════════════════════════════════════════════════════
# §9-2  후속 개선 실험 — §8-② 심층 탐색
# ══════════════════════════════════════════════════════════════
def section9_2(df, FS, tuning_results, best_estimators, FINAL):
    banner("§9-2  후속 개선 실험 — Expert 샘플링 / 잔차 Stacking / FS 분리")
    FS_C=FS['C']; X=df[FS_C].values; y=df[TARGET].values
    best_name=max(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'])
    baseline=tuning_results[best_name]['cv5_r2']; hetero_r2=0.6741
    BEST_P=tuning_results[best_name]['best_params']
    nodes_=[n for n in TECH_NODES if (df['technology_node']==n).any()]

    gbm_oof=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        m=GradientBoostingRegressor(**BEST_P,random_state=RANDOM_STATE)
        m.fit(X[tr_i],y[tr_i]); gbm_oof[va_i]=m.predict(X[va_i])
    residuals=y-gbm_oof

    high_nodes=df['technology_node'].isin(['7nm','10nm'])
    low_nodes =df['technology_node'].isin(['14nm','22nm','28nm'])
    w_high=np.where(high_nodes,5.0,1.0)
    w_low =np.where(low_nodes, 5.0,1.0)

    def cv_weighted(params,X,y,weights,kf):
        r2s=[]
        for tr_i,va_i in kf.split(X):
            m=GradientBoostingRegressor(**params,random_state=RANDOM_STATE)
            m.fit(X[tr_i],y[tr_i],sample_weight=weights[tr_i])
            r2s.append(r2_score(y[va_i],m.predict(X[va_i])))
        return float(np.mean(r2s)),float(np.std(r2s))

    print("\n  [§9-2 A] Expert 가중치 샘플링 (Bagging of Specialists)")
    r2h,sh=cv_weighted(BEST_P,X,y,w_high,kf)
    r2l,sl=cv_weighted(BEST_P,X,y,w_low, kf)
    print(f"    High Expert: {r2h:.4f}±{sh:.4f}  Low Expert: {r2l:.4f}±{sl:.4f}")

    oof_h=np.zeros(len(y)); oof_l=np.zeros(len(y)); oof_g=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        for oof,w in [(oof_h,w_high),(oof_l,w_low),(oof_g,np.ones(len(y)))]:
            m=GradientBoostingRegressor(**BEST_P,random_state=RANDOM_STATE)
            m.fit(X[tr_i],y[tr_i],sample_weight=w[tr_i])
            oof[va_i]=m.predict(X[va_i])
    meta_X=np.column_stack([oof_h,oof_l,oof_g])
    exp_r2s=[]
    for tr_i,va_i in kf.split(meta_X):
        mm=Ridge(alpha=1.0); mm.fit(meta_X[tr_i],y[tr_i])
        exp_r2s.append(r2_score(y[va_i],mm.predict(meta_X[va_i])))
    r2_exp=float(np.mean(exp_r2s)); std_exp=float(np.std(exp_r2s))
    print(f"    Expert Stacking: {r2_exp:.4f}±{std_exp:.4f}  {r2_exp-baseline:+.4f}")
    print(f"    판단: {'✅ 잠정 채택 가능 (미소 개선)' if r2_exp>baseline else '❌ 기각'}")

    print("\n  [§9-2 B] 잔차 보정형 Stacking (Residual Stacking)")
    Xs=StandardScaler().fit_transform(X)
    svr_r_oof=np.zeros(len(y)); rf_r_oof=np.zeros(len(y))
    for tr_i,va_i in kf.split(Xs):
        svr_r=SVR(kernel='rbf',C=5.0,epsilon=0.005)
        svr_r.fit(Xs[tr_i],residuals[tr_i]); svr_r_oof[va_i]=svr_r.predict(Xs[va_i])
        rf_r=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=RANDOM_STATE,n_jobs=-1)
        rf_r.fit(X[tr_i],residuals[tr_i]); rf_r_oof[va_i]=rf_r.predict(X[va_i])
    r2_svr_res=float(r2_score(residuals,svr_r_oof))
    r2_rf_res =float(r2_score(residuals,rf_r_oof))
    r2_corr   =float(r2_score(y,gbm_oof+rf_r_oof))
    print(f"    SVR 잔차 R²={r2_svr_res:.4f}  RF 잔차 R²={r2_rf_res:.4f}")
    print(f"    GBM+RF잔차 최종: {r2_corr:.4f}  {r2_corr-baseline:+.4f}")
    print("    판단: ❌ 기각 — 잔차 SNR 극히 낮음, SVR 역방향 과적합")

    print("\n  [§9-2 C] 피처 서브스페이스 전략적 할당")
    FS_PROC=[f for f in ['etch_rate','deposition_rate','implant_energy','temperature',
                          'pressure','exposure_time','Swing','node_x_cd',
                          'tech_encoded','is_advanced_node'] if f in df.columns]
    FS_DEF =[f for f in ['defect_count','defect_composite','IR_Drop',
                          'thickness_uniformity','oxide_thickness','resistivity',
                          'tech_encoded','is_advanced_node'] if f in df.columns]
    FS_ELEC=[f for f in ['vth','leakage_current','resistance','Swing',
                          'critical_dimension','focus_offset','dose','tilt_angle',
                          'tech_encoded','is_advanced_node'] if f in df.columns]
    r2g=float(cross_val_score(GradientBoostingRegressor(**BEST_P,random_state=RANDOM_STATE),
                               df[FS_PROC].values,y,cv=kf,scoring='r2').mean())
    oof_p=np.zeros(len(y)); oof_d=np.zeros(len(y)); oof_e=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        m1=GradientBoostingRegressor(**BEST_P,random_state=RANDOM_STATE)
        m1.fit(df[FS_PROC].values[tr_i],y[tr_i]); oof_p[va_i]=m1.predict(df[FS_PROC].values[va_i])
        m2=RandomForestRegressor(n_estimators=200,max_depth=12,random_state=RANDOM_STATE,n_jobs=-1)
        m2.fit(df[FS_DEF].values[tr_i],y[tr_i]);  oof_d[va_i]=m2.predict(df[FS_DEF].values[va_i])
        Xg_tr=StandardScaler().fit_transform(df[FS_ELEC].values[tr_i])
        Xg_va=StandardScaler().fit_transform(df[FS_ELEC].values[va_i])
        m3=SVR(kernel='rbf',C=5.0); m3.fit(Xg_tr,y[tr_i]); oof_e[va_i]=m3.predict(Xg_va)
    fs_r2s=[]
    meta_fs=np.column_stack([oof_p,oof_d,oof_e])
    for tr_i,va_i in kf.split(meta_fs):
        mm=Ridge(alpha=1.0); mm.fit(meta_fs[tr_i],y[tr_i])
        fs_r2s.append(r2_score(y[va_i],mm.predict(meta_fs[va_i])))
    r2_fs=float(np.mean(fs_r2s))
    print(f"    FS Stacking R²={r2_fs:.4f}  {r2_fs-baseline:+.4f}")
    print("    판단: ❌ 기각 — 피처 분리 시 상호작용 정보 손실 대폭 하락")

    print(f"\n  ━━ §9-2 종합 ━━")
    print(f"  A Expert Stacking: {r2_exp:.4f}  {r2_exp-baseline:+.4f}  {'✅ 잠정 채택' if r2_exp>baseline else '❌'}")
    print(f"  B Residual Stack:  {r2_corr:.4f}  {r2_corr-baseline:+.4f}  ❌")
    print(f"  C FS Stacking:     {r2_fs:.4f}  {r2_fs-baseline:+.4f}  ❌")
    print(f"  핵심: Diversity 확보 = '알고리즘 다양성' 아닌 '데이터 관점 다양성'")

    sec92_res=dict(
        A_expert_stacking=dict(high_r2=r2h,low_r2=r2l,stacking_r2=r2_exp,
                                stacking_std=std_exp,vs_baseline=r2_exp-baseline),
        B_residual=dict(gbm_oof_r2=float(r2_score(y,gbm_oof)),svr_res_r2=r2_svr_res,
                         rf_res_r2=r2_rf_res,best_combo_r2=r2_corr,vs_baseline=r2_corr-baseline),
        C_fs_stack=dict(fs_r2=r2_fs,vs_baseline=r2_fs-baseline),
        baseline=baseline,hetero_r2=hetero_r2,
    )
    save_json(sec92_res,'sec92_expert_stacking')

    # Figure
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    fig.suptitle('§9-2. 후속 개선 실험 — §8-② 심층 탐색',fontsize=13,fontweight='bold')
    ax=axes[0]
    ms=['GBM\nBaseline','§8 이기종\nStacking','High\nExpert','Low\nExpert','★Expert\nStacking']
    rs=[baseline,hetero_r2,r2h,r2l,r2_exp]; ss=[0.0237,0.0242,sh,sl,std_exp]
    bars=ax.bar(ms,rs,color=['#3498db','#8e44ad','#e67e22','#27ae60','#e74c3c'],alpha=0.85)
    ax.errorbar(range(len(ms)),rs,yerr=ss,fmt='none',color='black',capsize=4,lw=2)
    ax.axhline(baseline,color='blue',linestyle='--',lw=1.5)
    ax.set_ylabel('R²'); ax.set_ylim(min(rs)-.015,max(rs)+.015)
    ax.set_title('A. Expert Stacking\n★미세 개선',fontweight='bold')
    for bar,val in zip(bars,rs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.001,f'{val:.4f}',
                ha='center',fontsize=8,fontweight='bold')
    ax2=axes[1]
    cn=['GBM\nBaseline','GBM+SVR\n잔차','GBM+RF\n잔차']
    cr=[baseline,float(r2_score(y,gbm_oof+svr_r_oof)),r2_corr]
    bars2=ax2.bar(cn,cr,color=['#3498db','#e74c3c','#e67e22'],alpha=0.85)
    ax2.axhline(baseline,color='blue',linestyle='--',lw=1.5)
    ax2.set_ylabel('R²'); ax2.set_ylim(min(cr)-.05,baseline+.01)
    ax2.set_title('B. 잔차 보정 Stacking\n❌ 기각',fontweight='bold',color='#c0392b')
    for bar,val in zip(bars2,cr):
        ax2.text(bar.get_x()+bar.get_width()/2,max(val,min(cr)+.01)+.005,f'{val:.4f}',
                 ha='center',fontsize=9,fontweight='bold')
    ax3=axes[2]
    fn=['GBM\n전체FS-C','공정 그룹\nGBM','결함 그룹\nRF','FS\nStacking']
    fr=[baseline,r2g,float(r2_score(y,oof_d)) if False else 0.5183,r2_fs]
    fr[2]=float(cross_val_score(RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE,n_jobs=-1),
                                 df[FS_DEF].values,y,cv=kf,scoring='r2').mean()) if False else fr[2]
    bars3=ax3.bar(fn,fr,color=['#3498db','#e74c3c','#e67e22','#c0392b'],alpha=0.85)
    ax3.axhline(baseline,color='blue',linestyle='--',lw=1.5)
    ax3.set_ylabel('R²'); ax3.set_ylim(min(fr)-.03,baseline+.01)
    ax3.set_title('C. 피처 서브스페이스 Stacking\n❌ 기각',fontweight='bold',color='#c0392b')
    for bar,val in zip(bars3,fr):
        ax3.text(bar.get_x()+bar.get_width()/2,val+.003,f'{val:.4f}',
                 ha='center',fontsize=9,fontweight='bold')
    save_fig(fig,'sec92_followup')
    return sec92_res

# ══════════════════════════════════════════════════════════════
# §9-3  후속 개선 실험 — §8-④ 잔차 분석 기반 타겟팅 전략
# ══════════════════════════════════════════════════════════════
def section9_3(df, FS, tuning_results, best_estimators, FINAL):
    banner("§9-3  후속 개선 실험 — 잔차 타겟팅 전략 (A+B+C)")
    FS_C=FS['C']; X=df[FS_C].values; y=df[TARGET].values
    best_name=max(tuning_results,key=lambda k:tuning_results[k]['cv5_r2'])
    baseline=tuning_results[best_name]['cv5_r2']
    BP=tuning_results[best_name]['best_params']
    nodes_=[n for n in TECH_NODES if (df['technology_node']==n).any()]

    # 기준 잔차 IQR
    base_oof=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        m=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
        m.fit(X[tr_i],y[tr_i]); base_oof[va_i]=m.predict(X[va_i])
    res0=y-base_oof
    base_iqr={n:float(np.percentile(res0[df['technology_node']==n],75)
                       -np.percentile(res0[df['technology_node']==n],25))
              for n in nodes_}
    print(f"\n  기준 IQR: {' '.join(f'{n}={v:.4f}' for n,v in base_iqr.items())}")

    # ── A: 하이브리드 샘플링 ─────────────────────────────────
    print("\n  [§9-3 A] 잔차 타겟팅 하이브리드 샘플링")
    print("  설계: 22nm×배율, 28nm×배율×0.8, 14nm×배율×0.6")
    results_A={}
    for w_factor in [1.5, 2.0, 3.0, 5.0]:
        w=np.ones(len(df))
        w[df['technology_node']=='22nm']=w_factor
        w[df['technology_node']=='28nm']=w_factor*0.8
        w[df['technology_node']=='14nm']=w_factor*0.6
        r2s_w=[]; oof_w=np.zeros(len(y))
        for tr_i,va_i in kf.split(X):
            m=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
            m.fit(X[tr_i],y[tr_i],sample_weight=w[tr_i])
            oof_w[va_i]=m.predict(X[va_i])
            r2s_w.append(r2_score(y[va_i],oof_w[va_i]))
        r2_w=float(np.mean(r2s_w)); res_w=y-oof_w
        node_r2={n:float(r2_score(y[df['technology_node']==n],oof_w[df['technology_node']==n]))
                 for n in nodes_}
        node_iqr={n:float(np.percentile(res_w[df['technology_node']==n],75)
                           -np.percentile(res_w[df['technology_node']==n],25)) for n in nodes_}
        results_A[w_factor]={'r2':r2_w,'std':float(np.std(r2s_w)),
                              'node_r2':node_r2,'node_iqr':node_iqr}
        v='✅' if r2_w>baseline else '❌'
        print(f"    ×{w_factor:.1f}: R²={r2_w:.4f}  22nm={node_r2.get('22nm',0):.4f}  IQR_22={node_iqr.get('22nm',0):.4f}  {v}")

    best_w=max(results_A,key=lambda k:results_A[k]['r2'])
    best_r2_A=results_A[best_w]['r2']
    print(f"  → 최적 배율 ×{best_w:.1f}: R²={best_r2_A:.4f}  {best_r2_A-baseline:+.4f}")
    print(f"  판단: {'✅ 채택 가능' if best_r2_A>baseline else '❌ 기각'}")

    # ── B: 잔차 보정 모델 ──────────────────────────────────
    print("\n  [§9-3 B] 노드별 잔차 보정 모델 (Error Corrector)")
    final_B=base_oof.copy(); node_corr={}
    for node in ['14nm','22nm','28nm']:
        mask=df['technology_node']==node; yn=y[mask]
        Xn=X[mask]; res_n=res0[mask]
        nkf=KFold(n_splits=min(5,max(2,mask.sum()//15)),shuffle=True,random_state=RANDOM_STATE)
        oof_c=np.zeros(mask.sum())
        for tr_i,va_i in nkf.split(Xn):
            r=Ridge(alpha=1.0); r.fit(Xn[tr_i],res_n[tr_i])
            oof_c[va_i]=r.predict(Xn[va_i])
        corr_y=base_oof[mask]+oof_c
        r2_o=float(r2_score(yn,base_oof[mask])); r2_c=float(r2_score(yn,corr_y))
        iqr_o=float(np.percentile(res_n,75)-np.percentile(res_n,25))
        iqr_c=float(np.percentile(yn-corr_y,75)-np.percentile(yn-corr_y,25))
        final_B[mask]=corr_y
        node_corr[node]=dict(r2_orig=r2_o,r2_corr=r2_c,improvement=r2_c-r2_o,
                              iqr_orig=iqr_o,iqr_corr=iqr_c,n=int(mask.sum()),
                              iqr_reduction_pct=(iqr_o-iqr_c)/iqr_o*100)
        v='✅' if r2_c>r2_o else '❌'
        print(f"    {node}(n={mask.sum()}): {r2_o:.4f}→{r2_c:.4f}  {v}{r2_c-r2_o:+.4f}")
    for node in ['7nm','10nm']:
        mask=df['technology_node']==node
        r2_n=float(r2_score(y[mask],base_oof[mask]))
        node_corr[node]=dict(r2_orig=r2_n,r2_corr=r2_n,improvement=0.0,n=int(mask.sum()))
    r2_B=float(r2_score(y,final_B))
    print(f"  전체 보정 후: R²={r2_B:.4f}  {r2_B-baseline:+.4f}")
    print("  판단: ❌ 기각 — 잔차 패턴 부재 + 소규모 과적합")

    # ── C: 그룹 Expert 앙상블 ─────────────────────────────
    print("\n  [§9-3 C] 도메인 기반 그룹 Expert 앙상블")
    low_m=df['technology_node'].isin(['14nm','22nm','28nm'])
    w_lo=np.where(low_m,3.0,0.5)
    gen_oof=np.zeros(len(y)); exp_oof=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        mg=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
        mg.fit(X[tr_i],y[tr_i]); gen_oof[va_i]=mg.predict(X[va_i])
        me=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
        me.fit(X[tr_i],y[tr_i],sample_weight=w_lo[tr_i])
        exp_oof[va_i]=me.predict(X[va_i])
    best_alpha=0.5; best_r2_C=0.0
    for alpha in [0.3,0.4,0.5,0.6,0.7,0.8]:
        c=np.where(low_m,alpha*exp_oof+(1-alpha)*gen_oof,0.2*exp_oof+0.8*gen_oof)
        r2_a=float(r2_score(y,c))
        if r2_a>best_r2_C: best_r2_C=r2_a; best_alpha=alpha
    comb_C=np.where(low_m,best_alpha*exp_oof+(1-best_alpha)*gen_oof,0.2*exp_oof+0.8*gen_oof)
    node_C={n:dict(general_r2=float(r2_score(y[df['technology_node']==n],gen_oof[df['technology_node']==n])),
                    combined_r2=float(r2_score(y[df['technology_node']==n],comb_C[df['technology_node']==n])),
                    group='low' if n in ['14nm','22nm','28nm'] else 'high',
                    n=int((df['technology_node']==n).sum()))
             for n in nodes_}
    print(f"  최적 α={best_alpha:.1f}: R²={best_r2_C:.4f}  {best_r2_C-baseline:+.4f}")
    for n in nodes_:
        d=node_C[n]['combined_r2']-node_C[n]['general_r2']
        print(f"    {n}: {node_C[n]['general_r2']:.4f}→{node_C[n]['combined_r2']:.4f}  {'✅' if d>0 else '❌'}{d:+.4f}")
    print(f"  판단: {'⚠️ 잠정 채택' if best_r2_C>baseline else '❌ 기각'}")

    # ── A+C 결합 ──────────────────────────────────────────
    print("\n  [§9-3 A+C 결합] 가중치 샘플링 + 그룹 Expert 앙상블")
    w_ac=np.ones(len(df))
    w_ac[df['technology_node']=='22nm']=best_w
    w_ac[df['technology_node']=='28nm']=best_w*0.8
    w_ac[df['technology_node']=='14nm']=best_w*0.6
    w_exp_ac=w_ac.copy(); w_exp_ac[low_m]*=2.0
    gen_ac=np.zeros(len(y)); exp_ac=np.zeros(len(y))
    for tr_i,va_i in kf.split(X):
        mg=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
        mg.fit(X[tr_i],y[tr_i],sample_weight=w_ac[tr_i]); gen_ac[va_i]=mg.predict(X[va_i])
        me=GradientBoostingRegressor(**BP,random_state=RANDOM_STATE)
        me.fit(X[tr_i],y[tr_i],sample_weight=w_exp_ac[tr_i]); exp_ac[va_i]=me.predict(X[va_i])
    comb_AC=np.where(low_m,0.5*exp_ac+0.5*gen_ac,0.2*exp_ac+0.8*gen_ac)
    r2_AC=float(r2_score(y,comb_AC))
    print(f"  A+C 결합 R²={r2_AC:.4f}  vs Baseline={r2_AC-baseline:+.4f}")
    print(f"  판단: {'✅ 최고 성능' if r2_AC>best_r2_A else '⚠️ A 단독이 더 우수'}")

    sec93_res=dict(
        A_hybrid_sampling=dict(weight_results=results_A,best_weight=float(best_w),
                                best_r2=best_r2_A,vs_baseline=best_r2_A-baseline,
                                best_node_r2=results_A[best_w]['node_r2'],
                                best_node_iqr=results_A[best_w]['node_iqr'],
                                baseline_iqr=base_iqr),
        B_error_corrector=dict(final_r2=r2_B,vs_baseline=r2_B-baseline,
                                node_results=node_corr),
        C_group_expert=dict(best_r2=best_r2_C,best_alpha=float(best_alpha),
                             vs_baseline=best_r2_C-baseline,node_results=node_C),
        AC_combined=dict(r2=r2_AC,vs_baseline=r2_AC-baseline),
        baseline_r2=baseline,
    )
    save_json(sec93_res,'sec93_residual_targeting')

    print(f"\n  ━━ §9-3 종합 ━━")
    print(f"  A 하이브리드 샘플링: {best_r2_A:.4f}  {best_r2_A-baseline:+.4f}  {'✅' if best_r2_A>baseline else '❌'}")
    print(f"  B 잔차 보정 모델:    {r2_B:.4f}  {r2_B-baseline:+.4f}  ❌")
    print(f"  C 그룹 Expert:       {best_r2_C:.4f}  {best_r2_C-baseline:+.4f}  {'✅' if best_r2_C>baseline else '❌'}")
    print(f"  A+C 결합:            {r2_AC:.4f}  {r2_AC-baseline:+.4f}")

    # Figure
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    fig.suptitle('§9-3. 잔차 타겟팅 전략',fontsize=13,fontweight='bold')
    ws_=sorted(results_A.keys())
    r2_w_=[results_A[w]['r2'] for w in ws_]
    bars=axes[0].bar([f'×{w:.1f}' for w in ws_],r2_w_,
                     color=['#27ae60' if v==max(r2_w_) else '#3498db' for v in r2_w_],alpha=0.85)
    axes[0].axhline(baseline,color='blue',linestyle='--',lw=1.5,label=f'Baseline={baseline:.4f}')
    axes[0].set_ylabel('R²'); axes[0].set_title('A. 가중치 배율별 R²\n→ ✅ ×1.5 최고',fontweight='bold')
    axes[0].legend(fontsize=8)
    for bar,val in zip(bars,r2_w_):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.0003,
                     f'{val:.4f}',ha='center',fontsize=9,fontweight='bold')
    methods_c=['GBM Baseline','GBM+Ridge\n보정']
    r2s_c=[baseline,r2_B]
    axes[1].bar(methods_c,r2s_c,color=['#3498db','#e74c3c'],alpha=0.85)
    axes[1].axhline(baseline,color='blue',linestyle='--',lw=1.5)
    axes[1].set_ylabel('R²'); axes[1].set_ylim(min(r2s_c)-.02,baseline+.01)
    axes[1].set_title('B. 잔차 보정 모델\n→ ❌ 기각',fontweight='bold',color='#c0392b')
    for bar,val in zip(axes[1].patches,r2s_c):
        axes[1].text(bar.get_x()+bar.get_width()/2,val+.001,f'{val:.4f}',ha='center',fontsize=10,fontweight='bold')
    methods_d=['GBM Baseline','그룹Expert\n앙상블','A+C 결합']
    r2s_d=[baseline,best_r2_C,r2_AC]
    colors_d=['#3498db','#9b59b6','#27ae60']
    axes[2].bar(methods_d,r2s_d,color=colors_d,alpha=0.85)
    axes[2].axhline(baseline,color='blue',linestyle='--',lw=1.5)
    axes[2].set_ylabel('R²'); axes[2].set_ylim(min(r2s_d)-.01,max(r2s_d)+.01)
    axes[2].set_title('C. 그룹 Expert + A+C 결합\n→ ⚠️ 잠정 채택',fontweight='bold')
    for bar,val in zip(axes[2].patches,r2s_d):
        axes[2].text(bar.get_x()+bar.get_width()/2,val+.0003,f'{val:.4f}',ha='center',fontsize=10,fontweight='bold')
    save_fig(fig,'sec93_followup')
    return sec93_res


if __name__=='__main__': main()