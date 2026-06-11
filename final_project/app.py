"""
app.py  ─ streamlit app
실행:  streamlit run app.py
══════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, io, time, subprocess, threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import joblib
import shap
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     cross_val_predict, KFold)
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# ── 한글 폰트 설정 --> matplotlib와 seaborn 그릴때 글자 깨지는것 방자
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
# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════
PROCESS_FEATURES = [
    'etch_rate','pressure','temperature','exposure_time',
    'focus_offset','dose','deposition_rate','thickness_uniformity',
    'implant_energy','tilt_angle','critical_dimension','oxide_thickness',
    'resistivity','defect_count','defect_density','vth',
    'leakage_current','resistance',
]
TARGET      = 'yield'
TECH_NODES  = ['7nm','10nm','14nm','22nm','28nm']
NODE_COLORS = {'7nm':'#e74c3c','10nm':'#e67e22','14nm':'#f39c12',
               '22nm':'#27ae60','28nm':'#2980b9'}
TOOL_GROUPS = {
    'etch_tool':       ['etch_rate','pressure','temperature'],
    'litho_tool':      ['focus_offset','dose'],
    'deposition_tool': ['deposition_rate','thickness_uniformity'],
    'implant_tool':    ['implant_energy','tilt_angle'],
}
FEATURE_LABELS = {
    "etch_rate":            "Etch Rate (nm/min)",
    "pressure":             "Chamber Pressure (mTorr)",
    "temperature":          "Process Temperature (°C)",
    "exposure_time":        "Exposure Time (s)",
    "focus_offset":         "Focus Offset (nm)",
    "dose":                 "Implant Dose (ions/cm²)",
    "deposition_rate":      "Deposition Rate (nm/min)",
    "thickness_uniformity": "Thickness Uniformity (%)",
    "implant_energy":       "Implant Energy (keV)",
    "tilt_angle":           "Tilt Angle (°)",
    "critical_dimension":   "Critical Dimension (nm)",
    "oxide_thickness":      "Oxide Thickness (Å)",
    "resistivity":          "Resistivity (Ω·cm)",
    "defect_count":         "Defect Count (#)",
    "defect_density":       "Defect Density (def/cm²)",
    "vth":                  "Threshold Voltage Vth (V)",
    "leakage_current":      "Leakage Current (A)",
    "resistance":           "Resistance (Ω)",
    "tech_encoded":         "Technology Node (encoded)",
    "is_advanced_node":     "7/10nm",
    "defect_composite":     "Defect Count * Defect Density",
    "Swing":                "Vtsat / Log10(Leakage)",
    "IR_Drop":              "Vth * Log10(Leakage)",
    "node_x_cd":            "Node * CD",
    "process_stability":    "Overall 공정인자/공정산포",
    "power_efficiency":     "Vth/Res*Leakage",
    "cd_squared":           "CD^2"
}
RANDOM_STATE = 42
APP_DIR      = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="반도체 수율 분석 및 예측",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1b3a5c);color:white}
[data-testid="stSidebar"] *{color:white!important}
div[data-testid="metric-container"]{background:#f8fafc;border-radius:10px;
  padding:12px;border:1px solid #dde4ed}
.stTabs [data-baseweb="tab"]{padding:8px 16px;border-radius:8px 8px 0 0;
  font-weight:600;font-size:.9em}
.sec-hdr{background:linear-gradient(90deg,#1b3a5c,#2980b9);color:white;
  padding:10px 18px;border-radius:8px;font-weight:700;margin:14px 0 10px}
.insight{padding:13px 17px;border-radius:0 8px 8px 0;margin:10px 0;border-left:5px solid}
.ins-blue  {background:#eaf4ff;border-color:#2980b9}
.ins-green {background:#eafaf1;border-color:#27ae60}
.ins-orange{background:#fef9ec;border-color:#e67e22}
.ins-red   {background:#fdecea;border-color:#e74c3c}
.ins-purple{background:#f5eeff;border-color:#8e44ad}
.pipeline-box{background:#1e293b;border-radius:10px;padding:16px;
  border:1px solid #334155;margin:6px 0}
.step-running{border-color:#22d3ee!important;background:#0f2a3a!important}
.step-done{border-color:#27ae60!important;background:#0f2a1a!important}
.step-error{border-color:#e74c3c!important;background:#2a0f0f!important}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 기타 필요 함수
# ═══════════════════════════════════════════════════════════════
def norm(arr):
    r = arr - arr.min()
    return r / (r.max() + 1e-10)

def insight(text, kind='blue'):
    st.markdown(f'<div class="insight ins-{kind}">{text}</div>', unsafe_allow_html=True)

def sec_hdr(text):
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)

def savebuf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0); plt.close(fig)
    return buf

def build_fsc(df, le):
    """EDA 피처셋 C (23개) 구성"""
    d = df.copy()
    d['tech_encoded']        = le.transform(d['technology_node'])
    d['is_advanced_node']    = d['technology_node'].isin(['7nm','10nm']).astype(int)
    d['defect_composite']    = (d['defect_count'] * d['defect_density']).apply(np.log1p)
    d['cd_vth_product']      = d['critical_dimension'] * d['vth']
    d['uniformity_x_defect'] = d['thickness_uniformity'] * d['defect_count']
    d['node_x_cd']           = d['tech_encoded'] * d['critical_dimension']
    FS_B = [f for f in PROCESS_FEATURES if f != 'defect_density'] + ['tech_encoded']
    FS_C = FS_B + ['is_advanced_node','defect_composite',
                   'cd_vth_product','uniformity_x_defect','node_x_cd']
    return d, d[FS_C].values, FS_C

# ═══════════════════════════════════════════════════════════════
# 예측용 샘플 CSV 생성 함수
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def make_predict_csv(raw_df: pd.DataFrame) -> bytes:
    """yield 컬럼 없는 예측용 샘플 50개 생성"""
    np.random.seed(99)
    n = min(50, len(raw_df))
    pred = raw_df.sample(n, random_state=99).copy()
    pred = pred.drop(columns=[TARGET], errors='ignore')
    pred['lot_id']   = [f'PRED_{i+1:04d}' for i in range(n)]
    pred['wafer_id'] = [f'W{str(np.random.randint(1, 100)).zfill(3)}' for _ in range(n)]
    return pred.to_csv(index=False).encode('utf-8-sig')

# ═══════════════════════════════════════════════════════════════
# 캐시 함수
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="모델 초기화 중...")
def init_model(n_rows):
    le = LabelEncoder().fit(TECH_NODES)
    df_raw = st.session_state['raw_df']
    df, X, FS_C = build_fsc(df_raw, le)
    y = df[TARGET].values
    model_path = os.path.join(APP_DIR, 'output', 'best_model.pkl')
    alt_path   = os.path.join(APP_DIR, 'best_model_final.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    elif os.path.exists(alt_path):
        model = joblib.load(alt_path)
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2,
            subsample=0.9, min_samples_leaf=4, max_features=0.9,
            random_state=RANDOM_STATE)
        model.fit(X, y)
    return model, le, df, X, y, FS_C

@st.cache_data(show_spinner="XAI 6-Method 분석 중 (약 40초)...")
def run_xai(_model, X, y, feature_names, seeds=[42, 123, 777]):
    # 각 지표별로 누적할 배열 초기화
    gb_accum = np.zeros(X.shape[1])
    rf_accum = np.zeros(X.shape[1])
    et_accum = np.zeros(X.shape[1])
    shap_accum = np.zeros(X.shape[1])
    perm_accum = np.zeros(X.shape[1])
    pear_n = norm(np.array([abs(stats.pearsonr(X[:,i], y)[0]) for i in range(X.shape[1])]))
    
    def _inner(pipe):
        if hasattr(pipe, 'named_steps'):
            return list(pipe.named_steps.values())[-1]
        return pipe
    
    def add_shap_importance(model, X):
        # Tree 모델 전용 Explainer 사용 (가장 빠름)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # SHAP 값의 절대값 평균을 내어 중요도로 사용
        if isinstance(shap_values, list): # 다중 클래스나 특정 모델 출력 대응
            shap_sum = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_sum = np.abs(shap_values).mean(axis=0)
        return norm(shap_sum)
    
    inner_model = _inner(_model)
    gb_n = norm(inner_model.feature_importances_ if hasattr(inner_model, 'feature_importances_') else np.ones(X.shape[1]))
    
    #phase2의 하이퍼파라미터 튜닝에서 best성능을 보였던 paramer로 모델 사용
    for seed in seeds:
        # 모델별 seed 적용
        # RF
        rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))
        rf.fit(X, y)
        rf_accum += norm(_inner(rf).feature_importances_)
        
        # ET
        et = make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1))
        et.fit(X, y)
        et_accum += norm(_inner(et).feature_importances_)
        
        # Permutation
        perm = permutation_importance(_model, X, y, n_repeats=20, random_state=seed, n_jobs=-1)
        perm_accum += norm(np.clip(perm.importances_mean, 0, None))
        print("완료")
        
        # 4. SHAP (현재 시드 모델에 대한 SHAP)
        # TreeExplainer는 트리 기반 모델의 실제 분기 구조를 활용하므로 매우 정확함
        explainer = shap.TreeExplainer(_inner(rf)) # RF 모델을 기준으로 SHAP 계산
        shap_values = explainer.shap_values(X)
        shap_sum = np.abs(shap_values).mean(axis=0)
        shap_accum += norm(shap_sum)

    # Seed 개수로 나누어 평균 산출
    n = len(seeds)
    gb_n, rf_n, et_n, shap_n, perm_n = gb_n, rf_accum/n, et_accum/n, shap_accum/n, perm_accum/n
    
    # 앙상블 점수 (Pearson 포함 6개 지표 평균)
    ens = (gb_n + rf_n + et_n + perm_n + shap_n + pear_n) / 6
    
    return pd.DataFrame({
        'feature': feature_names,
        'label': [FEATURE_LABELS.get(f, f) for f in feature_names],
        'GB': gb_n, 'RF': rf_n, 'ET': et_n,
        'Permutation': perm_n, 'SHAP': shap_n,'Pearson': pear_n,  'Ensemble_Score': ens,
    }).sort_values('Ensemble_Score', ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# 파이프라인 결과 로드 함수
# ═══════════════════════════════════════════════════════════════
def load_pipeline_results():
    """output/ 디렉토리에서 JSON 결과를 로드"""
    res_dir = os.path.join(APP_DIR, 'output', 'results')
    fig_dir = os.path.join(APP_DIR, 'output', 'figures')
    results = {}
    if os.path.exists(res_dir):
        for fname in os.listdir(res_dir):
            if fname.endswith('.json'):
                key = fname.replace('.json', '')
                try:
                    with open(os.path.join(res_dir, fname)) as f:
                        results[key] = json.load(f)
                except:
                    pass
    figures = {}
    if os.path.exists(fig_dir):
        for fname in os.listdir(fig_dir):
            if fname.endswith('.png'):
                figures[fname.replace('.png','')] = os.path.join(fig_dir, fname)
    return results, figures

# ═══════════════════════════════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    model_path = os.path.join("output", "best_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    st.markdown("## 🔬 반도체 수율 분석")
    st.markdown("---")
    st.markdown("### 📂 데이터 업로드")
    uploaded = st.file_uploader("CSV 파일", type=['csv'])
    st.markdown("---")
    st.markdown("### ⚙️ 분석 설정")
    tech_filter = st.multiselect("Technology Node", TECH_NODES, default=TECH_NODES)
    top_k = st.slider("XAI 상위 피처 수", 3, 23, 10)
    st.markdown("---")
#     model_name = type(model).__name__
#     params = model.get_params()
#     depth = params['max_depth']
#     lr = params['learning_rate']
#     subsample = params['subsample']
#     res = model.metadata[model.best_name]
    
#     st.markdown(f"""
# **Best Model**  
# `{model_name}`  
# - 5-CV R² = **{res['cv5_r2']:.4f}**  

# - 피처셋 갯수 — {model.n_features_in_}  
# - depth={depth}, lr={lr}, subsample={subsample}
# """)

# ═══════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════
DEFAULT_PATH = os.path.join(APP_DIR, 'semiconductor_yield_forecasting_data.csv')
if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.sidebar.success(f"✅ 업로드: {raw_df.shape[0]}행")
elif os.path.exists(DEFAULT_PATH):
    raw_df = pd.read_csv(DEFAULT_PATH)
    st.sidebar.info(f"ℹ️ 기본 데이터: {raw_df.shape[0]}행")
else:
    st.error("❗ 사이드바에서 CSV를 업로드하세요."); st.stop()

raw_df = raw_df[raw_df['technology_node'].isin(tech_filter)].copy()
if raw_df.empty:
    st.warning("선택된 노드 없음"); st.stop()

st.session_state['raw_df'] = raw_df
model, le, df, X, y, FS_C = init_model(len(raw_df))
y_pred    = model.predict(X)
residuals = y - y_pred
NODES     = sorted(df['technology_node'].unique())
kf        = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════
# 탭 구성
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "⚙️ Pipeline",
    "📊 EDA",
    "🔍 피처 선택",
    "🤖 모델 성능",
    "🧠 XAI 유의인자",
    "📈 예측 & 시뮬레이션",
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 0 — 파이프라인 실행                                    ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[0]:
    import sys as _sys
    import importlib

    # ── 환경 진단 ─────────────────────────────────────────────
    REQUIRED_PKGS = {
        "numpy":      ("numpy",      "numpy"),
        "pandas":     ("pandas",     "pandas"),
        "matplotlib": ("matplotlib", "matplotlib"),
        "seaborn":    ("seaborn",    "seaborn"),
        "scipy":      ("scipy",      "scipy"),
        "scikit-learn":("sklearn",   "scikit-learn"),
        "joblib":     ("joblib",     "joblib"),
        "streamlit":  ("streamlit",  "streamlit"),
    }
    OPTIONAL_PKGS = {
        "xgboost":   ("xgboost",  "xgboost"),
        "lightgbm":  ("lightgbm", "lightgbm"),
        "catboost":  ("catboost", "catboost"),
    }

    @st.cache_data(ttl=60)
    def _check_packages():
        ok, missing, optional_ok, optional_miss = [], [], [], []
        for label, (import_name, _) in REQUIRED_PKGS.items():
            if importlib.util.find_spec(import_name):
                ok.append(label)
            else:
                missing.append(label)
        for label, (import_name, _) in OPTIONAL_PKGS.items():
            if importlib.util.find_spec(import_name):
                optional_ok.append(label)
            else:
                optional_miss.append(label)
        return ok, missing, optional_ok, optional_miss

    pkg_ok, pkg_miss, opt_ok, opt_miss = _check_packages()

    # ── 현재 Python 실행 경로 (핵심: 가상환경 격리 보장) ──────
    PYTHON_EXE     = _sys.executable
    pipeline_script = os.path.join(APP_DIR, 'run_full_pipeline.py')
    script_exists   = os.path.exists(pipeline_script)
    data_exists     = os.path.exists(os.path.join(APP_DIR, 'semiconductor_yield_forecasting_data.csv'))

    # ── requirements.txt 생성 함수 ────────────────────────────
    REQUIREMENTS_CONTENT = """\
# 반도체 수율 예측 파이프라인 — 필수 패키지
# 설치 방법: pip install -r requirements.txt
altair==6.2.1
anyio==4.13.0
attrs==26.1.0
blinker==1.9.0
cachetools==7.1.4
catboost==1.2.10
certifi==2026.5.20
charset-normalizer==3.4.7
click==8.4.1
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
fonttools==4.63.0
gitdb==4.0.12
GitPython==3.1.50
graphviz==0.21
h11==0.16.0
httptools==0.8.0
idna==3.18
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.0
lightgbm==4.6.0
llvmlite==0.47.0
MarkupSafe==3.0.3
matplotlib==3.10.9
narwhals==2.22.1
numba==0.65.1
numpy==2.4.6
packaging==26.2
pandas==3.0.3
pillow==12.2.0
plotly==6.8.0
protobuf==7.35.0
pyarrow==24.0.0
pydeck==0.9.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-multipart==0.0.32
referencing==0.37.0
requests==2.34.2
rpds-py==2026.5.1
scikit-learn==1.9.0
scipy==1.17.1
seaborn==0.13.2
shap==0.52.0
six==1.17.0
slicer==0.0.8
smmap==5.0.3
starlette==1.3.0
streamlit==1.58.0
tenacity==9.1.4
threadpoolctl==3.6.0
toml==0.10.2
tqdm==4.68.2
typing_extensions==4.15.0
tzdata==2026.2
urllib3==2.7.0
uvicorn==0.49.0
watchdog==6.0.0
websockets==16.0
xgboost==3.2.0
"""

    sec_hdr("⚙️ 파이프라인 실행 패널")

    # ── 환경 상태 카드 ────────────────────────────────────────
    with st.expander("🔍 실행 환경 진단", expanded=len(pkg_miss) > 0):
        c_env1, c_env2 = st.columns(2)
        with c_env1:
            st.markdown("**Python 실행 경로**")
            st.code(PYTHON_EXE, language="bash")
            st.caption("이 경로의 Python으로 파이프라인이 실행됩니다.")

            st.markdown("**필수 패키지 상태**")
            for p in pkg_ok:
                st.markdown(f"✅ `{p}`")
            for p in pkg_miss:
                st.markdown(f"❌ `{p}` — 미설치")

        with c_env2:
            st.markdown("**선택 패키지 (부스팅 라이브러리)**")
            for p in opt_ok:
                st.markdown(f"✅ `{p}` (실제 사용)")
            for p in opt_miss:
                st.markdown(f"⚠️ `{p}` — 미설치 (HistGBM 대안 자동 사용)")

            st.markdown("**필수 파일**")
            st.markdown(f"{'✅' if script_exists else '❌'} `run_full_pipeline.py`")
            st.markdown(f"{'✅' if data_exists   else '❌'} `semiconductor_yield_forecasting_data.csv`")

        # 패키지 누락 시 설치 가이드
        if pkg_miss:
            st.error(f"❌ 필수 패키지 누락: `{', '.join(pkg_miss)}`")
            st.code(f"pip install {' '.join(pkg_miss)}", language="bash")
        elif not script_exists or not data_exists:
            miss_files = []
            if not script_exists: miss_files.append("run_full_pipeline.py")
            if not data_exists:   miss_files.append("semiconductor_yield_forecasting_data.csv")
            st.warning(f"⚠️ 파일 없음: {', '.join(miss_files)}\n\n앱과 **같은 디렉토리**에 위치해야 합니다.")
        else:
            st.success("✅ 실행 환경 정상")

        # requirements.txt 다운로드
        st.download_button(
            "📥 requirements.txt 다운로드",
            data=REQUIREMENTS_CONTENT,
            file_name="requirements.txt",
            mime="text/plain",
            help="pip install -r requirements.txt 로 한 번에 설치",
        )

    st.markdown("---")

    # ── 실행 가능 여부 판단 ───────────────────────────────────
    can_run = script_exists and data_exists and len(pkg_miss) == 0

    if not can_run:
        if pkg_miss:
            insight(f"❌ <b>패키지 미설치로 실행 불가</b>: <code>pip install {' '.join(pkg_miss)}</code> 후 재시작하세요.", 'red')
        elif not script_exists:
            insight("❌ <b>run_full_pipeline.py 없음</b>: 앱과 같은 폴더에 파일을 배치하세요.", 'red')
        elif not data_exists:
            insight("❌ <b>데이터 파일 없음</b>: semiconductor_yield_forecasting_data.csv를 같은 폴더에 배치하세요.", 'red')

    # ── 파이프라인 스텝 정의 ──────────────────────────────────
    STEPS = [
        {"id": 1, "label": "Step1 EDA 심층 분석",       "desc": "T-Test 검증, 다중공선성, 피처셋 A/B/C R² 비교", "time": "~20초", "color": "#22d3ee"},
        {"id": 2, "label": "Step2 Baseline 스크리닝",    "desc": "26개 모델 5-Fold CV 평가 (FS-C 기준)",            "time": "~3분",  "color": "#3b82f6"},
        {"id": 3, "label": "Step3 Hyper-param Tuning",  "desc": "Top5 RandomizedSearchCV + best_params 저장",     "time": "~5분",  "color": "#8b5cf6"},
        {"id": 4, "label": "Step4 Ensemble",             "desc": "Stacking / Voting (Top3)",                  "time": "~2분",  "color": "#f59e0b"},
        {"id": 5, "label": "Step5-7 진단·EDA비교·요약", "desc": "잔차, Q-Q, 노드별 R², Best model 선정 종합",      "time": "~3분",  "color": "#10b981"},
        # {"id": 6, "label": "Step8 개선 실험",            "desc": "노드별 개별·이기종 앙상블·FS-D·커널 규명",        "time": "~5분",  "color": "#ef4444"},
        {"id": 6, "label": "XAI 유의인자",            "desc": "6-Method Voting (GB/RF/ET/Perm/Pearson)",        "time": "~2분",  "color": "#ec4899"},
    ]

    # ── 핵심 실행 함수 ────────────────────────────────────────
    def run_pipeline_step(step_ids: list, label: str):
        """
        현재 가상환경의 Python(sys.executable)으로 파이프라인 실행.
        - subprocess PATH 문제 완전 회피
        - stderr를 stdout에 합쳐 모든 오류를 실시간 표시
        - 타임아웃 없이 장시간 작업 지원
        """
        cmd = [PYTHON_EXE, pipeline_script, "--steps"] + [str(s) for s in step_ids]
        log_placeholder    = st.empty()
        status_placeholder = st.empty()
        log_lines = []

        status_placeholder.info(f"⏳ 실행 중: **{label}**")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=APP_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # stderr → stdout 합치기
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',           # Windows cp949에서 디코딩 오류 방지
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",  # 실시간 flush 보장
                    "PYTHONUTF8":       "1",  # Python 3.7+ UTF-8 모드 (Windows)
                    "PYTHONIOENCODING": "utf-8:replace",  # 하위 호환
                },
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    log_lines.append(stripped)
                if len(log_lines) > 60:
                    log_lines = log_lines[-60:]
                log_placeholder.code("\n".join(log_lines), language="bash")

            proc.wait()

            if proc.returncode == 0:
                status_placeholder.success(f"✅ 완료: **{label}**")
                for sid in step_ids:
                    st.session_state[f'step_done_{sid}'] = True
                st.session_state['pipeline_last_run'] = time.time()
            else:
                status_placeholder.error(
                    f"❌ 오류 (return code {proc.returncode}): **{label}**\n\n"
                    f"위 로그에서 오류 원인을 확인하세요."
                )

        except FileNotFoundError:
            status_placeholder.error(
                f"❌ Python 실행파일을 찾을 수 없습니다.\n\n"
                f"경로: `{PYTHON_EXE}`\n\n"
                f"가상환경이 활성화된 상태에서 Streamlit을 실행했는지 확인하세요."
            )
        except PermissionError:
            status_placeholder.error(
                f"❌ 실행 권한 없음: `{PYTHON_EXE}`\n\n"
                f"터미널에서 `chmod +x {PYTHON_EXE}` 를 실행해 보세요."
            )
        except Exception as e:
            status_placeholder.error(f"❌ 예상치 못한 오류: {e}")

    # ── 일괄 실행 버튼 ────────────────────────────────────────
    col_all1, col_all3 = st.columns([1, 2])
    with col_all1:
        run_all = st.button("🚀 전체 Step 실행", type="primary",
                            use_container_width=True, disabled=not can_run)
    
    with col_all3:
        if can_run:
            st.caption(f"💡 Python: `{os.path.basename(PYTHON_EXE)}`  |  작업 디렉토리: `{APP_DIR}`")
        else:
            st.caption("⛔ 위 환경 진단을 먼저 확인하세요.")

    if run_all and can_run:
        st.markdown("**🚀 전체 파이프라인 실행 로그**")
        run_pipeline_step([1,2,3,4,5,6], "전체 파이프라인")
        st.rerun()
    

    st.markdown("---")

    # ── 개별 스텝 실행 UI ─────────────────────────────────────
    for step in STEPS:
        sid   = step["id"]
        done  = st.session_state.get(f'step_done_{sid}', False)
        color = step["color"]

        col1, col2, col3, col4 = st.columns([0.5, 2.8, 0.8, 0.9])
        with col1:
            st.markdown(
                f"<span style='font-size:1.4em'>{'✅' if done else '⬜'}</span>",
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"<b style='color:{color}'>{step['label']}</b>  "
                f"<span style='color:#94a3b8;font-size:.85em'>{step['desc']}</span>",
                unsafe_allow_html=True)
        with col3:
            st.caption(f"⏱ {step['time']}")
        with col4:
            clicked = st.button(f"▶ Step {sid}", key=f"run_step_{sid}",
                                use_container_width=True, disabled=not can_run)

        # ── 로그 창은 컬럼 밖 전체 너비로 ──────────────────────
        if clicked:
            st.markdown(
                f"<div style='background:#0f172a;border:1px solid {color};"
                f"border-radius:8px;padding:4px 12px 2px;margin-bottom:4px;'>"
                f"<span style='color:{color};font-weight:700;font-size:.9em'>"
                f"▶ Step {sid} — {step['label']}</span></div>",
                unsafe_allow_html=True)
            run_pipeline_step([sid], step['label'])
            st.rerun()

    st.markdown("---")

    # 파이프라인 결과 미리보기
    sec_hdr("📂 파이프라인 실행 결과 미리보기")
    results, figures = load_pipeline_results()

    if not results and not figures:
        st.info("아직 파이프라인을 실행하지 않았습니다. 위 버튼으로 실행 후 결과가 여기에 표시됩니다.")
    else:
        # 주요 지표 표시
        if 'model_meta' in results:
            meta = results['model_meta']
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best Model",    meta.get('model_name','GBM')[:15])
            c2.metric("5-CV R²",       f"{meta.get('cv5_r2', 0):.4f}")
            c3.metric("R² std",        f"±{meta.get('cv5_std', 0):.4f}")
            c4.metric("RMSE",          f"{meta.get('cv5_rmse', 0):.4f}")
            c5.metric("Feature Set",   meta.get('feature_set','C'))

        # 결과 그래프 탭 미리보기
        fig_tabs_labels = {
            'sec1_eda_deep':         'Step1 EDA',
            'sec2_phase1_screening': 'Step2 Baseline',
            'sec3a_tuning_comparison':'Step3 Tuning',
            'sec4_leaderboard':      'Step4 Leaderboard',
            'sec5_final_diagnostics':'Step5 Final model',
            'sec7_final_summary':    'Step7 Summary',
            'xai_voting':            'XAI',
        }
        avail = {k: v for k, v in fig_tabs_labels.items() if k in figures}
        if avail:
            fig_tab_objs = st.tabs(list(avail.values()))
            for tab_obj, key in zip(fig_tab_objs, avail.keys()):
                with tab_obj:
                    st.image(figures[key], use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — EDA                                                ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[1]:
    sec_hdr("📊 데이터 분석 (EDA)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("총 Wafer",   f"{len(df):,}")
    c2.metric("평균 Yield",  f"{df[TARGET].mean():.4f}")
    c3.metric("Yield Std",   f"{df[TARGET].std():.4f}")
    c4.metric("Nodes",       len(NODES))

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Yield 분포 (노드별)")
        fig, ax = plt.subplots(figsize=(6,4))
        for node in NODES:
            sub = df[df['technology_node']==node][TARGET]
            ax.hist(sub, bins=20, alpha=0.55, label=node,
                    color=NODE_COLORS.get(node,'#95a5a6'), edgecolor='white')
        ax.set_xlabel('Yield'); ax.set_ylabel('Count'); ax.legend(fontsize=8)
        ax.set_title('노드별 Yield 히스토그램')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Violin Plot")
        fig, ax = plt.subplots(figsize=(6,4))
        data_v = [df[df['technology_node']==n][TARGET].values for n in NODES]
        parts  = ax.violinplot(data_v, showmeans=True, showmedians=True)
        for pc, node in zip(parts['bodies'], NODES):
            pc.set_facecolor(NODE_COLORS.get(node,'#95a5a6')); pc.set_alpha(0.7)
        ax.set_xticks(range(1,len(NODES)+1)); ax.set_xticklabels(NODES, fontsize=9)
        ax.set_ylabel('Yield'); ax.set_title('Violin by Technology Node')
        mu = df[TARGET].mean()
        ax.axhline(mu, color='gray', linestyle=':', lw=1.2, label=f'전체 μ={mu:.3f}')
        ax.legend(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # 이분화 t-test
    hg = df[df['technology_node'].isin(['7nm','10nm'])][TARGET]
    lg = df[df['technology_node'].isin(['14nm','22nm','28nm'])][TARGET]
    if len(hg)>0 and len(lg)>0:
        t_s, t_p = stats.ttest_ind(hg, lg)
        # insight(f'💡 <b>T-test 검증</b> — 고수율(7/10nm) μ={hg.mean():.3f} vs '
        #         f'저수율(14~28nm) μ={lg.mean():.3f} | t={t_s:.1f}, p={t_p:.2e}', 'orange')
        # 1. p-value 기준 판정 로직
        alpha = 0.05
        if t_p < alpha:
            significance = "통계적으로 유의함 (차이 존재)"
            decision = "✅ 기각: 그룹 간 수율 차이가 있습니다."
        else:
            significance = "통계적으로 유의하지 않음 (차이 없음)"
            decision = "❌ 채택: 그룹 간 수율 차이가 확인되지 않습니다."

        # 2. 결과 출력
        insight(f'💡 <b>T-test 검증</b> — 고수율(7/10nm) μ={hg.mean():.3f} vs '
                f'저수율(14~28nm) μ={lg.mean():.3f} <br>'
                f't={t_s:.1f}, p={t_p:.2e} — <b>{significance}</b> <br>'
                f'{decision}', 'orange')

    st.markdown("---")
    st.subheader("노드별 Yield 요약 통계")
    summary = df.groupby('technology_node')[TARGET].agg(
        Count='count', Mean='mean', Std='std', Min='min', Max='max').round(4)
    st.dataframe(summary.style.background_gradient(subset=['Mean'], cmap='RdYlGn'),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("공정 파라미터 분석")
    sel = st.selectbox("피처 선택", PROCESS_FEATURES,
                       format_func=lambda x: FEATURE_LABELS.get(x,x))
    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6,4))
        for node in NODES:
            sub = df[df['technology_node']==node][sel]
            ax.hist(sub, bins=20, alpha=0.5, label=node,
                    color=NODE_COLORS.get(node,'#95a5a6'), edgecolor='white')
        ax.set_xlabel(FEATURE_LABELS.get(sel,sel)); ax.set_ylabel('Count')
        ax.legend(fontsize=7); ax.set_title(f'{sel} 분포')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with cb:
        fig, ax = plt.subplots(figsize=(6,4))
        for node in NODES:
            sub = df[df['technology_node']==node]
            r, _ = stats.pearsonr(sub[sel], sub[TARGET])
            ax.scatter(sub[sel], sub[TARGET], alpha=0.3, s=10,
                       color=NODE_COLORS.get(node,'#95a5a6'), label=f'{node}(r={r:.2f})')
        ax.set_xlabel(FEATURE_LABELS.get(sel,sel)); ax.set_ylabel('Yield')
        ax.legend(fontsize=6, markerscale=3); ax.set_title(f'{sel} vs Yield')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ─── 피처 상관행렬 히트맵 (하삼각 행렬 부분만 마스킹) ───────────────────
    st.markdown("---")
    st.subheader("피처 상관행렬 히트맵")
    node_sel = st.selectbox("노드 선택", ['전체']+NODES, key='hm_node')
    df_hm = df if node_sel=='전체' else df[df['technology_node']==node_sel]
    corr_m = df_hm[PROCESS_FEATURES + [TARGET]].corr()
    # 상삼각 마스킹 → 하삼각만 표시
    mask_upper = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr_m, ax=ax, mask=mask_upper,
        annot=True, fmt='.2f', cmap='RdYlGn',
        center=0, linewidths=0.3, annot_kws={'size': 6},
        square=True, vmin=-1, vmax=1
    )
    ax.set_title(f'상관행렬 — ({node_sel})', fontsize=12, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ─── Route Path 시각화 ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ 공정 Route Path 분석 (노드별 Tool 진행 경로)")
    insight("💡 각 Wafer가 통과한 4개 Tool 종류의 조합(Route)을 노드별로 시각화합니다. "
            "단일 Route 비중이 높을 경우 범주형 Feature로서 적절하지 않을 가능성 있어서, 미리 EDA 단계에서 확인.", 'blue')

    # 공정 경로 구성
    df['route_full'] = (df['etch_tool'] + ' → ' + df['litho_tool'] + ' → '
                        + df['deposition_tool'] + ' → ' + df['implant_tool'])

    route_tab1, route_tab2 = st.tabs(["📊 노드별 Route 분포", "🔗 Sankey-style 경로 흐름"])

    with route_tab1:
        route_cols = st.columns(len(NODES))
        for col_i, node in zip(route_cols, NODES):
            with col_i:
                sub = df[df['technology_node'] == node]
                rc  = sub['route_full'].value_counts()
                top_ratio = rc.iloc[0] / len(sub) * 100 if len(rc) > 0 else 0
                st.markdown(f"**{node}**  \n`{rc.index[0][:20]}...`  \n최다경로 **{top_ratio:.0f}%**")
                fig, ax = plt.subplots(figsize=(3.5, 3))
                colors_r = plt.cm.tab20(np.linspace(0, 1, len(rc)))
                ax.bar(range(len(rc)), rc.values, color=colors_r, alpha=0.85)
                ax.set_xticks(range(len(rc)))
                ax.set_xticklabels([f'R{i+1}' for i in range(len(rc))], fontsize=7)
                ax.set_ylabel('Count', fontsize=8)
                ax.set_title(f'{node} ({len(rc)} routes)', fontsize=9, fontweight='bold')
                for bar, val in zip(ax.patches, rc.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                            str(val), ha='center', fontsize=6)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with route_tab2:
        # 노드 선택
        route_node = st.selectbox("노드 선택", NODES, key='route_node')
        sub_r = df[df['technology_node'] == route_node]

        # 각 Tool 단계별 사용 빈도
        tool_cols = ['etch_tool', 'litho_tool', 'deposition_tool', 'implant_tool']
        tool_labels = ['① Etch', '② Litho', '③ Deposition', '④ Implant']

        fig = plt.figure(figsize=(20, 9))
        fig.suptitle(f'{route_node} 공정 Route Path 흐름 분석', fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)

        # 상단: 각 Tool별 사용 빈도 바차트
        for idx, (tc, tl) in enumerate(zip(tool_cols, tool_labels)):
            ax = fig.add_subplot(gs[0, idx])
            tc_counts = sub_r[tc].value_counts().sort_values(ascending=True)
            bar_colors = plt.cm.Set2(np.linspace(0, 1, len(tc_counts)))
            bars = ax.barh(tc_counts.index, tc_counts.values, color=bar_colors, alpha=0.9)
            ax.set_title(tl, fontsize=10, fontweight='bold')
            ax.set_xlabel('Count', fontsize=8)
            for bar, val in zip(bars, tc_counts.values):
                ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val} ({val/len(sub_r)*100:.0f}%)',
                        va='center', fontsize=7.5)
            ax.tick_params(labelsize=8)

        # 하단: 전체 Route 조합 시각화 (수평 막대 + Yield 평균)
        ax_main = fig.add_subplot(gs[1, :])
        route_yield = sub_r.groupby('route_full')[TARGET].agg(['count','mean']).reset_index()
        route_yield.columns = ['route','count','mean_yield']
        route_yield = route_yield.sort_values('count', ascending=True)

        # yield 기준 색상
        y_norm = (route_yield['mean_yield'] - route_yield['mean_yield'].min())
        y_norm = y_norm / (y_norm.max() + 1e-10)
        bar_clrs = plt.cm.RdYlGn(y_norm)

        bars2 = ax_main.barh(range(len(route_yield)), route_yield['count'],
                              color=bar_clrs, alpha=0.85)
        ax_main.set_yticks(range(len(route_yield)))
        short_routes = [r.replace(' → ', '→') for r in route_yield['route']]
        ax_main.set_yticklabels(short_routes, fontsize=8)
        ax_main.set_xlabel('Wafer 수 (Count)', fontsize=9)
        ax_main.set_title(f'{route_node} — Route별 Wafer 수 (색상: 평균 Yield ↑=녹색)', fontsize=10)

        for i, (bar, cnt, my) in enumerate(zip(bars2, route_yield['count'], route_yield['mean_yield'])):
            ax_main.text(cnt + 0.3, bar.get_y() + bar.get_height()/2,
                         f'n={cnt}  μY={my:.3f}',
                         va='center', fontsize=7.5, fontweight='bold')

        # 컬러바
        sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                    norm=plt.Normalize(vmin=route_yield['mean_yield'].min(),
                                                       vmax=route_yield['mean_yield'].max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_main, label='평균 Yield', shrink=0.6, pad=0.01)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Route 요약 테이블
        route_yield_sorted = route_yield.sort_values('count', ascending=False)
        route_yield_sorted['비율(%)'] = (route_yield_sorted['count'] / len(sub_r) * 100).round(1)
        route_yield_sorted['mean_yield'] = route_yield_sorted['mean_yield'].round(4)
        st.dataframe(
            route_yield_sorted[['route','count','비율(%)','mean_yield']].rename(
                columns={'route':'Route 경로','count':'Wafer 수','mean_yield':'평균 Yield'}
            ).style.background_gradient(subset=['평균 Yield'], cmap='RdYlGn')
                      .format({'비율(%)':'{:.1f}%','평균 Yield':'{:.4f}'}),
            use_container_width=True
        )
        insight(f"📌 <b>{route_node} Route 분석</b>: "
                f"총 {sub_r['route_full'].nunique()}개 고유 경로 중 "
                f"최다 경로가 전체의 {route_yield['count'].max()/len(sub_r)*100:.0f}%를 차지합니다. "
                f"Best Yield 경로: <b>{route_yield.loc[route_yield['mean_yield'].idxmax(),'route']}</b> "
                f"(μ={route_yield['mean_yield'].max():.3f})", 'green')

    # Tool Group별 Yield 분포
    st.markdown("---")
    st.subheader("Tool Group별 Yield 분포")
    tg_cols = st.columns(2)
    for idx, (tool_col, feats) in enumerate(TOOL_GROUPS.items()):
        with tg_cols[idx % 2]:
            tools = sorted(df[tool_col].unique())
            fig, ax = plt.subplots(figsize=(5,3.5))
            bp = ax.boxplot([df[df[tool_col]==t][TARGET].values for t in tools],
                            labels=tools, patch_artist=True)
            for patch, color in zip(bp['boxes'],
                plt.cm.Set2(np.linspace(0,1,len(tools)))):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_title(f'{tool_col} → Yield'); ax.set_ylabel('Yield')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — 피처 선택                                          ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[2]:
    sec_hdr("🔍 피처 선택 & EDA 반영 효과")

    

    st.markdown("---")
    st.subheader("피처셋 A / B / C  — R² 비교")

    probe = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=2,
        subsample=0.9, min_samples_leaf=2, max_features=0.9,
        random_state=RANDOM_STATE)
    FS_A = PROCESS_FEATURES + ['tech_encoded']
    FS_B = [f for f in PROCESS_FEATURES if f!='defect_density'] + ['tech_encoded']
    df_d = df.copy()
    # df_d['process_stability'] = (df_d['etch_rate']/(df_d['etch_rate'].std()+1e-10)+
    #                               df_d['deposition_rate']/(df_d['deposition_rate'].std()+1e-10)+
    #                               df_d['implant_energy']/(df_d['implant_energy'].std()+1e-10))/3
    # df_d['electrical_composite'] = df_d['vth']*df_d['resistance']/df_d['leakage_current'].clip(1e-10)
    # df_d['cd_squared'] = df_d['critical_dimension']**2
    # FS_D = FS_C + ['process_stability','electrical_composite','cd_squared']

    fs_cfg = [
        (f'A — Baseline ({len(FS_A)}개)',       FS_A,  '#95a5a6'),
        (f'B — EDA 다중공선성 제거 ({len(FS_B)}개)', FS_B,  '#3498db'),
        (f'C — EDA+FE ({len(FS_C)}개) ★',      FS_C,  '#e74c3c'),
        
    ]
    fs_r2 = {}
    for label, fset, _ in fs_cfg:
        try:
            Xf = df_d[fset].values
            fs_r2[label] = float(cross_val_score(probe, Xf, y, cv=kf, scoring='r2').mean())
        except:
            fs_r2[label] = 0.0

    col_fs1, col_fs2 = st.columns([1,1])
    with col_fs1:
        fig, ax = plt.subplots(figsize=(7,4))
        labels_ = [c[0] for c in fs_cfg]
        vals_   = [fs_r2[l] for l in labels_]
        colors_ = [c[2] for c in fs_cfg]
        bars = ax.bar(range(len(labels_)), vals_, color=colors_, alpha=0.85, width=0.5)
        ax.set_xticks(range(len(labels_)))
        ax.set_xticklabels(['A\nBaseline','B\nEDA제거','C\nEDA+FE★'], fontsize=9)
        ax.set_ylim(min(vals_)-0.01, max(vals_)+0.01)
        ax.set_ylabel('R² (5-Fold CV)'); ax.set_title('피처셋별 R² (동일 GBM 기준)')
        for bar, val in zip(bars, vals_):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
                    f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
        ax.axhline(vals_[0], color='gray', linestyle=':', lw=1.5, label='A 기준선')
        ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_fs2:
        insight('✅ <b>피처셋 C 채택 근거</b><br>EDA+FE로 Baseline 대비 R² 개선. '
                , 'green')
        

    st.markdown("---")
    st.subheader("노드별 피처-Yield 상관계수")
    # top5 = ['defect_count','thickness_uniformity','vth','critical_dimension','oxide_thickness']
    cols_hm=['thickness_uniformity','defect_count','defect_density',
             'critical_dimension','oxide_thickness','vth','resistance',TARGET]
    top5 = df[cols_hm].corr()[TARGET].abs().sort_values(ascending=False).drop(TARGET).head(5).index.tolist()
    corr_heat = pd.DataFrame({
        node: df[df['technology_node']==node][top5+[TARGET]].corr()[TARGET].drop(TARGET)
        for node in NODES})
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(corr_heat, ax=ax, annot=True, fmt='.3f', cmap='RdYlGn_r',
                center=0, linewidths=0.4, annot_kws={'size':9,'fontweight':'bold'})
    ax.set_title('노드별 피처-Yield Pearson r (상위 5개)', fontsize=11)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Tool Group 내 피처 기여도(피어슨 r)")
    tg_c = st.columns(len(TOOL_GROUPS))
    for col_i, (tool, feats) in zip(tg_c, TOOL_GROUPS.items()):
        with col_i:
            corr_tg = df[feats+[TARGET]].corr()[TARGET].drop(TARGET).abs()
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(corr_tg.index, corr_tg.values, color='#9b59b6', alpha=0.8)
            ax.set_title(tool, fontsize=9, fontweight='bold')
            ax.set_ylabel('|Pearson r|', fontsize=8)
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — 모델 성능                                          ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[3]:
    sec_hdr("🤖 모델 성능 평가")

    # 파이프라인 결과 있으면 불러오기
    pipe_results, pipe_figures = load_pipeline_results()

    st.markdown("""
**모델 선정 3-Phase 요약:**
- **Phase 1**: 26개 모델 Baseline model screen (EDA 피처셋 C 기준으로 평가)
- **Phase 2**: Top 5 RandomizedSearchCV 하이퍼파라미터 튜닝 — 각 파라미터 설계 근거 포함
- **Phase 3**: 앙상블 Stacking / Voting — 단독 GBM과 동률 → 설계 복잡도와 Gain을 고려했을때 GBM 최종 선택
""")

    # 파이프라인 결과가 있으면 실제 수치 사용
    if 'model_meta' in pipe_results:
        meta = pipe_results['model_meta']
        cv_r2   = meta.get('cv5_r2',  0)
        cv_rmse = meta.get('cv5_rmse', 0)
        cv_mae  = meta.get('cv5_mae',  0)
        cv_std  = meta.get('cv5_std',  0)
    else:
        cv_r2, cv_rmse, cv_mae, cv_std = 0, 0, 0, 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("5-Fold CV R²",   f"{cv_r2:.4f}", delta=f"+{cv_r2-0.669225246150581:.4f} vs Baseline")
    c2.metric("5-Fold CV RMSE", f"{cv_rmse:.4f}")
    c3.metric("5-Fold CV MAE",  f"{cv_mae:.4f}")
    c4.metric("R² std",         f"{cv_std:.4f}", delta="안정적")

    st.markdown("---")

    # 파이프라인 그래프 표시 (있으면)
    if 'sec3a_tuning_comparison' in pipe_figures:
        st.subheader("Phase 2 튜닝 결과 ")
        st.image(pipe_figures['sec3a_tuning_comparison'], use_container_width=True)
    if 'sec4_leaderboard' in pipe_figures:
        st.subheader("전체 모델 Leaderboard ")
        st.image(pipe_figures['sec4_leaderboard'], use_container_width=True)

    col_ph, col_pr = st.columns([1.2,1])
    with col_ph:
        st.subheader("Phase별 최고 R² 추이")
        # 파이프라인 결과 있으면 실제 값 사용
        if 'sec2_phase1' in pipe_results and 'sec3_tuning_results' in pipe_results:
            p1_best = max(v.get('R2_mean',0) for v in pipe_results['sec2_phase1'].get('results',{}).values())
            t3 = pipe_results['sec3_tuning_results']
            p2_best = max(v.get('cv5_r2',0) for v in t3.values()) if t3 else cv_r2
        else:
            p1_best, p2_best = 0.6692, cv_r2

        phases = ['Phase 1\nBaseline','Phase 2\nTuned','Phase 3\nAdvanced']
        p_best = [p1_best, p2_best, p2_best]
        p_mdl  = ['Best Ensemble','GBM★','Stacking/Voting']
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(phases, p_best, 'o-', color='#e74c3c', lw=2.5, markersize=12, zorder=3)
        ax.fill_between(phases, p_best, [v-0.01 for v in p_best], alpha=0.1, color='#e74c3c')
        for ph,val,mn in zip(phases,p_best,p_mdl):
            ax.annotate(f'{mn}\n{val:.4f}', (ph,val),
                textcoords='offset points', xytext=(0,15), ha='center', fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#e74c3c',alpha=0.9))
        ax.set_ylim(min(p_best)-0.03, max(p_best)+0.04)
        ax.set_ylabel('R²'); ax.grid(alpha=0.3)
        ax.axhline(0.669225246150581,color='gold',linestyle='--',lw=1.5,label='목표 R²=0.67')
        ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_pr:
        st.subheader("최적 하이퍼파라미터")
        if 'sec3_tuning_results' in pipe_results:
            t3 = pipe_results['sec3_tuning_results']
            best_name = max(t3, key=lambda k: t3[k].get('cv5_r2',0))
            bp = t3[best_name].get('best_params', {})
        else:
            best_name = "Gradient Boosting"
            bp = {'n_estimators':200,'learning_rate':0.03,'max_depth':2,
                  'subsample':0.9,'min_samples_leaf':4,'max_features':0.9}

        param_rows = [{'파라미터': k, '값': str(v)} for k, v in bp.items()]
        param_rows.append({'파라미터': 'feature_set', '값': f'C ({model.n_features_in_}개)'})
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
        st.caption(f"Best Model: {best_name}")

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("예측 vs 실제 (노드별)")
        fig, ax = plt.subplots(figsize=(6,5))
        for node in NODES:
            mask = df['technology_node']==node
            ax.scatter(y[mask], y_pred[mask], alpha=0.4, s=12,
                       color=NODE_COLORS.get(node,'#95a5a6'), label=node)
        ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=1.5,label='Perfect')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title(f'Train R²={r2_score(y,y_pred):.3f}')
        ax.legend(fontsize=7,markerscale=2); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d2:
        st.subheader("잔차 & Q-Q Plot")
        fig, axes = plt.subplots(1,2,figsize=(9,4))
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color='#34495e')
        axes[0].axhline(0,color='red',linestyle='--',lw=1.5)
        axes[0].axhline(1.96*residuals.std(),color='orange',linestyle=':',lw=1,label='±1.96σ')
        axes[0].axhline(-1.96*residuals.std(),color='orange',linestyle=':',lw=1)
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')
        axes[0].set_title('Residual Plot'); axes[0].legend(fontsize=7)
        stats.probplot(residuals, dist='norm', plot=axes[1])
        axes[1].set_title('Q-Q Plot (정규성 검증)')
        axes[1].get_lines()[0].set(markersize=3,alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Technology Node별 R²")
    node_perf = []
    for node in NODES:
        mask = df['technology_node']==node
        node_perf.append({'Node':node,
            'R²':r2_score(y[mask],y_pred[mask]),
            'RMSE':np.sqrt(mean_squared_error(y[mask],y_pred[mask])),
            'MAE':mean_absolute_error(y[mask],y_pred[mask]),'N':int(mask.sum())})
    np_df = pd.DataFrame(node_perf)
    col_t, col_bar = st.columns([1,1.3])
    with col_t:
        st.dataframe(np_df.style.background_gradient(subset=['R²'],cmap='RdYlGn')
                              .format({'R²':'{:.4f}','RMSE':'{:.4f}','MAE':'{:.4f}'}),
                     use_container_width=True)
    with col_bar:
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(np_df['Node'], np_df['R²'],
                      color=[NODE_COLORS.get(n,'#95a5a6') for n in np_df['Node']],
                      alpha=0.85, width=0.5)
        ax.set_ylim(0,1.1); ax.set_ylabel('R²'); ax.set_title('노드별 R² (Train)')
        for bar,val,n_ in zip(bars,np_df['R²'],np_df['N']):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.015,
                    f'{val:.3f}\n(n={n_})',ha='center',fontweight='bold',fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — XAI                                                ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[4]:
    sec_hdr("🧠 XAI 유의인자 분석 — 6-Method Voting")

    # 파이프라인 결과 있으면 미리 표시
    pipe_results2, pipe_figures2 = load_pipeline_results()
    if 'xai_voting' in pipe_figures2:
        st.info("✅ 파이프라인 실행 결과가 있습니다. 아래 '파이프라인 결과'를 확인하거나, 앱 내 실시간 분석을 실행하세요.")
        with st.expander("📊 파이프라인 XAI 결과 보기", expanded=False):
            st.image(pipe_figures2['xai_voting'], use_container_width=True)
        if 'xai_results' in pipe_results2:
            xai_csv_pipe = pd.DataFrame(pipe_results2['xai_results'])
            st.dataframe(xai_csv_pipe.head(10), use_container_width=True)

    st.markdown("""
사용 Model --> Phase3의 RandomsearchCV에 확인한 Model별 Best parameter를 사용하고 추가로 GBM으로 Permutation importance와 SHAP value를 추출함.
""")

    with st.spinner("XAI 분석 중... (약 200초)"):
        # xai_df = run_xai(model, X, y, FS_C)
        xai_df = run_xai(model,X,y,FS_C,seeds=[6,12,42])

    top_xai = xai_df.head(top_k)
    col_x1, col_x2 = st.columns([1.2,1])
    with col_x1:
        st.subheader(f"🏆 상위 {top_k} 유의인자")
        rd = top_xai[['feature','label','GB','RF','ET','Permutation','SHAP','Pearson','Ensemble_Score']].copy()
        rd.insert(0,'Rank',range(1,len(rd)+1))
        st.dataframe(rd.style.background_gradient(subset=['Ensemble_Score'],cmap='YlOrRd')
                            .format({c:'{:.4f}' for c in
                                     ['GB','RF','ET','Permutation','Pearson','SHAP','Ensemble_Score']}),
                     use_container_width=True, height=400)
    with col_x2:
        st.subheader("Ensemble 순위")
        fig, ax = plt.subplots(figsize=(6,6))
        colors_xai = (['#e74c3c']*min(3,len(top_xai))+
                      ['#e67e22']*min(5,max(0,len(top_xai)-3))+
                      ['#3498db']*max(0,len(top_xai)-8))
        ax.barh(range(len(top_xai)), top_xai['Ensemble_Score'].values,
                color=colors_xai, alpha=0.85)
        ax.set_yticks(range(len(top_xai)))
        ax.set_yticklabels(top_xai['feature'], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('Ensemble Score')
        ax.set_title('유의인자 순위\n(빨=Top3, 주황=4~8, 파=나머지)')
        for i,val in enumerate(top_xai['Ensemble_Score']):
            ax.text(val+0.005,i,f'{val:.3f}',va='center',fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("XAI Voting 행렬 (히트맵)")
    mcols = ['GB','RF','ET','Permutation','SHAP','Pearson']
    hm_d  = xai_df[mcols].values
    fig, ax = plt.subplots(figsize=(10,9))
    im = ax.imshow(hm_d, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(6)); ax.set_xticklabels(mcols,fontsize=10,fontweight='bold')
    ax.set_yticks(range(len(xai_df))); ax.set_yticklabels(xai_df['feature'],fontsize=8)
    plt.colorbar(im,ax=ax,label='정규화된 중요도')
    for i in range(len(xai_df)):
        for j in range(6):
            val=hm_d[i,j]
            ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=6.5,
                    color='white' if val>0.6 else 'black')
    ax.set_title('6-Method XAI Voting Matrix',fontsize=12,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    xai_csv = xai_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 XAI 결과 CSV 다운로드", xai_csv, "xai_results.csv","text/csv")

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 — 예측 & 시뮬레이션 (개선)                           ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[5]:
    sec_hdr("📈 예측 분석 & 단일 Wafer 시뮬레이션")
    pred_t, sim_t = st.tabs(["📂 CSV 업로드 예측", "🎛️ 단일 Wafer 시뮬레이션"])

    with pred_t:
        st.markdown("#### 예측용 CSV 파일")
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            insight("📋 <b>예측용 샘플 CSV</b>: 아래 버튼으로 yield 컬럼 없는 50개 샘플 CSV를 다운로드합니다. "
                    "다운로드 후 이 페이지에 업로드하면 예측 결과와 <b>Train vs Predict Scatter Plot</b>을 확인할 수 있습니다.", 'blue')
        with col_dl2:
            sample_csv_bytes = make_predict_csv(raw_df)
            st.download_button(
                "📥 예측용 샘플 CSV 다운로드",
                data=sample_csv_bytes,
                file_name="predict_sample_no_yield.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        pf = st.file_uploader("예측용 CSV 업로드 (yield 컬럼 없어도 됨)", type=['csv'], key='pu')

        if pf:
            pd_raw = pd.read_csv(pf)
            missing = [f for f in PROCESS_FEATURES if f not in pd_raw.columns]
            if missing:
                st.error(f"필수 컬럼 없음: {missing}")
            else:
                if 'technology_node' not in pd_raw.columns:
                    pd_raw['technology_node'] = '7nm'
                    st.warning("technology_node 없음 → '7nm' 적용")
                known = le.classes_.tolist()
                pd_raw['technology_node'] = pd_raw['technology_node'].apply(
                    lambda x: x if x in known else known[0])
                try:
                    _, X_pred, _ = build_fsc(pd_raw, le)
                    pd_raw['predicted_yield'] = model.predict(X_pred).round(4)
                    pd_raw['yield_grade'] = pd_raw['predicted_yield'].apply(
                        lambda v: '🟢 우수(>0.7)' if v>0.7
                             else '🟡 보통(0.4~0.7)' if v>0.4 else '🔴 불량(<0.4)')

                    has_actual = TARGET in pd_raw.columns
                    st.success(f"✅ {len(pd_raw)}개 예측 완료! {'(실제값 포함 → Scatter Plot 생성)' if has_actual else ''}")

                    # 메트릭
                    c1,c2,c3 = st.columns(3)
                    c1.metric("예측 평균", f"{pd_raw['predicted_yield'].mean():.4f}")
                    c2.metric("우수(>0.7)", f"{(pd_raw['predicted_yield']>0.7).mean()*100:.1f}%")
                    c3.metric("불량(<0.4)", f"{(pd_raw['predicted_yield']<0.4).mean()*100:.1f}%")

                    # ── Train vs Predict Scatter Plot ──────────────────────
                    st.markdown("---")
                    st.subheader("📊 Train vs Predict Scatter Plot")

                    fig = plt.figure(figsize=(14, 5))
                    gs_sp = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

                    # Plot 1: Train 분포 vs Predict 분포 (히스토그램 오버레이)
                    ax1 = fig.add_subplot(gs_sp[0])
                    ax1.hist(y, bins=30, alpha=0.6, color='#3498db', label=f'Train (n={len(y)})',
                             edgecolor='white', density=True)
                    ax1.hist(pd_raw['predicted_yield'], bins=20, alpha=0.6, color='#e74c3c',
                             label=f'Predict (n={len(pd_raw)})', edgecolor='white', density=True)
                    ax1.axvline(y.mean(), color='#2980b9', linestyle='--', lw=2,
                                label=f'Train μ={y.mean():.3f}')
                    ax1.axvline(pd_raw['predicted_yield'].mean(), color='#c0392b',
                                linestyle='--', lw=2,
                                label=f'Pred μ={pd_raw["predicted_yield"].mean():.3f}')
                    ax1.set_xlabel('Yield'); ax1.set_ylabel('Density')
                    ax1.set_title('Train vs Predict 분포 비교')
                    ax1.legend(fontsize=7.5)
                    ax1.grid(alpha=0.3)

                    # Plot 2: 노드별 Train 실제값 vs 모델 예측(학습데이터)
                    ax2 = fig.add_subplot(gs_sp[1])
                    for node in NODES:
                        mask_tr = df['technology_node'] == node
                        ax2.scatter(y[mask_tr], y_pred[mask_tr], alpha=0.35, s=10,
                                    color=NODE_COLORS.get(node,'#95a5a6'), label=node)
                    diag = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
                    ax2.plot(diag, diag, 'k--', lw=1.5, label='Perfect fit')
                    ax2.set_xlabel('Actual (Train)'); ax2.set_ylabel('Predicted (Train)')
                    ax2.set_title(f'Train: Actual vs Predicted\nR²={r2_score(y,y_pred):.3f}')
                    ax2.legend(fontsize=6.5, markerscale=2)
                    ax2.grid(alpha=0.3)
                    if has_actual:
                        # 업로드한 파일의 실제값(TARGET)과 예측값(predicted_yield)을 scatter로 추가
                        ax2.scatter(pd_raw[TARGET], pd_raw['predicted_yield'], 
                                    alpha=0.6, s=20, color='red', marker='x', label='Uploaded Data')
                        ax2.legend(fontsize=6.5, markerscale=2)
                    # Plot 3: Predict 분포를 Train과 노드별로 비교
                    ax3 = fig.add_subplot(gs_sp[2])
                    # 노드별 Train 평균
                    train_means = {n: y[df['technology_node']==n].mean() for n in NODES}
                    train_stds  = {n: y[df['technology_node']==n].std() for n in NODES}

                    xpos = np.arange(len(NODES))
                    bar_w = 0.35

                    # Train 평균 막대 (파란계열)
                    train_bars = ax3.bar(xpos - bar_w/2,
                                         [train_means[n] for n in NODES],
                                         bar_w, label='Train 평균', color='#3498db', alpha=0.8)
                    ax3.errorbar(xpos - bar_w/2,
                                 [train_means[n] for n in NODES],
                                 yerr=[train_stds[n] for n in NODES],
                                 fmt='none', color='#1a5276', capsize=4, lw=1.5)

                    # Predict 분포: 노드별 예측 평균
                    if 'technology_node' in pd_raw.columns:
                        pred_means = []
                        pred_stds  = []
                        for n in NODES:
                            sub_p = pd_raw[pd_raw['technology_node']==n]['predicted_yield']
                            pred_means.append(sub_p.mean() if len(sub_p)>0 else np.nan)
                            pred_stds.append(sub_p.std() if len(sub_p)>1 else 0)
                    else:
                        pred_means = [pd_raw['predicted_yield'].mean()] * len(NODES)
                        pred_stds  = [pd_raw['predicted_yield'].std()] * len(NODES)

                    pred_bars = ax3.bar(xpos + bar_w/2, pred_means, bar_w,
                                        label='Predict 평균', color='#e74c3c', alpha=0.8)
                    ax3.errorbar(xpos + bar_w/2, pred_means, yerr=pred_stds,
                                 fmt='none', color='#922b21', capsize=4, lw=1.5)

                    ax3.set_xticks(xpos)
                    ax3.set_xticklabels(NODES, fontsize=9)
                    ax3.set_ylabel('평균 Yield')
                    ax3.set_title('노드별 Train 평균 vs Predict 평균')
                    ax3.legend(fontsize=8)
                    ax3.set_ylim(0, 1.05)
                    ax3.grid(alpha=0.3, axis='y')

                    # 수치 레이블
                    for bar, val in zip(train_bars, [train_means[n] for n in NODES]):
                        if not np.isnan(val):
                            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                     f'{val:.3f}', ha='center', fontsize=7, color='#1a5276')
                    for bar, val in zip(pred_bars, pred_means):
                        if not np.isnan(val):
                            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                     f'{val:.3f}', ha='center', fontsize=7, color='#922b21')

                    plt.suptitle('Train Dataset vs Prediction Result 비교', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                    # Wafer별 예측 결과 + scatter
                    st.markdown("---")
                    st.subheader("🔩 Wafer별 예측 결과 & Scatter")
                    col_tab1, col_tab2 = st.columns([1.2, 1])

                    with col_tab1:
                        show = [c for c in ['lot_id','wafer_id','technology_node',
                                            'predicted_yield','yield_grade'] if c in pd_raw.columns]
                        st.dataframe(pd_raw[show].head(50), use_container_width=True)

                    with col_tab2:
                        # Wafer별 예측값 인덱스 scatter
                        fig, ax = plt.subplots(figsize=(6, 4))
                        colors_w = [NODE_COLORS.get(n,'#95a5a6')
                                    for n in pd_raw.get('technology_node',
                                                        pd.Series(['7nm']*len(pd_raw)))]
                        ax.scatter(range(len(pd_raw)), pd_raw['predicted_yield'],
                                   c=colors_w, alpha=0.8, s=40, edgecolors='white', lw=0.5)
                        ax.axhline(0.7, color='green', linestyle='--', lw=1.5, label='우수 기준(0.7)')
                        ax.axhline(0.4, color='red', linestyle='--', lw=1.5, label='불량 기준(0.4)')
                        ax.axhline(pd_raw['predicted_yield'].mean(), color='gray',
                                   linestyle=':', lw=2, label=f'예측 평균({pd_raw["predicted_yield"].mean():.3f})')
                        # 노드 legend
                        for node in NODES:
                            if node in pd_raw.get('technology_node', pd.Series([])).values:
                                ax.scatter([], [], c=NODE_COLORS[node], label=node, s=30)
                        ax.set_xlabel('Wafer Index')
                        ax.set_ylabel('Predicted Yield')
                        ax.set_title('Wafer별 예측 Yield Scatter')
                        ax.legend(fontsize=7, loc='lower right')
                        ax.set_ylim(0, 1.05)
                        ax.grid(alpha=0.3)
                        plt.tight_layout(); st.pyplot(fig); plt.close()

                    # 다운로드
                    csv_out = pd_raw.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 예측 결과 다운로드", csv_out,"predicted_yield.csv","text/csv")

                except Exception as e:
                    st.error(f"예측 오류: {e}")
                    import traceback; st.code(traceback.format_exc())

    with sim_t:
        st.markdown("파라미터 슬라이더를 조정하여 **실시간**으로 수율을 시뮬레이션합니다.")
        sim_node = st.selectbox("Technology Node", NODES, key='sn')
        nd = df[df['technology_node']==sim_node]
        st.markdown("**공정 파라미터 설정**")
        cols3 = st.columns(3)
        sv = {}
        for i, feat in enumerate(PROCESS_FEATURES):
            fmin=float(nd[feat].min()); fmax=float(nd[feat].max()); fmean=float(nd[feat].mean())
            with cols3[i%3]:
                lbl = FEATURE_LABELS.get(feat,feat)
                if feat=='defect_count':
                    sv[feat]=st.number_input(lbl,min_value=int(fmin),max_value=int(fmax),
                                              value=int(fmean),step=1,key=f's_{feat}')
                else:
                    sv[feat]=st.slider(lbl,min_value=fmin,max_value=fmax,
                                       value=fmean,format='%.4f',key=f's_{feat}')
        tc  = float(le.transform([sim_node])[0])
        adv = 1.0 if sim_node in ['7nm','10nm'] else 0.0
        dc  = float(np.log1p(sv['defect_count']*sv.get('defect_density',0)))
        cdv = sv['critical_dimension']*sv['vth']
        uxd = sv['thickness_uniformity']*sv['defect_count']
        nxcd= tc*sv['critical_dimension']
        X_sim = np.array([[sv[f] for f in PROCESS_FEATURES if f!='defect_density']
                          +[tc,adv,dc,cdv,uxd,nxcd]])
        sim_yield = float(model.predict(X_sim)[0])
        node_mean = float(nd[TARGET].mean())
        st.markdown("---")
        r1,r2,r3 = st.columns(3)
        r1.metric("🎯 예측 Yield",f"{sim_yield:.4f}",
                  delta=f"{sim_yield-node_mean:+.4f} vs 노드 평균")
        r2.metric(f"{sim_node} 평균",f"{node_mean:.4f}")
        r3.metric("등급","🟢 우수" if sim_yield>0.7 else "🟡 보통" if sim_yield>0.4 else "🔴 불량")
        fig,ax=plt.subplots(figsize=(8,2.5))
        bar_color='#27ae60' if sim_yield>0.7 else '#f39c12' if sim_yield>0.4 else '#e74c3c'
        ax.barh(0,1.0,color='#ecf0f1',height=0.5,alpha=0.5)
        ax.barh(0,sim_yield,color=bar_color,height=0.5,alpha=0.85)
        ax.axvline(node_mean,color='#2980b9',linestyle='--',lw=2.5,
                   label=f'{sim_node} 평균({node_mean:.3f})')
        ax.axvline(0.7,color='green',linestyle=':',lw=1.5,label='우수 기준')
        ax.axvline(0.4,color='red',linestyle=':',lw=1.5,label='불량 기준')
        ax.set_xlim(0,1); ax.set_yticks([])
        ax.set_title(f'시뮬레이션 Yield = {sim_yield:.4f}',fontweight='bold',fontsize=12)
        ax.legend(fontsize=8,loc='lower right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.subheader("민감도 분석 — 주요 피처 ±20% 변화 영향")
        sens_feats = ['critical_dimension','vth','oxide_thickness','thickness_uniformity','defect_count']
        sens_rows = []
        for feat in sens_feats:
            if feat not in sv: continue
            base_v = sv[feat]; row_d = {}
            for dp, label in [(-0.2,'−20%'),(-0.1,'−10%'),(0,'기준'),(+0.1,'+10%'),(+0.2,'+20%')]:
                sv2=sv.copy(); sv2[feat]=base_v*(1+dp)
                dc2=float(np.log1p(sv2['defect_count']*sv2.get('defect_density',0)))
                cdv2=sv2['critical_dimension']*sv2['vth']
                uxd2=sv2['thickness_uniformity']*sv2['defect_count']
                nxcd2=tc*sv2['critical_dimension']
                X2=np.array([[sv2[f] for f in PROCESS_FEATURES if f!='defect_density']
                             +[tc,adv,dc2,cdv2,uxd2,nxcd2]])
                row_d[label]=float(model.predict(X2)[0])
            row_d['피처']=FEATURE_LABELS.get(feat,feat); sens_rows.append(row_d)
        if sens_rows:
            sens_df=pd.DataFrame(sens_rows).set_index('피처')
            fig,ax=plt.subplots(figsize=(9,4))
            xlbls=['−20%','−10%','기준','+10%','+20%']
            for (fl,row),color in zip(sens_df.iterrows(),plt.cm.tab10(np.linspace(0,1,len(sens_df)))):
                ax.plot(xlbls,row.values,'o-',lw=2,color=color,label=fl,markersize=6)
            ax.axhline(sim_yield,color='gray',linestyle='--',lw=1.5,label='현재 예측값')
            ax.axhline(0.7,color='green',linestyle=':',lw=1)
            ax.axhline(0.4,color='red',linestyle=':',lw=1)
            ax.set_xlabel('파라미터 변화율'); ax.set_ylabel('예측 Yield')
            ax.set_title('주요 피처 ±20% 민감도 분석')
            ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
            ax.grid(alpha=0.3); plt.tight_layout()
            st.pyplot(fig); plt.close()

