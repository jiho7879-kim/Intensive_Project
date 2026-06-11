"""
config.py

"""

import os

# ── 경로 ──────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "semiconductor_yield_forecasting_data.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 피처 / 타겟 ───────────────────────────────
TARGET = "yield"

PROCESS_FEATURES = [
    "etch_rate", "pressure", "temperature", "exposure_time",
    "focus_offset", "dose", "deposition_rate", "thickness_uniformity",
    "implant_energy", "tilt_angle", "critical_dimension", "oxide_thickness",
    "resistivity", "defect_count", "defect_density", "vth", "leakage_current",
    "resistance",
]

# technology_node 를 Label Encoding 한 컬럼을 포함한 최종 피처 리스트
# (실제 LabelEncoder 는 각 스크립트에서 fit 후 아래 이름으로 컬럼 추가)
ENCODED_NODE_COL = "tech_encoded"
ALL_FEATURES = PROCESS_FEATURES + [ENCODED_NODE_COL]

# ── Tool Group 매핑 --> 특정 tool의 파생 공정 변수로 EDA 분석에 활용 예정
TOOL_GROUPS = {
    "etch_tool":       ["etch_rate", "pressure", "temperature"],
    "litho_tool":      ["focus_offset", "dose"],
    "deposition_tool": ["deposition_rate", "thickness_uniformity"],
    "implant_tool":    ["implant_energy", "tilt_angle"],
}

# ── 피처 한글/설명 라벨 -->> plot등 후속 description에 활용예정
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

# ── Technology Node ───────────────────────────
TECH_NODES  = ["7nm", "10nm", "14nm", "22nm", "28nm"]
NODE_COLORS = {
    "7nm":  "#e74c3c",
    "10nm": "#e67e22",
    "14nm": "#f39c12",
    "22nm": "#27ae60",
    "28nm": "#2980b9",
}

# ── ML 공통 설정 ──────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── EDA 반영 피처셋 C 파생 변수 (step3/4에서 생성)
DERIVED_FEATURES = [
    "is_advanced_node",    # 7/10nm=1, 나머지=0 (이분화 명시)
    "defect_composite",    # log(defect_count × defect_density) — 복합 결함 지수
    "Swing",      # Vtsat / Log10(leakage_current) --> Subthredshold swing과 유사한 인자(도메인 knowledge 기반)
    "IR_Drop", # V=IR --> 도메인 knowledge 기반
    "node_x_cd",           # tech_encoded × critical_dimension — 노드×CD 상호작용
]

# ── EDA 반영 최적 피처셋 C
# defect_density 제거 (thickness_uniformity와 r=0.9996) + 파생 변수 5개 추가
FS_B_BASE = [f for f in PROCESS_FEATURES if f != "defect_density"] + [ENCODED_NODE_COL]
FS_C_FEATURES = FS_B_BASE + DERIVED_FEATURES   
