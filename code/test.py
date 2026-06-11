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
print(df['tech_encoded'].unique())
