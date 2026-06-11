# 반도체 수율 예측 & XAI 분석 — Final Version

## 빠른 시작
```bash
pip install -r requirements.txt
streamlit run semiconductor_yield_xai.py  # 서비스
python run_full_pipeline.py               # 파이프라인 전체
python run_full_pipeline.py --steps 3 4  # 튜닝+앙상블만
```

## 23개 모델 구성
| 계열 | 모델 |
|---|---|
| 선형(5) | Linear · Ridge · Lasso · ElasticNet · PLS |
| 이상치강건(5) | Huber · BayesianRidge · TheilSen · KernelRidge · RANSAC+Ridge |
| 비선형(4) | SVR · NuSVR · KNN · MLP |
| 트리(1) | Decision Tree |
| 앙상블(4) | Bagging · RF · ExtraTrees · AdaBoost |
| 부스팅(4) | GBM · XGB(실제/대안) · LGBM(실제/대안) · CB(실제/대안) |

XGBoost/LightGBM/CatBoost 미설치 시 **HistGBM 대안 자동 적용**

## §8 의사결정 흐름 (Streamlit 탭 6)
```
① 22nm R²<0.5 발견 → 노드별 개별 모델 → ❌ 기각 (샘플부족+정보손실)
② 동종Stacking=GBM → 이기종(GBM+SVR+RF) → ❌ 기각 (공통편향 공유)
③ R² 정체 → FS-D 추가 파생 → ❌ 기각 (GBM 내부 비선형 포착)
④ 커널모델 낮음 → 이분화 구조 문제 규명 🔍
→ 결론: 한계 원인은 알고리즘 아님. 데이터 규모 + 이분화 구조
```

## 재현성 보장
- `best_params` 포함 모든 중간 결과 `output/results/` JSON 저장
- 각 단계 독립 실행 (이전 결과 자동 복원)
- `RANDOM_STATE=42` 전역 고정
