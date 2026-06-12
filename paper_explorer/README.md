# 🔬 Paper Explorer

DOI 기반 논문 인용 네트워크 탐색 + RAG 인사이트 도출 시스템

## 📐 아키텍처

```
paper_explorer/
├── app.py                  # Streamlit 메인 앱
├── requirements.txt
├── core/
│   ├── paper_fetcher.py    # Semantic Scholar API 래퍼
│   └── graph_builder.py    # BFS 인용 네트워크 구축
├── ui/
│   ├── graph_renderer.py   # vis.js 인터랙티브 그래프 HTML 생성
│   └── sidebar.py          # 설정 사이드바
└── rag/
    ├── pdf_downloader.py   # Unpaywall / OpenAlex PDF 수집
    └── rag_engine.py       # ChromaDB + Claude API RAG
```

## 🔧 설치

```bash
# 1. 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
streamlit run app.py
```

## 🚀 사용 방법

### Step 1 — API 키 설정 (사이드바)
- **Anthropic API Key**: RAG 질의응답에 필요 (`sk-ant-...`)
- **Semantic Scholar API Key**: 선택사항, Rate limit 완화

### Step 2 — 논문 네트워크 탐색
1. DOI 입력 (예: `10.1038/nature12373`)
2. **탐색 깊이** / **최대 논문 수** 설정
3. `🕸️ 네트워크 탐색` 클릭
4. **인용 그래프 탭**에서 인터랙티브 시각화 확인

### Step 3 — PDF 수집
1. `📥 전체 PDF 수집 시작` 클릭
2. Unpaywall / OpenAlex 통해 오픈액세스 PDF 자동 다운로드

### Step 4 — RAG 인사이트
1. `🧠 RAG 인덱스 구축` 클릭 (PDF 청킹 + 임베딩)
2. **RAG 인사이트 탭**에서 AI 질의응답
3. 프리셋 질문 또는 자유 질문 입력

## 📊 그래프 해석

| 노드 색상 | 의미 |
|-----------|------|
| 🟣 보라 (대) | 루트 논문 |
| 🟣 보라 (소) | 피인용 500회+ |
| 🔵 파랑 | 피인용 100~499회 |
| 🟢 청록 | 피인용 20~99회 |
| ⚫ 회색 | 피인용 ~19회 |

- **노드 크기** ∝ 피인용 수
- **화살표 방향**: A → B = "A가 B를 인용함"

## ⚠️ 주의사항

- **오픈액세스 논문만** 자동 다운로드 가능 (유료 저널은 기관 접속 필요)
- Semantic Scholar API 무료 티어: 분당 100 요청
- ChromaDB는 **인메모리** 기본 동작 (앱 재시작 시 재인덱싱 필요)
  → 영속화: `chromadb.PersistentClient(path="./chroma_db")` 로 변경

## 🔑 무료 API 발급

| API | URL |
|-----|-----|
| Semantic Scholar | https://www.semanticscholar.org/product/api |
| Anthropic | https://console.anthropic.com |
