"""
하이브리드 검색 + 쿼리 강화 모듈

기능:
  1. HyDE  — 가상 문서로 임베딩 품질 향상
  2. Multi-query — 질문을 3개로 분해해 재검색
  3. BM25  — TF-IDF 키워드 검색 (rank_bm25 없으면 자체 구현)
  4. RRF   — Reciprocal Rank Fusion으로 결과 병합
  5. Cross-encoder 재순위화 (sentence-transformers 있을 때)
"""
import re
import math
import requests
from typing import Optional
from collections import defaultdict

try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except ImportError:
    HAS_CE = False


# ── BM25 경량 구현 ─────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = corpus
        self.tokenized = [self._tokenize(d) for d in corpus]
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[a-zA-Z가-힣]{2,}', text.lower())

    def _build(self):
        N = len(self.tokenized)
        if N == 0:
            return
        total_len = 0
        for tokens in self.tokenized:
            total_len += len(tokens)
            seen = set(tokens)
            for t in seen:
                self.df[t] += 1
        self.avgdl = total_len / N
        for term, df in self.df.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: str) -> list[float]:
        q_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.tokenized):
            dl = len(doc_tokens)
            score = 0.0
            tf_map: dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1
            for t in q_tokens:
                if t not in self.idf:
                    continue
                tf = tf_map.get(t, 0)
                score += self.idf[t] * (
                    tf * (self.k1 + 1) /
                    (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                )
            scores.append(score)
        return scores

    def get_top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(idx, sc) for idx, sc in ranked[:n] if sc > 0]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """여러 순위 목록을 RRF로 병합"""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── 쿼리 강화 ──────────────────────────────────────────────────────────────────
class QueryEnhancer:
    def __init__(self, backend: str, model: str, api_key: str, base_url: str):
        self.backend  = backend
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url

    def expand_queries(self, question: str) -> list[str]:
        """질문을 3개의 다른 관점으로 확장"""
        prompt = f"""다음 논문 검색 질문을 3가지 다른 관점으로 바꿔서, JSON 배열로만 반환하세요.
원본 질문과 의미는 같되 표현을 달리해주세요.
원본: "{question}"
반환 형식: ["질문1", "질문2", "질문3"]
설명 없이 JSON만 반환하세요."""
        try:
            resp = self._call_llm(prompt, max_tokens=300)
            # JSON 파싱 시도
            import json
            match = re.search(r'\[.*?\]', resp, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                return [question] + [v for v in variants if isinstance(v, str)][:3]
        except Exception:
            pass
        return [question]

    def generate_hyde_doc(self, question: str) -> str:
        """HyDE: 질문에 대한 가상의 논문 구절 생성 → 임베딩 품질 향상"""
        prompt = f"""다음 질문에 답하는 학술 논문의 한 구절을 150단어 이내로 작성하세요.
실제 논문처럼 구체적이고 기술적으로 작성하되 영어로 작성하세요.
질문: {question}
구절:"""
        try:
            return self._call_llm(prompt, max_tokens=200)
        except Exception:
            return question

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        if self.backend == "ollama":
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt,
                      "stream": False, "options": {"num_predict": max_tokens, "temperature": 0.3}},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.3},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ── Cross-Encoder 재순위화 ───────────────────────────────────────────────────
class Reranker:
    CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._model = None
        if HAS_CE:
            try:
                self._model = CrossEncoder(self.CE_MODEL)
            except Exception as e:
                print(f"[Reranker] CrossEncoder 로드 실패: {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, docs: list[str], top_k: int = 6) -> list[int]:
        """재순위화 후 상위 top_k 인덱스 반환"""
        if not self._model or not docs:
            return list(range(min(top_k, len(docs))))
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return ranked[:top_k]
