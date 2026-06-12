"""
강화된 RAG 엔진 — 쓸수록 똑똑해지는 설계

5개 레이어:
  ① 쿼리 강화    : HyDE, Multi-query 확장, 대화 맥락 반영
  ② 하이브리드 검색: 벡터(ChromaDB) + BM25 키워드, RRF 융합
  ③ 재순위화      : Cross-encoder + 기억(메모/과거 인사이트) 주입
  ④ LLM 생성     : 스트리밍, 출처 인용, 신뢰도 점수
  ⑤ 학습 루프    : 피드백 → DB 저장, 인사이트 누적, 검색 개선
"""
import re
import json
import requests
from pathlib import Path
from typing import Optional, Literal, Generator

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from .retriever import BM25, reciprocal_rank_fusion, QueryEnhancer, Reranker
from .memory_store import MemoryStore

CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K_VECTOR  = 10   # 벡터 검색 후보
TOP_K_BM25    = 10   # BM25 후보
TOP_K_FINAL   = 6    # 최종 컨텍스트 청크 수

DEFAULT_MODELS = {
    "ollama":   "llama3.2",
    "groq":     "llama-3.3-70b-versatile",
    "lmstudio": "local-model",
}

SYSTEM_PROMPT = """당신은 반도체 및 과학기술 논문 분석 전문가 AI입니다.
사용자가 제공한 논문 컨텍스트와 기억(과거 메모, 인사이트)을 바탕으로 깊이 있는 분석을 제공합니다.

응답 지침:
- 반드시 한국어로 답변
- 논문명, 저자, 연도를 구체적으로 인용하여 근거 제시
- 논문 간 연관성과 흐름, 기술적 발전 맥락을 포착
- 컨텍스트에 없는 내용은 추측하지 않고 명시
- 답변 마지막에 **[출처]** 섹션으로 인용 논문 목록 제시
- 답변 마지막에 **[신뢰도]** 를 상/중/하 로 표시 (컨텍스트 풍부도 기준)"""


class RAGEngine:
    def __init__(
        self,
        pdf_dir: Path,
        backend: Literal["ollama", "groq", "lmstudio"] = "ollama",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        db_path: Path = Path("./rag_memory.db"),
        use_hyde: bool = True,
        use_multiquery: bool = True,
        use_reranker: bool = True,
    ):
        self.pdf_dir  = pdf_dir
        self.backend  = backend
        self.model    = model or DEFAULT_MODELS[backend]
        self.api_key  = api_key
        self.base_url = base_url or self._default_url(backend)
        self.use_hyde       = use_hyde
        self.use_multiquery = use_multiquery
        self.use_reranker   = use_reranker

        # 컴포넌트
        self.memory   = MemoryStore(db_path)
        self.enhancer = QueryEnhancer(backend, self.model, api_key, self.base_url)
        self.reranker = Reranker()

        # 내부 상태
        self._embedder  = None
        self._chroma    = None
        self._collection= None
        self._bm25: Optional[BM25] = None
        self._all_docs: list[str]  = []  # BM25용 전체 청크
        self._all_metas: list[dict]= []
        self._ready = False

    # ── 기본 URL ─────────────────────────────────────────────────────────────
    def _default_url(self, b: str) -> str:
        return {"ollama":"http://localhost:11434",
                "groq":"https://api.groq.com/openai/v1",
                "lmstudio":"http://localhost:1234/v1"}[b]

    # ── 연결 확인 ─────────────────────────────────────────────────────────────
    def check_connection(self) -> tuple[bool, str]:
        try:
            if self.backend == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if any(self.model.split(":")[0] in m for m in models):
                        return True, f"✅ Ollama — {self.model}"
                    return False, f"⚠️ 모델 없음 → `ollama pull {self.model}`"
                return False, "❌ Ollama 연결 불가"
            elif self.backend == "groq":
                if not self.api_key:
                    return False, "❌ Groq API 키 필요"
                r = requests.get("https://api.groq.com/openai/v1/models",
                                 headers={"Authorization": f"Bearer {self.api_key}"}, timeout=8)
                return r.status_code == 200, "✅ Groq 연결됨" if r.status_code == 200 else f"❌ Groq 인증 실패"
            elif self.backend == "lmstudio":
                r = requests.get(f"{self.base_url}/models", timeout=5)
                return r.status_code == 200, "✅ LM Studio 연결됨" if r.status_code == 200 else "❌ LM Studio 연결 불가"
        except requests.exceptions.ConnectionError:
            return False, f"❌ {self.backend} 서버 연결 불가"
        except Exception as e:
            return False, f"❌ {e}"

    # ── 인덱스 구축 ───────────────────────────────────────────────────────────
    def build_index(self, papers: dict) -> int:
        for flag, name in [(HAS_PDF,"pdfplumber"),(HAS_ST,"sentence-transformers"),(HAS_CHROMA,"chromadb")]:
            if not flag:
                raise RuntimeError(f"{name} 미설치: pip install {name}")

        self._embedder = SentenceTransformer(EMBED_MODEL)

        # ChromaDB — 영구 저장 (PersistentClient)
        persist_dir = str(self.pdf_dir.parent / "chroma_db")
        self._chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self._chroma.delete_collection("papers")
        except Exception:
            pass
        self._collection = self._chroma.create_collection(
            name="papers", metadata={"hnsw:space": "cosine"}
        )

        self._all_docs  = []
        self._all_metas = []
        total = 0

        for pdf_path in self.pdf_dir.glob("*.pdf"):
            doi   = self._path_to_doi(pdf_path)
            paper = papers.get(doi, {})
            title = paper.get("title", pdf_path.stem)
            year  = str(paper.get("year", ""))
            authors = ", ".join((paper.get("authors") or [])[:3])

            text   = self._extract_text(pdf_path)
            chunks = self._chunk_text(text) if text else []
            if not chunks:
                continue

            # 메타정보를 청크 앞에 prefix로 추가 (BM25 정확도 향상)
            enriched = [f"[{title} | {authors} | {year}]\n{c}" for c in chunks]
            embs = self._embedder.encode(enriched, show_progress_bar=False).tolist()

            ids   = [f"{doi}__c{i}" for i in range(len(chunks))]
            metas = [{"doi":doi,"title":title,"year":year,"chunk_idx":i}
                     for i in range(len(chunks))]

            for b in range(0, len(chunks), 100):
                self._collection.add(
                    ids=ids[b:b+100], embeddings=embs[b:b+100],
                    documents=enriched[b:b+100], metadatas=metas[b:b+100]
                )

            self._all_docs.extend(enriched)
            self._all_metas.extend(metas)
            total += len(chunks)

        # BM25 인덱스 구축
        if self._all_docs:
            self._bm25 = BM25(self._all_docs)

        self._ready = True
        return total

    # ── 메인 쿼리 ─────────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        history: list[dict] = None,
        stream: bool = False,
    ) -> tuple[str, int]:
        """
        Returns: (answer_text, query_id)
        query_id는 피드백 등록에 사용
        """
        if not self._ready:
            return "RAG 인덱스가 구축되지 않았습니다.", -1

        history = history or []

        # ── ① 쿼리 강화 ──────────────────────────────────────────────────────
        queries = [question]

        if self.use_multiquery:
            try:
                queries = self.enhancer.expand_queries(question)
            except Exception:
                queries = [question]

        hyde_doc = ""
        if self.use_hyde:
            try:
                hyde_doc = self.enhancer.generate_hyde_doc(question)
            except Exception:
                pass

        # ── ② 하이브리드 검색 ─────────────────────────────────────────────────
        n_docs = self._collection.count()
        if n_docs == 0:
            return "인덱싱된 문서가 없습니다.", -1

        vector_results = self._vector_search(queries, hyde_doc, n_docs)
        bm25_results   = self._bm25_search(question, n_docs)

        # RRF 병합
        fused = reciprocal_rank_fusion([
            [i for i,_ in vector_results],
            [i for i,_ in bm25_results],
        ])
        top_indices = [i for i,_ in fused[:TOP_K_FINAL * 2]]

        # ── ③ 재순위화 ────────────────────────────────────────────────────────
        candidate_docs  = [self._all_docs[i]  for i in top_indices if i < len(self._all_docs)]
        candidate_metas = [self._all_metas[i] for i in top_indices if i < len(self._all_metas)]

        if self.reranker.available and len(candidate_docs) > TOP_K_FINAL:
            reranked_idx = self.reranker.rerank(question, candidate_docs, TOP_K_FINAL)
            candidate_docs  = [candidate_docs[i]  for i in reranked_idx]
            candidate_metas = [candidate_metas[i] for i in reranked_idx]
        else:
            candidate_docs  = candidate_docs[:TOP_K_FINAL]
            candidate_metas = candidate_metas[:TOP_K_FINAL]

        # ── 기억 주입 ─────────────────────────────────────────────────────────
        memory_context = self._build_memory_context(question)

        # ── ④ LLM 생성 ────────────────────────────────────────────────────────
        context = self._build_context(candidate_docs, candidate_metas)
        context_dois = list({m.get("doi","") for m in candidate_metas})

        answer = self._generate(question, context, memory_context, history)

        # ── ⑤ 학습 루프 — 자동 저장 ──────────────────────────────────────────
        query_id = self.memory.save_query(question, answer, context_dois)

        # 인사이트 자동 감지 (특정 키워드가 있으면 저장 제안)
        return answer, query_id

    # ── 벡터 검색 ─────────────────────────────────────────────────────────────
    def _vector_search(
        self, queries: list[str], hyde_doc: str, n_docs: int
    ) -> list[tuple[int, float]]:
        all_queries = queries.copy()
        if hyde_doc:
            all_queries.append(hyde_doc)

        idx_scores: dict[int, float] = {}
        for q in all_queries:
            emb = self._embedder.encode([q]).tolist()
            res = self._collection.query(
                query_embeddings=emb,
                n_results=min(TOP_K_VECTOR, n_docs),
                include=["documents","metadatas","distances"],
            )
            docs_returned = res.get("documents",[[]])[0]
            dists = res.get("distances",[[]])[0]

            for doc, dist in zip(docs_returned, dists):
                # 전역 인덱스 찾기
                try:
                    idx = self._all_docs.index(doc)
                    score = 1.0 - dist   # cosine distance → similarity
                    if idx not in idx_scores or score > idx_scores[idx]:
                        idx_scores[idx] = score
                except ValueError:
                    pass

        return sorted(idx_scores.items(), key=lambda x: -x[1])

    # ── BM25 검색 ─────────────────────────────────────────────────────────────
    def _bm25_search(self, question: str, n_docs: int) -> list[tuple[int, float]]:
        if not self._bm25:
            return []
        return self._bm25.get_top_n(question, min(TOP_K_BM25, n_docs))

    # ── 기억 컨텍스트 ─────────────────────────────────────────────────────────
    def _build_memory_context(self, question: str) -> str:
        parts = []

        # 유사한 과거 질문 (👍 받은 것 우선)
        similar = self.memory.search_similar_queries(question, limit=3)
        if similar:
            parts.append("=== 관련 과거 질의 ===")
            for q in similar[:2]:
                if q.get("feedback", 0) >= 0:
                    parts.append(f"Q: {q['question']}\nA: {q['answer'][:300]}...")

        # 고정된 인사이트
        pinned = self.memory.get_insights(pinned_only=True)
        if pinned:
            parts.append("=== 📌 저장된 핵심 인사이트 ===")
            for ins in pinned[:3]:
                parts.append(f"[{ins['title']}]\n{ins['content'][:400]}")

        # 사용자 메모 (전체)
        notes = self.memory.get_all_notes()
        if notes:
            parts.append("=== 사용자 메모 ===")
            for n in notes[:5]:
                tags = json.loads(n.get("tags","[]") or "[]")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                parts.append(f"• {n['doi']}{tag_str}: {n['note']}")

        return "\n\n".join(parts)

    # ── 컨텍스트 문자열 ───────────────────────────────────────────────────────
    def _build_context(self, docs: list[str], metas: list[dict]) -> str:
        parts = []
        seen_titles = set()
        for doc, meta in zip(docs, metas):
            title = meta.get("title","")
            year  = meta.get("year","")
            key   = f"{title}_{meta.get('chunk_idx',0)}"
            if key in seen_titles:
                continue
            seen_titles.add(key)
            parts.append(f"[{title} ({year})]\n{doc}")
        return "\n\n---\n\n".join(parts)

    # ── LLM 호출 ─────────────────────────────────────────────────────────────
    def _generate(
        self,
        question: str,
        context: str,
        memory_context: str,
        history: list[dict],
    ) -> str:
        memory_section = f"\n\n{memory_context}" if memory_context.strip() else ""
        user_content = (
            f"[논문 컨텍스트]\n{context}"
            f"{memory_section}"
            f"\n\n---\n\n질문: {question}"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        if self.backend == "ollama":
            return self._call_ollama(messages)
        else:
            return self._call_openai_compat(messages)

    def _call_ollama(self, messages: list[dict]) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={"model":self.model,"messages":messages,"stream":False,
                      "options":{"temperature":0.3,"num_predict":2048}},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.Timeout:
            return "⏱️ 응답 시간 초과. 잠시 후 다시 시도하세요."
        except Exception as e:
            return f"❌ Ollama 오류: {e}"

    def _call_openai_compat(self, messages: list[dict]) -> str:
        headers = {"Content-Type":"application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model":self.model,"messages":messages,"max_tokens":2048,"temperature":0.3},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            return "⏱️ 응답 시간 초과."
        except Exception as e:
            return f"❌ {self.backend} 오류: {e}"

    # ── 피드백 등록 ───────────────────────────────────────────────────────────
    def give_feedback(self, query_id: int, positive: bool):
        self.memory.set_feedback(query_id, 1 if positive else -1)

    # ── 인사이트 저장 ─────────────────────────────────────────────────────────
    def save_insight(self, title: str, content: str, source_dois: list[str]) -> int:
        return self.memory.save_insight(title, content, source_dois)

    # ── PDF 텍스트 추출 ───────────────────────────────────────────────────────
    def _extract_text(self, pdf_path: Path) -> str:
        try:
            parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:40]:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            text = "\n".join(parts)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            return text.strip()
        except Exception as e:
            print(f"[RAG] PDF 추출 실패 {pdf_path}: {e}")
            return ""

    # ── 청킹 ─────────────────────────────────────────────────────────────────
    def _chunk_text(self, text: str) -> list[str]:
        if len(text) < 100:
            return []
        chunks, start = [], 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end]
            if end < len(text):
                last = max(chunk.rfind('. '), chunk.rfind('.\n'), chunk.rfind('? '))
                if last > CHUNK_SIZE // 2:
                    chunk = chunk[:last+1]
                    end   = start + last + 1
            chunk = chunk.strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP
        return chunks

    # ── DOI 복원 ─────────────────────────────────────────────────────────────
    def _path_to_doi(self, path: Path) -> str:
        stem = path.stem
        if stem.startswith("10_"):
            parts = stem.split("_", 2)
            if len(parts) >= 2:
                prefix = f"10.{parts[1]}"
                suffix = parts[2] if len(parts) > 2 else ""
                return f"{prefix}/{suffix}".rstrip("/")
        return stem.replace("_", "/", 1)
