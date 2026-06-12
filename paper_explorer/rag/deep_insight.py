"""
Deep Insight Engine — 엔지니어링 특화 분석
  1. 파라미터 자동 추출 (VDD, Vth, Cell Ratio 등)
  2. 실험 조건 매칭 엔진
  3. 논문 비교 뷰어 (테이블 형식)
  4. Debate Mode — 상반된 논문 토론
  5. 증상 기반 역추적 (Symptom-to-Theory)
"""
import json
import re
import requests
from typing import Optional


# ── SRAM/반도체 파라미터 패턴 사전 ─────────────────────────────────────────────
PARAM_PATTERNS = {
    # 전압/전력
    "VDD":        r'V[Dd][Dd]\s*[=:≈~]\s*([\d.]+)\s*(V|mV)?',
    "Vth":        r'V(?:th|t)\s*[=:≈~]\s*([\d.]+)\s*(V|mV)?',
    "Vmin":       r'V[Mm]in\s*[=:≈~]\s*([\d.]+)\s*(V|mV)?',
    "SNM":        r'SNM\s*[=:≈~]\s*([\d.]+)\s*(mV|V)?',
    "RSNM":       r'RSNM\s*[=:≈~]\s*([\d.]+)\s*(mV|V)?',
    "WSNM":       r'WSNM\s*[=:≈~]\s*([\d.]+)\s*(mV|V)?',
    # 전류
    "Ion":        r'I[Oo]n\s*[=:≈~]\s*([\d.]+)\s*(µA|uA|nA|mA)?',
    "Ioff":       r'I[Oo]ff\s*[=:≈~]\s*([\d.]+)\s*(fA|pA|nA)?',
    "Ileak":      r'I[Ll]eak(?:age)?\s*[=:≈~]\s*([\d.]+)\s*(fA|pA|nA)?',
    # 공정/구조
    "Cell Ratio": r'[Cc]ell\s*[Rr]atio\s*[=:≈~]\s*([\d.]+)',
    "Pull-up Ratio": r'[Pp]ull[-\s]?[Uu]p\s*[Rr]atio\s*[=:≈~]\s*([\d.]+)',
    "Fin Width":  r'[Ff]in\s*[Ww](?:idth)?\s*[=:≈~]\s*([\d.]+)\s*(nm)?',
    "Gate Length":r'[Ll][Gg]?\s*[=:≈~]\s*([\d.]+)\s*(nm)?',
    "Metal Pitch":r'[Mm]etal\s*[Pp]itch\s*[=:≈~]\s*([\d.]+)\s*(nm)?',
    "Node":       r'(\d+)\s*nm\s*(?:node|process|technology)',
    # 성능
    "Access Time":r'[Aa]ccess\s*[Tt]ime\s*[=:≈~]\s*([\d.]+)\s*(ns|ps)?',
    "Write Time": r'[Ww]rite\s*[Tt]ime\s*[=:≈~]\s*([\d.]+)\s*(ns|ps)?',
    "Area":       r'[Cc]ell\s*[Aa]rea\s*[=:≈~]\s*([\d.]+)\s*(µm²|um²|nm²)?',
    "Yield":      r'[Yy]ield\s*[=:≈~]\s*([\d.]+)\s*(%)?',
    "Sigma":      r'(\d+(?:\.\d+)?)\s*[-–]\s*sigma',
    # 온도
    "Temp":       r'(?:T(?:emp)?|temperature)\s*[=:≈~]\s*(-?[\d.]+)\s*(°C|K|C)?',
}

# 기술 키워드 → 관련 파라미터 매핑
TECH_PARAM_MAP = {
    "FinFET":     ["Fin Width","Gate Length","VDD","Vmin","SNM"],
    "GAA":        ["Gate Length","VDD","Vmin","SNM","Ion","Ioff"],
    "SRAM":       ["VDD","Vmin","SNM","Cell Ratio","Yield","Sigma"],
    "write":      ["WSNM","Write Time","Cell Ratio","Pull-up Ratio"],
    "read":       ["RSNM","Access Time","Cell Ratio"],
    "leakage":    ["Ioff","Ileak","Temp","VDD"],
    "yield":      ["Yield","Sigma","Vmin"],
}


class DeepInsightEngine:
    def __init__(self, backend: str, model: str, api_key: str, base_url: str):
        self.backend  = backend
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url

    # ══════════════════════════════════════════════════════════════════════════
    # 1. 파라미터 자동 추출
    # ══════════════════════════════════════════════════════════════════════════
    def extract_parameters(self, chunks: list[str], metas: list[dict]) -> dict:
        """
        청크 텍스트에서 물리적 파라미터를 추출하여 논문별로 정리
        Returns: { doi: { param_name: [(value, unit, context_snippet), ...] } }
        """
        results: dict[str, dict] = {}

        for chunk, meta in zip(chunks, metas):
            doi = meta.get("doi","")
            title = meta.get("title","")
            if doi not in results:
                results[doi] = {"title": title, "params": {}}

            for param_name, pattern in PARAM_PATTERNS.items():
                matches = re.finditer(pattern, chunk, re.IGNORECASE)
                for m in matches:
                    value = m.group(1)
                    unit = m.group(2) if m.lastindex >= 2 else ""
                    # 주변 문맥 (30자)
                    start = max(0, m.start() - 30)
                    end   = min(len(chunk), m.end() + 30)
                    context = chunk[start:end].replace("\n", " ").strip()

                    if param_name not in results[doi]["params"]:
                        results[doi]["params"][param_name] = []
                    results[doi]["params"][param_name].append({
                        "value": value,
                        "unit": unit or "",
                        "context": context,
                    })

        # LLM으로 추가 파라미터 심층 추출 (정규식이 놓친 것)
        return results

    def extract_parameters_llm(
        self, chunks: list[str], metas: list[dict]
    ) -> list[dict]:
        """
        LLM 기반 구조화 파라미터 추출 — 정규식보다 유연
        Returns: [ {paper, param, value, unit, condition, source_text} ]
        """
        # 논문별로 청크 합치기
        paper_texts: dict[str, str] = {}
        for chunk, meta in zip(chunks, metas):
            doi = meta.get("doi","")
            title = meta.get("title","")
            key = f"{doi}|{title}"
            paper_texts[key] = paper_texts.get(key,"") + "\n" + chunk[:1500]

        all_params = []
        for key, text in list(paper_texts.items())[:5]:  # 최대 5편
            doi, title = key.split("|",1)
            prompt = f"""다음 반도체 논문 텍스트에서 물리적 파라미터와 수치를 추출하세요.

논문: {title}
텍스트:
{text[:2000]}

다음 JSON 배열 형식으로만 반환하세요 (설명 없이):
[
  {{"param": "파라미터명", "value": "수치", "unit": "단위", "condition": "측정조건", "context": "원문30자"}},
  ...
]

추출 대상: VDD, Vth, Vmin, SNM, RSNM, WSNM, Ion, Ioff, Cell Ratio, Fin Width, Gate Length, Metal Pitch, Node, Yield, Sigma, Temperature 등"""

            try:
                resp_text = self._call_llm(prompt, max_tokens=600)
                match = re.search(r'\[.*?\]', resp_text, re.DOTALL)
                if match:
                    params = json.loads(match.group())
                    for p in params:
                        p["doi"] = doi
                        p["paper_title"] = title[:60]
                        all_params.append(p)
            except Exception as e:
                print(f"[DeepInsight] LLM 파라미터 추출 오류: {e}")

        return all_params

    # ══════════════════════════════════════════════════════════════════════════
    # 2. 실험 조건 매칭 엔진
    # ══════════════════════════════════════════════════════════════════════════
    def match_conditions(
        self,
        user_conditions: dict,          # {"technology": "FinFET", "node_nm": 7, ...}
        chunks: list[str],
        metas: list[dict],
    ) -> list[dict]:
        """
        사용자 공정 조건과 가장 유사한 논문 섹션을 점수순으로 반환
        """
        scored = []
        condition_keywords = self._conditions_to_keywords(user_conditions)

        for chunk, meta in zip(chunks, metas):
            score = 0
            matched_terms = []
            chunk_lower = chunk.lower()

            for kw in condition_keywords:
                if kw.lower() in chunk_lower:
                    score += 1
                    matched_terms.append(kw)

            # 노드 숫자 근접도 점수
            node = user_conditions.get("node_nm")
            if node:
                node_matches = re.findall(r'(\d+)\s*nm', chunk)
                for nm in node_matches:
                    diff = abs(int(nm) - int(node))
                    if diff == 0:
                        score += 3
                    elif diff <= 3:
                        score += 2
                    elif diff <= 10:
                        score += 1

            if score > 0:
                scored.append({
                    "score": score,
                    "matched_terms": matched_terms,
                    "doi": meta.get("doi",""),
                    "title": meta.get("title",""),
                    "year": meta.get("year",""),
                    "chunk": chunk[:500],
                })

        scored.sort(key=lambda x: -x["score"])
        return scored[:10]

    def _conditions_to_keywords(self, conditions: dict) -> list[str]:
        keywords = []
        tech = conditions.get("technology","")
        if tech:
            keywords.append(tech)
            keywords.extend(TECH_PARAM_MAP.get(tech, []))
        if conditions.get("node_nm"):
            keywords.append(f"{conditions['node_nm']}nm")
        if conditions.get("metal_pitch"):
            keywords.append(f"metal pitch")
            keywords.append(str(conditions["metal_pitch"]))
        keywords.extend(conditions.get("extra_keywords", []))
        return list(set(keywords))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. 논문 비교 뷰어
    # ══════════════════════════════════════════════════════════════════════════
    def compare_papers(
        self,
        papers: list[dict],             # [{"doi", "title", "chunks"}]
        aspects: list[str] = None,
    ) -> dict:
        """
        2~3편의 논문을 기술적 측면에서 비교 분석
        Returns: { "table": [...], "summary": str, "winner_by_aspect": {...} }
        """
        aspects = aspects or [
            "주요 기여/novelty",
            "사용 기술/공정 (FinFET/GAA/플래너)",
            "노드 (nm)",
            "핵심 파라미터 (Vmin/SNM/Yield)",
            "실험 방법론",
            "주요 결과",
            "한계점/제약",
            "향후 과제",
        ]

        papers_context = ""
        for i, p in enumerate(papers[:3]):
            chunks_text = "\n".join(p.get("chunks",[])[:3])[:1500]
            papers_context += f"\n\n=== 논문 {i+1}: {p['title']} ===\n{chunks_text}"

        aspects_str = "\n".join(f"- {a}" for a in aspects)
        prompt = f"""다음 {len(papers)}편의 반도체 논문을 비교 분석하세요.

{papers_context}

다음 측면에서 각 논문을 비교하여 JSON으로 반환하세요:
{aspects_str}

반환 형식 (JSON만, 설명 없이):
{{
  "table": [
    {{
      "aspect": "측면명",
      "papers": ["논문1 내용", "논문2 내용", "논문3 내용"]
    }}
  ],
  "key_differences": "핵심 차이점 요약 (한국어, 200자)",
  "recommendation": "어떤 상황에 어떤 논문이 더 유용한지 (한국어, 150자)"
}}"""

        try:
            resp = self._call_llm(prompt, max_tokens=1500)
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                result = json.loads(match.group())
                result["paper_titles"] = [p["title"] for p in papers[:3]]
                return result
        except Exception as e:
            print(f"[Compare] 오류: {e}")

        return {"table": [], "paper_titles": [], "key_differences": "분석 실패", "recommendation": ""}

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Debate Mode
    # ══════════════════════════════════════════════════════════════════════════
    def run_debate(
        self,
        topic: str,
        pro_chunks: list[str],          # 찬성 측 논문 청크
        pro_title: str,
        con_chunks: list[str],          # 반대 측 논문 청크
        con_title: str,
        user_context: str = "",         # 사내 공정 환경
    ) -> dict:
        """
        특정 기술 주제에 대해 상반된 두 논문의 토론 진행
        + 공정 엔지니어 관점에서 Hidden Assumption 지적
        """
        pro_text = "\n".join(pro_chunks[:3])[:1500]
        con_text = "\n".join(con_chunks[:3])[:1500]

        prompt = f"""당신은 20년 경력의 반도체 공정 엔지니어이자 논문 비평가입니다.

토론 주제: {topic}

[찬성 측 논문: {pro_title}]
{pro_text}

[반대/비판 측 논문: {con_title}]
{con_text}

{f"[현재 사내 공정 환경] {user_context}" if user_context else ""}

다음 형식으로 JSON만 반환하세요:
{{
  "topic_summary": "토론 주제 한 줄 요약",
  "pro_argument": {{
    "paper": "{pro_title[:50]}",
    "main_claim": "주요 주장 (100자)",
    "key_evidence": ["근거1", "근거2", "근거3"],
    "strength": "강점 (80자)",
    "weakness": "약점 (80자)"
  }},
  "con_argument": {{
    "paper": "{con_title[:50]}",
    "main_claim": "주요 주장 (100자)",
    "key_evidence": ["근거1", "근거2", "근거3"],
    "strength": "강점 (80자)",
    "weakness": "약점 (80자)"
  }},
  "hidden_assumptions": [
    "저자들이 암묵적으로 가정한 것 1",
    "저자들이 암묵적으로 가정한 것 2",
    "저자들이 암묵적으로 가정한 것 3"
  ],
  "engineer_verdict": "공정 엔지니어 관점 최종 의견 — 실제 양산 환경에서의 적용 가능성 평가 (200자)",
  "practical_risk": ["현장 적용 시 리스크1", "리스크2"]
}}"""

        try:
            resp = self._call_llm(prompt, max_tokens=1800)
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"[Debate] 오류: {e}")

        return {"topic_summary": topic, "pro_argument": {}, "con_argument": {},
                "hidden_assumptions": [], "engineer_verdict": "분석 실패", "practical_risk": []}

    # ══════════════════════════════════════════════════════════════════════════
    # 5. 증상 기반 역추적 (Symptom-to-Theory)
    # ══════════════════════════════════════════════════════════════════════════
    def symptom_to_theory(
        self,
        symptom: str,                   # "특정 코너에서 SNM 급격히 저하"
        all_chunks: list[str],
        all_metas: list[dict],
        embedder,                        # SentenceTransformer 인스턴스
        collection,                      # ChromaDB collection
    ) -> dict:
        """
        증상 설명 → 물리 메커니즘 추론 → 관련 논문 역추적
        """
        # Step 1: 증상에서 물리 메커니즘 추론
        mechanism_prompt = f"""반도체 SRAM 공정 전문가로서 다음 증상의 물리적 메커니즘을 분석하세요.

증상: {symptom}

다음 JSON으로만 반환하세요:
{{
  "root_causes": [
    {{"mechanism": "메커니즘1", "physics": "물리적 설명 (60자)", "keywords": ["검색키워드1","검색키워드2","검색키워드3"]}},
    {{"mechanism": "메커니즘2", "physics": "물리적 설명 (60자)", "keywords": ["검색키워드1","검색키워드2"]}},
    {{"mechanism": "메커니즘3", "physics": "물리적 설명 (60자)", "keywords": ["검색키워드1","검색키워드2"]}}
  ],
  "differential_diagnosis": "감별 진단 요약 (100자)",
  "recommended_experiments": ["권장 실험1", "권장 실험2"]
}}"""

        mechanisms = {}
        try:
            resp = self._call_llm(mechanism_prompt, max_tokens=800)
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                mechanisms = json.loads(match.group())
        except Exception as e:
            print(f"[Symptom] 메커니즘 추론 오류: {e}")
            mechanisms = {"root_causes": [], "differential_diagnosis": symptom}

        # Step 2: 추론된 키워드로 RAG 검색
        all_keywords = []
        for rc in mechanisms.get("root_causes", []):
            all_keywords.extend(rc.get("keywords", []))

        search_query = symptom + " " + " ".join(all_keywords[:6])
        emb = embedder.encode([search_query]).tolist()

        n_docs = collection.count()
        res = collection.query(
            query_embeddings=emb,
            n_results=min(8, n_docs),
            include=["documents","metadatas"],
        )
        related_docs  = res.get("documents",[[]])[0]
        related_metas = res.get("metadatas",[[]])[0]

        # Step 3: 관련 논문 요약
        related_papers = []
        seen_dois = set()
        for doc, meta in zip(related_docs, related_metas):
            doi = meta.get("doi","")
            if doi not in seen_dois:
                seen_dois.add(doi)
                related_papers.append({
                    "doi": doi,
                    "title": meta.get("title",""),
                    "year": meta.get("year",""),
                    "relevant_chunk": doc[:300],
                })

        return {
            "symptom": symptom,
            "mechanisms": mechanisms,
            "related_papers": related_papers,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 6. 기술 계통도 데이터 생성
    # ══════════════════════════════════════════════════════════════════════════
    def build_evolution_tree(
        self,
        papers: dict,                   # doi → paper dict
        all_chunks: list[str],
        all_metas: list[dict],
    ) -> dict:
        """
        논문들의 기술 계보를 분석하여 계통도 데이터 생성
        Returns: { nodes: [...], edges: [...], clusters: [...] }
        """
        paper_summaries = []
        seen = set()
        for chunk, meta in zip(all_chunks, all_metas):
            doi = meta.get("doi","")
            if doi in seen:
                continue
            seen.add(doi)
            paper = papers.get(doi, {})
            paper_summaries.append({
                "doi": doi,
                "title": meta.get("title","")[:80],
                "year": meta.get("year",""),
                "authors": ", ".join((paper.get("authors") or [])[:2]),
                "snippet": chunk[:400],
            })

        if not paper_summaries:
            return {"nodes": [], "edges": [], "clusters": []}

        summaries_text = "\n".join(
            f"[{p['year']}] {p['title']} ({p['authors']})\n{p['snippet']}"
            for p in paper_summaries[:12]
        )

        prompt = f"""반도체 SRAM/공정 논문들의 기술 계통도를 분석하세요.

논문 목록:
{summaries_text}

각 논문 간 기술적 계승 관계(A가 B의 기반/확장/개선/반박)를 분석하여 JSON으로 반환하세요:
{{
  "clusters": [
    {{"id": "c1", "name": "클러스터명", "description": "기술 흐름 설명", "color": "blue"}},
    ...
  ],
  "nodes": [
    {{
      "doi": "논문doi",
      "label": "짧은제목(25자)",
      "year": 연도숫자,
      "cluster": "c1",
      "importance": 1~5,
      "innovation": "핵심혁신 한줄(40자)"
    }},
    ...
  ],
  "edges": [
    {{"from": "doi1", "to": "doi2", "relation": "extends|improves|contradicts|applies", "description": "관계설명(30자)"}},
    ...
  ]
}}"""

        try:
            resp = self._call_llm(prompt, max_tokens=2000)
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                tree = json.loads(match.group())
                # 누락된 노드 보완
                existing_dois = {n["doi"] for n in tree.get("nodes",[])}
                for p in paper_summaries:
                    if p["doi"] not in existing_dois:
                        tree["nodes"].append({
                            "doi": p["doi"],
                            "label": p["title"][:25],
                            "year": p["year"],
                            "cluster": "c1",
                            "importance": 2,
                            "innovation": "",
                        })
                return tree
        except Exception as e:
            print(f"[EvolutionTree] 오류: {e}")

        # fallback: 연도 순서대로 단순 계통
        nodes = [{"doi": p["doi"], "label": p["title"][:25], "year": p["year"],
                  "cluster":"c1", "importance":2, "innovation":""} for p in paper_summaries]
        edges = []
        sorted_nodes = sorted(nodes, key=lambda x: x.get("year") or 0)
        for i in range(len(sorted_nodes)-1):
            edges.append({"from": sorted_nodes[i]["doi"],
                          "to": sorted_nodes[i+1]["doi"],
                          "relation": "extends", "description": "시간순"})
        return {"clusters": [{"id":"c1","name":"전체","description":"","color":"blue"}],
                "nodes": nodes, "edges": edges}

    # ── LLM 호출 ──────────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str, max_tokens: int = 800) -> str:
        if self.backend == "ollama":
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens, "temperature": 0.2}},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("response","")
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model,
                      "messages":[{"role":"user","content":prompt}],
                      "max_tokens": max_tokens, "temperature": 0.2},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
