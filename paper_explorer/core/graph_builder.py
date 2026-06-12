"""
인용 네트워크 그래프 구축 — BFS
OpenAlex 기반: oaid(W-ID)를 primary key로 사용
"""
from collections import deque


class PaperGraphBuilder:
    def __init__(self):
        self.papers: dict[str, dict] = {}   # doi → paper
        self.edges:  list[tuple]     = []   # (src_doi, dst_doi, edge_type)

    def build_graph(
        self,
        root_doi: str,
        fetcher,
        depth: int = 2,
        max_papers: int = 80,
        direction: str = "references",
    ) -> dict:
        self.papers = {}
        self.edges  = []
        visited_doi  = set()
        visited_oaid = set()
        queue = deque()   # (doi, oaid, depth)

        # ── 루트 논문 ────────────────────────────────────────────────────────
        root = fetcher.fetch_paper(root_doi)
        if root is None:
            return {"nodes": [], "edges": [], "error": "DOI를 찾을 수 없습니다. OpenAlex에 등록되지 않은 논문일 수 있습니다."}

        root["is_root"] = True
        self._register(root, visited_doi, visited_oaid)
        queue.append((root["doi"], root.get("oaid", ""), 0))

        # ── BFS ──────────────────────────────────────────────────────────────
        while queue and len(self.papers) < max_papers:
            cur_doi, cur_oaid, cur_depth = queue.popleft()

            if cur_depth >= depth:
                continue
            if not cur_oaid:
                continue   # oaid 없으면 related 조회 불가

            remaining = max_papers - len(self.papers) + 5

            # ── references (선행연구) ─────────────────────────────────────────
            if direction in ("references", "both"):
                refs = fetcher.fetch_references(cur_oaid, limit=min(50, remaining))
                for nb in refs:
                    if len(self.papers) >= max_papers:
                        break
                    existing = self._find_existing(nb, visited_doi, visited_oaid)
                    if existing:
                        self.edges.append((cur_doi, existing, "reference"))
                        continue
                    # 상세 정보 fetch
                    full = fetcher.fetch_by_oaid(nb["oaid"]) if nb.get("oaid") else None
                    if full is None:
                        full = {**nb, "abstract": "", "venue": ""}
                    self._register(full, visited_doi, visited_oaid)
                    self.edges.append((cur_doi, full["doi"], "reference"))
                    queue.append((full["doi"], full.get("oaid", ""), cur_depth + 1))

            # ── citations (후속연구) ──────────────────────────────────────────
            if direction in ("citations", "both"):
                cites = fetcher.fetch_citations(cur_oaid, limit=min(50, remaining))
                for nb in cites:
                    if len(self.papers) >= max_papers:
                        break
                    existing = self._find_existing(nb, visited_doi, visited_oaid)
                    if existing:
                        self.edges.append((existing, cur_doi, "citation"))
                        continue
                    full = fetcher.fetch_by_oaid(nb["oaid"]) if nb.get("oaid") else None
                    if full is None:
                        full = {**nb, "abstract": "", "venue": ""}
                    self._register(full, visited_doi, visited_oaid)
                    self.edges.append((full["doi"], cur_doi, "citation"))
                    queue.append((full["doi"], full.get("oaid", ""), cur_depth + 1))

        # ── 직렬화 ───────────────────────────────────────────────────────────
        nodes = [
            {
                "id":             doi,
                "label":          self._short(p.get("title", "?")),
                "title":          p.get("title", ""),
                "year":           p.get("year"),
                "citation_count": p.get("citation_count", 0),
                "authors":        p.get("authors", []),
                "is_root":        p.get("is_root", False),
            }
            for doi, p in self.papers.items()
        ]
        edges_data = [
            {"from": src, "to": dst, "type": etype}
            for src, dst, etype in self.edges
            if src in self.papers and dst in self.papers
        ]
        return {"nodes": nodes, "edges": edges_data}

    # ── 헬퍼 ─────────────────────────────────────────────────────────────────
    def _register(self, paper: dict, visited_doi: set, visited_oaid: set):
        doi  = paper.get("doi", "")
        oaid = paper.get("oaid", "")
        if doi:
            visited_doi.add(doi)
            self.papers[doi] = paper
        if oaid:
            visited_oaid.add(oaid)

    def _find_existing(self, nb: dict, visited_doi: set, visited_oaid: set) -> Optional[str]:
        """이미 방문한 노드면 해당 doi 반환, 아니면 None"""
        ndoi  = nb.get("doi", "")
        noaid = nb.get("oaid", "")
        if ndoi and ndoi in visited_doi:
            return ndoi
        if noaid and noaid in visited_oaid:
            # oaid로 doi 역조회
            for doi, p in self.papers.items():
                if p.get("oaid") == noaid:
                    return doi
        return None

    def _short(self, title: str, n: int = 38) -> str:
        return title if len(title) <= n else title[:n - 3] + "..."


# Optional import (타입 힌트용, 런타임 불필요)
from typing import Optional
