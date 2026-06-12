"""
논문 메타데이터 수집 — OpenAlex API (완전 무료, 인증 불필요)

OpenAlex 주요 특징:
  - API 키 불필요 (polite pool: email 파라미터 권장)
  - Works 엔드포인트로 단일 논문 조회
  - referenced_works / related_works 필드로 인용 네트워크 수집
  - cited_by_api_url 로 피인용 논문 페이지네이션
  - open_access.oa_url 로 무료 PDF URL 직접 제공

API 구조:
  GET https://api.openalex.org/works/https://doi.org/{doi}
  GET https://api.openalex.org/works?filter=cites:{openalex_id}&per-page=50
"""
import time
import requests
from typing import Optional

OA_BASE = "https://api.openalex.org"

# 단일 논문 select 필드 (응답 크기 최소화)
WORK_SELECT = (
    "id,doi,title,abstract_inverted_index,authorships,"
    "publication_year,cited_by_count,referenced_works,"
    "open_access,primary_location,concepts,cited_by_api_url"
)

# 목록 조회용 (referenced_works 배치 조회)
LIST_SELECT = (
    "id,doi,title,authorships,publication_year,"
    "cited_by_count,open_access,primary_location"
)


class PaperFetcher:
    def __init__(self, email: str = "researcher@example.com", rate_limit_delay: float = 0.3):
        """
        email: OpenAlex polite pool 사용을 위해 권장 (속도 향상)
               실제 인증은 필요 없음
        """
        self.email = email
        self.delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"PaperExplorer/2.0 (mailto:{email})"
        })

    # ── DOI → 단일 논문 조회 ─────────────────────────────────────────────────
    def fetch_paper(self, doi: str) -> Optional[dict]:
        doi = self._clean_doi(doi)
        url = f"{OA_BASE}/works/https://doi.org/{doi}"
        resp = self._request(url, params={"select": WORK_SELECT, "mailto": self.email})
        if resp is None or resp.status_code != 200:
            # fallback: filter 검색
            return self._search_by_doi(doi)
        return self._normalize(resp.json())

    # ── OpenAlex ID → 단일 논문 조회 ─────────────────────────────────────────
    def fetch_by_oaid(self, oaid: str) -> Optional[dict]:
        """oaid 예: W2741809807"""
        if not oaid:
            return None
        url = f"{OA_BASE}/works/{oaid}"
        resp = self._request(url, params={"select": WORK_SELECT, "mailto": self.email})
        if resp is None or resp.status_code != 200:
            return None
        return self._normalize(resp.json())

    # ── 선행연구 (references) 수집 ───────────────────────────────────────────
    def fetch_references(self, oaid: str, limit: int = 50) -> list[dict]:
        """
        이 논문이 인용한 논문들 (referenced_works).
        OpenAlex는 referenced_works를 ID 배열로 제공 → 배치 조회
        """
        if not oaid:
            return []

        # 1) 루트 논문의 referenced_works ID 목록 가져오기
        url = f"{OA_BASE}/works/{oaid}"
        resp = self._request(url, params={
            "select": "referenced_works",
            "mailto": self.email
        })
        if resp is None or resp.status_code != 200:
            return []

        ref_ids: list[str] = resp.json().get("referenced_works", []) or []
        if not ref_ids:
            return []

        # 2) 상위 limit개만 배치 조회
        ref_ids = ref_ids[:limit]
        return self._fetch_works_batch(ref_ids)

    # ── 후속연구 (citations) 수집 ────────────────────────────────────────────
    def fetch_citations(self, oaid: str, limit: int = 50) -> list[dict]:
        """
        이 논문을 인용한 논문들.
        cited_by_api_url 엔드포인트로 페이지네이션 수집
        """
        if not oaid:
            return []

        results = []
        url = f"{OA_BASE}/works"
        params = {
            "filter": f"cites:{oaid}",
            "select": LIST_SELECT,
            "per-page": min(limit, 200),
            "sort": "cited_by_count:desc",   # 영향력 높은 순
            "mailto": self.email,
        }

        resp = self._request(url, params=params)
        if resp is None or resp.status_code != 200:
            return []

        data = resp.json()
        for raw in (data.get("results") or []):
            normalized = self._normalize_list_item(raw)
            if normalized:
                results.append(normalized)
            if len(results) >= limit:
                break

        return results

    # ── 배치 조회 (ID 리스트 → works) ────────────────────────────────────────
    def _fetch_works_batch(self, oaids: list[str]) -> list[dict]:
        """OpenAlex ID 목록을 pipe 필터로 한 번에 조회"""
        results = []
        batch_size = 50   # OpenAlex filter OR 최대 50개

        for i in range(0, len(oaids), batch_size):
            batch = oaids[i : i + batch_size]
            # ID 목록을 | 로 연결
            id_filter = "|".join(batch)
            params = {
                "filter": f"openalex_id:{id_filter}",
                "select": LIST_SELECT,
                "per-page": len(batch),
                "mailto": self.email,
            }
            resp = self._request(f"{OA_BASE}/works", params=params)
            if resp and resp.status_code == 200:
                for raw in (resp.json().get("results") or []):
                    item = self._normalize_list_item(raw)
                    if item:
                        results.append(item)

        return results

    # ── DOI 검색 fallback ─────────────────────────────────────────────────────
    def _search_by_doi(self, doi: str) -> Optional[dict]:
        params = {
            "filter": f"doi:{doi}",
            "select": WORK_SELECT,
            "mailto": self.email,
        }
        resp = self._request(f"{OA_BASE}/works", params=params)
        if resp is None or resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        return self._normalize(results[0])

    # ── 정규화: 단일 Work (전체 필드) ────────────────────────────────────────
    def _normalize(self, raw: dict) -> Optional[dict]:
        if not raw or not raw.get("id"):
            return None

        oaid  = raw["id"].replace("https://openalex.org/", "")  # W2741809807
        doi   = (raw.get("doi") or "").replace("https://doi.org/", "")
        if not doi:
            doi = f"oa:{oaid}"

        # 저자
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (raw.get("authorships") or [])
        ]
        authors = [a for a in authors if a]

        # 초록: inverted_index → 평문 복원
        abstract = self._reconstruct_abstract(
            raw.get("abstract_inverted_index") or {}
        )

        # 오픈액세스 PDF URL
        oa      = raw.get("open_access") or {}
        oa_url  = oa.get("oa_url") or ""
        # primary_location best_oa_location 보완
        pl = raw.get("primary_location") or {}
        if not oa_url:
            oa_url = (pl.get("pdf_url") or "")

        # 저널/컨퍼런스
        source = (pl.get("source") or {})
        venue  = source.get("display_name") or ""

        return {
            "doi":             doi,
            "oaid":            oaid,
            "title":           raw.get("title") or "(제목 없음)",
            "abstract":        abstract,
            "authors":         authors,
            "year":            raw.get("publication_year"),
            "citation_count":  raw.get("cited_by_count", 0) or 0,
            "reference_count": len(raw.get("referenced_works") or []),
            "venue":           venue,
            "oa_pdf_url":      oa_url,
            "references":      [],
            "citations":       [],
        }

    # ── 정규화: 목록 조회 결과 (간소 필드) ───────────────────────────────────
    def _normalize_list_item(self, raw: dict) -> Optional[dict]:
        if not raw or not raw.get("id"):
            return None

        oaid = raw["id"].replace("https://openalex.org/", "")
        doi  = (raw.get("doi") or "").replace("https://doi.org/", "")
        if not doi:
            doi = f"oa:{oaid}"

        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (raw.get("authorships") or [])
        ]
        authors = [a for a in authors if a]

        oa     = raw.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        pl     = raw.get("primary_location") or {}
        if not oa_url:
            oa_url = (pl.get("pdf_url") or "")

        return {
            "doi":            doi,
            "oaid":           oaid,
            "title":          raw.get("title") or "(제목 없음)",
            "abstract":       "",
            "authors":        authors,
            "year":           raw.get("publication_year"),
            "citation_count": raw.get("cited_by_count", 0) or 0,
            "reference_count": 0,
            "venue":          "",
            "oa_pdf_url":     oa_url,
            "references":     [],
            "citations":      [],
        }

    # ── 초록 복원 (inverted index → 평문) ────────────────────────────────────
    @staticmethod
    def _reconstruct_abstract(inv_index: dict) -> str:
        """
        OpenAlex abstract_inverted_index 형식:
        {"word": [pos1, pos2, ...], ...}
        """
        if not inv_index:
            return ""
        pos_word: dict[int, str] = {}
        for word, positions in inv_index.items():
            for pos in positions:
                pos_word[pos] = word
        if not pos_word:
            return ""
        return " ".join(pos_word[i] for i in sorted(pos_word))

    # ── DOI 정규화 ────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_doi(doi: str) -> str:
        return (doi.strip()
                .removeprefix("https://doi.org/")
                .removeprefix("http://doi.org/")
                .removeprefix("doi.org/"))

    # ── HTTP 요청 (rate-limit 자동 재시도) ───────────────────────────────────
    def _request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"[OpenAlex] Rate limit → {wait}초 대기")
                time.sleep(wait)
                resp = self.session.get(url, params=params, timeout=20)
            return resp
        except requests.exceptions.Timeout:
            print(f"[OpenAlex] Timeout: {url}")
            return None
        except Exception as e:
            print(f"[OpenAlex] 요청 오류: {e}")
            return None
        finally:
            time.sleep(self.delay)
