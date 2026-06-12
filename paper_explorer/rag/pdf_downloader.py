"""
오픈액세스 PDF 다운로더 — OpenAlex 기반
우선순위:
  1) OpenAlex oa_pdf_url (paper_fetcher가 이미 수집)
  2) OpenAlex API 재조회 (best_oa_location)
  3) Unpaywall API
  4) arXiv 휴리스틱
"""
import time
import requests
from pathlib import Path
from typing import Optional

OA_BASE        = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


class PDFDownloader:
    def __init__(self, save_dir: Path, email: str = "researcher@example.com"):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.email   = email
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"Mozilla/5.0 PaperExplorer/2.0 (mailto:{email})"
        })
        self.session.max_redirects = 10

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────
    def download(self, paper: dict) -> str:
        """Returns: 'downloaded' | 'already' | 'failed' | 'skipped'"""
        doi = paper.get("doi", "")
        if not doi or doi.startswith("oa:"):
            return "skipped"

        path = self._doi_to_path(doi)
        if path.exists() and path.stat().st_size > 1000:
            return "already"

        for url in self._collect_urls(doi, paper):
            if url and self._try_download(url, path):
                return "downloaded"

        return "failed"

    def get_pdf_path(self, doi: str) -> Optional[Path]:
        p = self._doi_to_path(doi)
        return p if (p.exists() and p.stat().st_size > 1000) else None

    def list_downloaded(self) -> list[Path]:
        return [p for p in self.save_dir.glob("*.pdf") if p.stat().st_size > 1000]

    # ── URL 수집 (우선순위 순) ────────────────────────────────────────────────
    def _collect_urls(self, doi: str, paper: dict) -> list[str]:
        urls: list[str] = []

        # 1) paper_fetcher가 이미 수집한 OA URL
        if paper.get("oa_pdf_url"):
            urls.append(paper["oa_pdf_url"])

        # 2) OpenAlex API 재조회 (best_oa_location.pdf_url)
        oa_urls = self._openalex_pdf_urls(doi, paper.get("oaid", ""))
        for u in oa_urls:
            if u and u not in urls:
                urls.append(u)

        # 3) Unpaywall
        time.sleep(0.3)
        for u in self._unpaywall_urls(doi):
            if u and u not in urls:
                urls.append(u)

        # 4) arXiv 휴리스틱
        ax = self._arxiv_url(doi)
        if ax and ax not in urls:
            urls.append(ax)

        return [u for u in urls if u]

    # ── OpenAlex PDF URL 조회 ────────────────────────────────────────────────
    def _openalex_pdf_urls(self, doi: str, oaid: str = "") -> list[str]:
        urls = []
        try:
            # oaid가 있으면 직접 조회, 없으면 DOI 조회
            if oaid:
                api_url = f"{OA_BASE}/works/{oaid}"
            else:
                api_url = f"{OA_BASE}/works/https://doi.org/{doi}"

            resp = self.session.get(
                api_url,
                params={
                    "select": "open_access,primary_location,locations",
                    "mailto": self.email,
                },
                timeout=(8, 20),
            )
            if resp.status_code != 200:
                return urls

            data = resp.json()

            # open_access.oa_url
            oa = data.get("open_access") or {}
            if oa.get("oa_url"):
                urls.append(oa["oa_url"])

            # primary_location.pdf_url
            pl = data.get("primary_location") or {}
            if pl.get("pdf_url"):
                u = pl["pdf_url"]
                if u not in urls:
                    urls.append(u)

            # locations 전체 순회
            for loc in (data.get("locations") or []):
                u = loc.get("pdf_url") or ""
                if u and u not in urls:
                    urls.append(u)

        except Exception as e:
            print(f"[OpenAlex PDF] {doi[:30]}: {e}")
        finally:
            time.sleep(0.2)
        return urls

    # ── Unpaywall ─────────────────────────────────────────────────────────────
    def _unpaywall_urls(self, doi: str) -> list[str]:
        urls = []
        try:
            resp = self.session.get(
                f"{UNPAYWALL_BASE}/{doi}",
                params={"email": self.email},
                timeout=(8, 20),
            )
            if resp.status_code != 200:
                return urls
            data = resp.json()

            best = data.get("best_oa_location") or {}
            u = best.get("url_for_pdf") or best.get("url", "")
            if u:
                urls.append(u)

            for loc in (data.get("oa_locations") or []):
                u = loc.get("url_for_pdf") or loc.get("url", "")
                if u and u not in urls:
                    urls.append(u)
        except Exception as e:
            print(f"[Unpaywall] {doi[:30]}: {e}")
        return urls

    # ── arXiv 휴리스틱 ────────────────────────────────────────────────────────
    def _arxiv_url(self, doi: str) -> Optional[str]:
        if "arxiv" in doi.lower():
            parts = doi.split("arXiv.")
            if len(parts) == 2:
                return f"https://arxiv.org/pdf/{parts[1].strip()}"
        return None

    # ── 실제 다운로드 ─────────────────────────────────────────────────────────
    def _try_download(self, url: str, path: Path) -> bool:
        try:
            resp = self.session.get(
                url, timeout=(10, 60), stream=True, allow_redirects=True
            )
            if resp.status_code != 200:
                return False

            ct = resp.headers.get("content-type", "").lower()
            if any(b in ct for b in ("text/html", "application/json", "text/xml")):
                return False

            data = b""
            for chunk in resp.iter_content(16384):
                data += chunk
                if len(data) > 50 * 1024 * 1024:
                    break

            if len(data) < 2000:
                return False
            if not data[:5].startswith(b"%PDF-"):
                return False

            path.write_bytes(data)
            print(f"[PDF] ✅ {path.name} ({len(data)//1024}KB)")
            return True

        except Exception as e:
            print(f"[PDF] ✗ {url[:60]}: {e}")
            return False

    # ── DOI → 저장 경로 ──────────────────────────────────────────────────────
    def _doi_to_path(self, doi: str) -> Path:
        safe = doi.replace("/", "__").replace(":", "_").replace(" ", "_")
        return self.save_dir / f"{safe}.pdf"
