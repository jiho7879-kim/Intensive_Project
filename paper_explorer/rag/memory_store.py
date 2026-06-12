"""
영구 기억 저장소 — SQLite 기반
쓸수록 누적되는 4가지 데이터:
  1. user_notes     : 사용자가 직접 작성한 메모 (논문별)
  2. query_history  : 과거 질의 + 응답 + 피드백
  3. insights       : LLM이 생성한 인사이트 중 저장된 것
  4. paper_tags     : 논문에 붙인 사용자 태그/메모
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

DB_PATH = Path("./rag_memory.db")


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                question  TEXT NOT NULL,
                answer    TEXT NOT NULL,
                context_dois TEXT,          -- JSON 배열: 검색된 논문 DOI
                feedback  INTEGER DEFAULT 0, -- +1: 좋아요, -1: 싫어요, 0: 없음
                timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_notes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doi       TEXT NOT NULL,
                note      TEXT NOT NULL,
                tags      TEXT DEFAULT '[]', -- JSON 배열
                timestamp REAL NOT NULL,
                UNIQUE(doi, note)
            );
            CREATE TABLE IF NOT EXISTS insights (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                title     TEXT NOT NULL,
                content   TEXT NOT NULL,
                source_dois TEXT DEFAULT '[]',
                pinned    INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_tags (
                doi       TEXT NOT NULL,
                tag       TEXT NOT NULL,
                PRIMARY KEY (doi, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_qh_ts   ON query_history(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_notes_doi ON user_notes(doi);
            """)

    # ── Query History ─────────────────────────────────────────────────────────
    def save_query(self, question: str, answer: str, context_dois: list[str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO query_history (question, answer, context_dois, timestamp) VALUES (?,?,?,?)",
                (question, answer, json.dumps(context_dois), time.time())
            )
            return cur.lastrowid

    def set_feedback(self, query_id: int, feedback: int):
        """feedback: +1 또는 -1"""
        with self._conn() as conn:
            conn.execute("UPDATE query_history SET feedback=? WHERE id=?", (feedback, query_id))

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def search_similar_queries(self, question: str, limit: int = 5) -> list[dict]:
        """간단한 키워드 기반 유사 질문 검색"""
        keywords = [w for w in question.split() if len(w) > 2]
        if not keywords:
            return []
        conditions = " OR ".join(["question LIKE ?" for _ in keywords])
        params = [f"%{k}%" for k in keywords] + [limit]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM query_history WHERE ({conditions}) AND feedback >= 0 ORDER BY feedback DESC, timestamp DESC LIMIT ?",
                params
            ).fetchall()
        return [dict(r) for r in rows]

    def get_positive_examples(self, limit: int = 10) -> list[dict]:
        """👍 피드백 받은 좋은 답변들"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM query_history WHERE feedback > 0 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── User Notes ────────────────────────────────────────────────────────────
    def save_note(self, doi: str, note: str, tags: list[str] = None) -> int:
        tags = tags or []
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO user_notes (doi, note, tags, timestamp) VALUES (?,?,?,?)",
                    (doi, note, json.dumps(tags), time.time())
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return -1

    def get_notes_for_doi(self, doi: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM user_notes WHERE doi=? ORDER BY timestamp DESC", (doi,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_notes(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM user_notes ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM user_notes WHERE id=?", (note_id,))

    # ── Insights ──────────────────────────────────────────────────────────────
    def save_insight(self, title: str, content: str, source_dois: list[str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO insights (title, content, source_dois, timestamp) VALUES (?,?,?,?)",
                (title, content, json.dumps(source_dois), time.time())
            )
            return cur.lastrowid

    def toggle_pin_insight(self, insight_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE insights SET pinned = 1 - pinned WHERE id=?", (insight_id,)
            )

    def delete_insight(self, insight_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM insights WHERE id=?", (insight_id,))

    def get_insights(self, pinned_only: bool = False) -> list[dict]:
        where = "WHERE pinned=1" if pinned_only else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM insights {where} ORDER BY pinned DESC, timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Paper Tags ────────────────────────────────────────────────────────────
    def add_tag(self, doi: str, tag: str):
        with self._conn() as conn:
            try:
                conn.execute("INSERT INTO paper_tags (doi, tag) VALUES (?,?)", (doi, tag))
            except sqlite3.IntegrityError:
                pass

    def remove_tag(self, doi: str, tag: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM paper_tags WHERE doi=? AND tag=?", (doi, tag))

    def get_tags(self, doi: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT tag FROM paper_tags WHERE doi=?", (doi,)).fetchall()
        return [r["tag"] for r in rows]

    def get_all_tags(self) -> dict[str, list[str]]:
        """doi → [tags] 전체 맵"""
        with self._conn() as conn:
            rows = conn.execute("SELECT doi, tag FROM paper_tags").fetchall()
        result: dict[str, list[str]] = {}
        for r in rows:
            result.setdefault(r["doi"], []).append(r["tag"])
        return result

    # ── 통계 ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        with self._conn() as conn:
            n_queries  = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
            n_positive = conn.execute("SELECT COUNT(*) FROM query_history WHERE feedback>0").fetchone()[0]
            n_negative = conn.execute("SELECT COUNT(*) FROM query_history WHERE feedback<0").fetchone()[0]
            n_notes    = conn.execute("SELECT COUNT(*) FROM user_notes").fetchone()[0]
            n_insights = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
        return {
            "queries": n_queries,
            "positive": n_positive,
            "negative": n_negative,
            "notes": n_notes,
            "insights": n_insights,
        }

    def format_ts(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
