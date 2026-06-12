"""
Paper Explorer - 논문 인용 네트워크 탐색 및 RAG 기반 인사이트 도출 시스템
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import json

from core.graph_builder import PaperGraphBuilder
from core.paper_fetcher import PaperFetcher
from rag.pdf_downloader import PDFDownloader
from rag.rag_engine import RAGEngine
from rag.deep_insight import DeepInsightEngine
from ui.graph_renderer import render_graph_html
from ui.sidebar import render_sidebar

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }
  .main .block-container { padding-top: 1.5rem; max-width: 1600px; }

  /* 헤더 */
  .hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 70% 30%, rgba(99,102,241,0.15) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
  }
  .hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
  }

  /* DOI 입력 카드 */
  .input-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
  }

  /* 메트릭 */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
  .metric-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    text-align: center;
  }
  .metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

  /* 탭 */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #1e1b4b !important;
    color: #818cf8 !important;
  }

  /* 버튼 */
  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 0.6rem 1.6rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
  }

  /* 논문 카드 */
  .paper-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 3px solid #4f46e5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
  }
  .paper-card:hover { border-left-color: #818cf8; }
  .paper-title { font-weight: 600; color: #e2e8f0; font-size: 0.92rem; margin-bottom: 0.3rem; }
  .paper-meta { font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
  .paper-abstract { font-size: 0.83rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.6; }

  /* RAG 인사이트 */
  .insight-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .insight-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
  }

  /* 진행 상태 */
  .status-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .badge-success { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .badge-warn    { background: #422006; color: #fb923c; border: 1px solid #9a3412; }
  .badge-info    { background: #1e1b4b; color: #818cf8; border: 1px solid #3730a3; }
</style>
""", unsafe_allow_html=True)

# ── 세션 초기화 ───────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "graph_data": None,
        "papers": {},
        "selected_paper": None,
        "rag_ready": False,
        "chat_history": [],
        "download_status": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🔬 Paper Explorer</div>
  <div class="hero-sub">DOI 기반 논문 인용 네트워크 탐색 · PDF 자동 수집 · RAG 인사이트 도출</div>
</div>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
config = render_sidebar()

# ── 메인 레이아웃 ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1.9])

with col_left:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("#### 📄 논문 DOI 입력")
    doi_input = st.text_input(
        "DOI",
        placeholder="예: 10.1038/nature12373",
        label_visibility="collapsed",
        key="doi_input"
    )
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        explore_btn = st.button("🕸️ 네트워크 탐색", use_container_width=True)
    with col_b2:
        clear_btn = st.button("🗑️ 초기화", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clear_btn:
        for k in ["graph_data","papers","selected_paper","rag_ready","chat_history","download_status"]:
            st.session_state[k] = {} if k in ["papers","download_status"] else (
                [] if k == "chat_history" else (False if k == "rag_ready" else None)
            )
        st.rerun()

    # ── 그래프 탐색 실행 ──────────────────────────────────────────────────────
    if explore_btn and doi_input.strip():
        doi = doi_input.strip()
        fetcher = PaperFetcher(email=config.get("oa_email", "researcher@example.com"))
        builder = PaperGraphBuilder()

        with st.spinner("📡 루트 논문 메타데이터 수집 중..."):
            root_paper = fetcher.fetch_paper(doi)

        if root_paper is None:
            st.error("❌ DOI를 찾을 수 없습니다. 형식을 확인해주세요.")
        else:
            direction = config.get("direction", "references")
            with st.spinner(
                f"🕸️ 인용 네트워크 구축 중 — "
                f"방향={direction}, depth={config['depth']}, max={config['max_papers']} …"
            ):
                graph_data = builder.build_graph(
                    root_doi=doi,
                    fetcher=fetcher,
                    depth=config["depth"],
                    max_papers=config["max_papers"],
                    direction=direction,
                )
            err = graph_data.get("error")
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.graph_data = graph_data
                st.session_state.papers = builder.papers
                n_p = len(builder.papers)
                n_e = len(graph_data["edges"])
                st.success(f"✅ {n_p}편 논문, {n_e}개 연결 수집 완료")

    # ── 메트릭 ────────────────────────────────────────────────────────────────
    if st.session_state.papers:
        n_papers = len(st.session_state.papers)
        n_edges = len(st.session_state.graph_data.get("edges", []))
        n_dl = sum(1 for v in st.session_state.download_status.values() if v == "downloaded")
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">{n_papers}</div><div class="metric-label">논문 수</div></div>
          <div class="metric-box"><div class="metric-val">{n_edges}</div><div class="metric-label">인용 연결</div></div>
          <div class="metric-box"><div class="metric-val">{n_dl}</div><div class="metric-label">PDF 수집</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── 논문 목록 ─────────────────────────────────────────────────────────────
    if st.session_state.papers:
        st.markdown("#### 📚 수집된 논문 목록")
        search_q = st.text_input("🔍 논문 검색", placeholder="제목 / 저자 검색...", label_visibility="collapsed")

        papers_list = list(st.session_state.papers.values())
        if search_q:
            q = search_q.lower()
            papers_list = [p for p in papers_list if
                           q in p.get("title","").lower() or
                           any(q in a.lower() for a in p.get("authors",[]))]

        # 인용 수 내림차순
        papers_list.sort(key=lambda p: p.get("citation_count", 0), reverse=True)

        for p in papers_list[:50]:
            dl_status = st.session_state.download_status.get(p["doi"], "")
            badge = ""
            if dl_status == "downloaded":
                badge = '<span class="status-badge badge-success">PDF ✓</span>'
            elif dl_status == "failed":
                badge = '<span class="status-badge badge-warn">PDF ✗</span>'
            else:
                badge = '<span class="status-badge badge-info">미수집</span>'

            authors_str = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors_str += " et al."

            abstract_preview = (p.get("abstract","") or "")[:150]
            if len(p.get("abstract","") or "") > 150:
                abstract_preview += "..."

            st.markdown(f"""
            <div class="paper-card">
              <div class="paper-title">{p.get('title','(제목 없음)')}</div>
              <div class="paper-meta">{authors_str} · {p.get('year','?')} · 피인용 {p.get('citation_count',0):,}회 &nbsp; {badge}</div>
              {"<div class='paper-abstract'>" + abstract_preview + "</div>" if abstract_preview else ""}
            </div>
            """, unsafe_allow_html=True)

with col_right:
    tab_graph, tab_download, tab_rag, tab_deep = st.tabs([
        "🕸️ 인용 그래프", "⬇️ PDF 수집", "🤖 RAG 인사이트", "🔬 Deep Insight"
    ])

    # ── 탭 1: 그래프 ──────────────────────────────────────────────────────────
    with tab_graph:
        if st.session_state.graph_data:
            graph_html = render_graph_html(
                st.session_state.graph_data,
                st.session_state.papers,
                height=650,
            )
            components.html(graph_html, height=670, scrolling=False)
            st.caption("💡 노드를 클릭하면 논문 정보를 확인할 수 있습니다. 스크롤로 줌 조절 가능.")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 5rem 2rem; color: #374151;">
              <div style="font-size:4rem; margin-bottom:1rem;">🕸️</div>
              <div style="font-size:1.1rem; color:#6b7280;">왼쪽에서 DOI를 입력하고<br>네트워크 탐색을 시작하세요</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 탭 2: PDF 다운로드 ────────────────────────────────────────────────────
    with tab_download:
        if not st.session_state.papers:
            st.info("먼저 DOI를 입력해 네트워크를 탐색하세요.")
        else:
            st.markdown("#### ⬇️ 오픈액세스 PDF 자동 수집")
            st.caption("Unpaywall / OpenAlex API를 통해 오픈액세스 PDF를 수집합니다.")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                dl_btn = st.button("📥 전체 PDF 수집 시작", use_container_width=True)
            with col_d2:
                build_rag_btn = st.button("🧠 RAG 인덱스 구축", use_container_width=True)

            if dl_btn:
                downloader = PDFDownloader(
                    save_dir=Path(config["pdf_dir"]),
                    email=config.get("oa_email", "researcher@example.com"),
                )
                progress = st.progress(0)
                status_text = st.empty()
                papers = list(st.session_state.papers.values())
                for i, paper in enumerate(papers):
                    doi = paper.get("doi","")
                    if not doi:
                        continue
                    status_text.text(f"[{i+1}/{len(papers)}] {paper.get('title','')[:60]}...")
                    result = downloader.download(paper)
                    st.session_state.download_status[doi] = result
                    progress.progress((i+1) / len(papers))
                status_text.text("✅ PDF 수집 완료!")
                n_ok = sum(1 for v in st.session_state.download_status.values() if v == "downloaded")
                st.success(f"총 {n_ok}편 PDF 수집 성공")

            if build_rag_btn:
                n_dl = sum(1 for v in st.session_state.download_status.values() if v == "downloaded")
                if n_dl == 0:
                    st.warning("먼저 PDF를 수집해주세요.")
                else:
                    rag_opts = st.session_state.get("rag_options", {})
                    engine = RAGEngine(
                        pdf_dir=Path(config["pdf_dir"]),
                        backend=config["backend"],
                        model=config["model"],
                        api_key=config["api_key"],
                        base_url=config["base_url"],
                        use_hyde=rag_opts.get("use_hyde", True),
                        use_multiquery=rag_opts.get("use_multiquery", True),
                        use_reranker=rag_opts.get("use_reranker", True),
                    )
                    # 연결 사전 확인
                    ok, msg = engine.check_connection()
                    if not ok:
                        st.error(msg)
                    else:
                        st.info(msg)
                        with st.spinner(f"🔨 {n_dl}편 PDF 청킹 및 임베딩 중..."):
                            n_chunks = engine.build_index(st.session_state.papers)
                        st.session_state.rag_ready = True
                        st.session_state.rag_engine = engine
                        st.success(f"✅ RAG 인덱스 완료 — {n_chunks:,}개 청크 인덱싱")

            # 다운로드 상태 표
            if st.session_state.download_status:
                st.markdown("---")
                st.markdown("**수집 현황**")
                for doi, status in list(st.session_state.download_status.items())[:30]:
                    paper = st.session_state.papers.get(doi, {})
                    icon = "✅" if status == "downloaded" else ("❌" if status == "failed" else "⏭️")
                    st.markdown(f"{icon} `{doi}` — {paper.get('title','')[:60]}")

    # ── 탭 3: RAG 인사이트 ────────────────────────────────────────────────────
    with tab_rag:
        if not st.session_state.get("rag_ready"):
            st.info("PDF 수집 탭에서 RAG 인덱스를 먼저 구축해주세요.")
        else:
            engine: RAGEngine = st.session_state.rag_engine
            mem = engine.memory

            # ── 서브탭 ───────────────────────────────────────────────────────
            sub_chat, sub_memory, sub_history = st.tabs([
                "💬 질의응답", "📚 지식 관리", "📈 학습 이력"
            ])

            # ════════════════════════════════════════
            # 서브탭 1: 질의응답
            # ════════════════════════════════════════
            with sub_chat:
                # 프리셋 질문
                preset_col1, preset_col2 = st.columns(2)
                presets = [
                    ("📊 핵심 트렌드", "이 논문 그룹의 핵심 연구 트렌드와 방향성을 요약해줘"),
                    ("🔍 방법론 비교", "각 논문이 사용한 주요 방법론을 비교 분석해줘"),
                    ("💡 연구 공백", "이 분야에서 아직 해결되지 않은 연구 공백은 무엇인가?"),
                    ("🏆 핵심 기여", "가장 영향력 있는 논문들의 핵심 기여를 정리해줘"),
                    ("⚙️ 기술 비교", "각 논문의 기술적 접근법 차이를 비교 분석해줘"),
                    ("🔮 미래 방향", "이 연구 흐름이 향후 어떻게 발전할지 예측해줘"),
                ]
                for i, (label, q) in enumerate(presets):
                    col = preset_col1 if i % 2 == 0 else preset_col2
                    with col:
                        if st.button(label, use_container_width=True, key=f"preset_{i}"):
                            st.session_state.chat_history.append({"role":"user","content":q})
                            with st.spinner("🔍 하이브리드 검색 + 재순위화 중..."):
                                answer, qid = engine.query(q, st.session_state.chat_history[:-1])
                            st.session_state.chat_history.append({
                                "role":"assistant","content":answer,"query_id":qid
                            })
                            st.rerun()

                st.markdown("---")

                # 대화 렌더링
                for idx, msg in enumerate(st.session_state.chat_history):
                    if msg["role"] == "user":
                        st.markdown(f"**🙋 질문:** {msg['content']}")
                    else:
                        st.markdown(
                            f'<div class="insight-box"><div class="insight-label">🤖 AI 인사이트</div>{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                        # 피드백 + 저장 버튼
                        qid = msg.get("query_id", -1)
                        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1,1,2,2])
                        with fb_col1:
                            if st.button("👍", key=f"like_{idx}", help="좋은 답변"):
                                engine.give_feedback(qid, True)
                                st.toast("피드백 저장됨 — 다음 검색에 반영됩니다")
                        with fb_col2:
                            if st.button("👎", key=f"dislike_{idx}", help="개선 필요"):
                                engine.give_feedback(qid, False)
                                st.toast("피드백 저장됨")
                        with fb_col3:
                            save_title = st.text_input("인사이트 제목", key=f"ins_title_{idx}",
                                                        placeholder="저장할 제목 입력", label_visibility="collapsed")
                        with fb_col4:
                            if st.button("📌 인사이트 저장", key=f"save_ins_{idx}"):
                                if save_title.strip():
                                    dois = [m.get("doi","") for m in st.session_state.chat_history
                                            if m.get("role")=="user"]
                                    engine.save_insight(save_title, msg["content"], dois)
                                    st.toast(f"📌 '{save_title}' 저장됨")

                # 자유 질문 입력
                q_col, btn_col = st.columns([4, 1])
                with q_col:
                    user_q = st.text_input(
                        "질문", key="rag_query",
                        placeholder="예: Vmin과 공정 변동의 상관관계는?",
                        label_visibility="collapsed"
                    )
                with btn_col:
                    ask_btn = st.button("📨 질문", use_container_width=True)

                if ask_btn and user_q.strip():
                    st.session_state.chat_history.append({"role":"user","content":user_q})
                    with st.spinner("🔍 HyDE + 멀티쿼리 + 하이브리드 검색 중..."):
                        answer, qid = engine.query(user_q, st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append({
                        "role":"assistant","content":answer,"query_id":qid
                    })
                    st.rerun()

                if st.session_state.chat_history:
                    if st.button("🗑️ 대화 초기화"):
                        st.session_state.chat_history = []
                        st.rerun()

            # ════════════════════════════════════════
            # 서브탭 2: 지식 관리
            # ════════════════════════════════════════
            with sub_memory:
                m_col1, m_col2 = st.columns(2)

                # ── 인사이트 관리 ─────────────────────────────────────────────
                with m_col1:
                    st.markdown("#### 📌 저장된 인사이트")
                    insights = mem.get_insights()
                    if not insights:
                        st.caption("아직 저장된 인사이트가 없습니다. 질의응답 탭에서 답변을 저장하세요.")
                    for ins in insights:
                        pinned_icon = "📌" if ins["pinned"] else "📄"
                        with st.expander(f"{pinned_icon} {ins['title']} ({mem.format_ts(ins['timestamp'])})"):
                            st.markdown(ins["content"][:600] + ("..." if len(ins["content"]) > 600 else ""))
                            col_p, col_d = st.columns(2)
                            with col_p:
                                if st.button("📌 고정 토글", key=f"pin_{ins['id']}"):
                                    mem.toggle_pin_insight(ins["id"])
                                    st.rerun()
                            with col_d:
                                if st.button("🗑️ 삭제", key=f"del_ins_{ins['id']}"):
                                    mem.delete_insight(ins["id"])
                                    st.rerun()

                # ── 논문 메모 ─────────────────────────────────────────────────
                with m_col2:
                    st.markdown("#### 📝 논문 메모")

                    # 메모 추가
                    if st.session_state.papers:
                        doi_options = {
                            f"{p.get('title',doi)[:45]}": doi
                            for doi, p in st.session_state.papers.items()
                        }
                        selected_label = st.selectbox("논문 선택", list(doi_options.keys()),
                                                       label_visibility="collapsed")
                        selected_doi = doi_options.get(selected_label, "")
                        note_text = st.text_area("메모 내용", key="note_input",
                                                  placeholder="이 논문에 대한 메모를 입력하세요...",
                                                  label_visibility="collapsed", height=80)
                        tags_text = st.text_input("태그 (쉼표 구분)", key="tags_input",
                                                   placeholder="예: 핵심논문, Vmin, FinFET",
                                                   label_visibility="collapsed")
                        if st.button("💾 메모 저장", use_container_width=True):
                            if note_text.strip() and selected_doi:
                                tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                                mem.save_note(selected_doi, note_text, tags)
                                st.toast("메모 저장됨")
                                st.rerun()

                    # 저장된 메모 목록
                    all_notes = mem.get_all_notes()
                    if all_notes:
                        st.markdown("---")
                        for note in all_notes[:20]:
                            paper = st.session_state.papers.get(note["doi"], {})
                            title = paper.get("title", note["doi"])[:40]
                            tags = json.loads(note.get("tags","[]") or "[]")
                            tag_str = " ".join([f"`{t}`" for t in tags]) if tags else ""
                            with st.expander(f"📝 {title} — {mem.format_ts(note['timestamp'])}"):
                                st.markdown(note["note"])
                                if tag_str:
                                    st.markdown(tag_str)
                                if st.button("🗑️", key=f"del_note_{note['id']}"):
                                    mem.delete_note(note["id"])
                                    st.rerun()

            # ════════════════════════════════════════
            # 서브탭 3: 학습 이력
            # ════════════════════════════════════════
            with sub_history:
                stats = mem.get_stats()
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("총 질의", stats["queries"])
                s2.metric("👍 좋음", stats["positive"])
                s3.metric("👎 개선", stats["negative"])
                s4.metric("📝 메모", stats["notes"])
                s5.metric("📌 인사이트", stats["insights"])

                st.markdown("---")
                st.markdown("#### 📋 최근 질의 이력")
                recent = mem.get_recent_queries(limit=30)
                if not recent:
                    st.caption("아직 질의 이력이 없습니다.")
                for q in recent:
                    fb_icon = "👍" if q["feedback"] > 0 else ("👎" if q["feedback"] < 0 else "⬜")
                    with st.expander(f"{fb_icon} {q['question'][:70]} — {mem.format_ts(q['timestamp'])}"):
                        st.markdown("**질문:**")
                        st.markdown(q["question"])
                        st.markdown("**답변 (앞 500자):**")
                        st.markdown(q["answer"][:500] + "...")
                        # 재질문 버튼
                        if st.button("🔁 이 질문 다시하기", key=f"reask_{q['id']}"):
                            st.session_state.chat_history.append({"role":"user","content":q["question"]})
                            with st.spinner("재질의 중..."):
                                answer, qid = engine.query(q["question"], [])
                            st.session_state.chat_history.append({
                                "role":"assistant","content":answer,"query_id":qid
                            })
                            st.rerun()

    # ── 탭 4: Deep Insight ────────────────────────────────────────────────────
    with tab_deep:
        if not st.session_state.get("rag_ready"):
            st.info("RAG 인사이트 탭에서 인덱스를 먼저 구축해주세요.")
        else:
            engine: RAGEngine = st.session_state.rag_engine
            di = DeepInsightEngine(
                backend=engine.backend, model=engine.model,
                api_key=engine.api_key, base_url=engine.base_url,
            )
            all_chunks = engine._all_docs
            all_metas  = engine._all_metas

            deep_tabs = st.tabs([
                "⚗️ 파라미터 추출",
                "🔬 조건 매칭",
                "📊 논문 비교",
                "⚔️ Debate Mode",
                "🩺 증상 역추적",
                "🌳 기술 계통도",
            ])

            # ════════════════════════════════════════
            # ⚗️ 파라미터 자동 추출
            # ════════════════════════════════════════
            with deep_tabs[0]:
                st.markdown("#### ⚗️ 물리적 파라미터 자동 추출")
                st.caption("논문 본문에서 VDD, Vth, SNM, Cell Ratio 등 수치를 자동 구조화합니다.")

                col_r, col_l = st.columns([1, 3])
                with col_r:
                    use_llm_extract = st.toggle("LLM 심층 추출", value=False,
                                                 help="정규식 추출 후 LLM으로 추가 파라미터 발굴 (느림)")
                    extract_btn = st.button("⚗️ 파라미터 추출 시작", use_container_width=True)

                if extract_btn:
                    with st.spinner("텍스트에서 파라미터 추출 중..."):
                        regex_results = di.extract_parameters(all_chunks, all_metas)

                    if use_llm_extract:
                        with st.spinner("LLM 심층 분석 중 (1~2분)..."):
                            llm_params = di.extract_parameters_llm(all_chunks, all_metas)
                        st.session_state["llm_params"] = llm_params
                    else:
                        llm_params = []

                    st.session_state["regex_params"] = regex_results

                # 결과 표시
                if "regex_params" in st.session_state:
                    regex_results = st.session_state["regex_params"]
                    llm_params    = st.session_state.get("llm_params", [])

                    # 논문별 파라미터 테이블
                    for doi, data in regex_results.items():
                        if not data.get("params"):
                            continue
                        with st.expander(f"📄 {data['title'][:70]}"):
                            rows = []
                            for param, occurrences in data["params"].items():
                                for occ in occurrences[:3]:
                                    rows.append({
                                        "파라미터": param,
                                        "값": occ["value"],
                                        "단위": occ["unit"],
                                        "문맥": occ["context"][:60],
                                    })
                            if rows:
                                import pandas as pd
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # LLM 추출 결과
                    if llm_params:
                        st.markdown("---")
                        st.markdown("#### 🤖 LLM 추출 파라미터")
                        import pandas as pd
                        df = pd.DataFrame(llm_params)
                        if not df.empty:
                            cols = [c for c in ["paper_title","param","value","unit","condition","context"] if c in df.columns]
                            st.dataframe(df[cols], use_container_width=True, hide_index=True)

                    # CSV 다운로드
                    if "regex_params" in st.session_state:
                        all_rows = []
                        for doi, data in st.session_state["regex_params"].items():
                            for param, occs in data.get("params",{}).items():
                                for occ in occs:
                                    all_rows.append({"doi":doi,"title":data["title"][:60],
                                                      "param":param,"value":occ["value"],"unit":occ["unit"]})
                        if all_rows:
                            import pandas as pd
                            csv = pd.DataFrame(all_rows).to_csv(index=False)
                            st.download_button("📥 CSV 다운로드", csv, "parameters.csv", "text/csv")

            # ════════════════════════════════════════
            # 🔬 실험 조건 매칭
            # ════════════════════════════════════════
            with deep_tabs[1]:
                st.markdown("#### 🔬 실험 조건 매칭 엔진")
                st.caption("현재 작업 중인 공정 조건과 가장 유사한 논문 섹션을 우선순위로 필터링합니다.")

                mc1, mc2 = st.columns(2)
                with mc1:
                    tech_choice = st.selectbox("기술 노드",
                        ["FinFET","GAA","Planar","FDSOI","FinFET+GAA"], index=0)
                    node_nm = st.number_input("노드 (nm)", min_value=1, max_value=130, value=7)
                with mc2:
                    metal_pitch = st.number_input("Metal Pitch (nm, 0=미지정)", min_value=0, value=0)
                    extra_kw = st.text_input("추가 키워드 (쉼표 구분)",
                                              placeholder="예: write assist, Vmin, BTI")

                match_btn = st.button("🔍 유사 조건 검색", use_container_width=False)
                if match_btn:
                    conditions = {
                        "technology": tech_choice,
                        "node_nm": node_nm,
                        "metal_pitch": metal_pitch if metal_pitch > 0 else None,
                        "extra_keywords": [k.strip() for k in extra_kw.split(",") if k.strip()],
                    }
                    with st.spinner("조건 매칭 중..."):
                        matches = di.match_conditions(conditions, all_chunks, all_metas)
                    st.session_state["cond_matches"] = matches

                if "cond_matches" in st.session_state:
                    matches = st.session_state["cond_matches"]
                    if not matches:
                        st.warning("일치하는 섹션을 찾지 못했습니다.")
                    else:
                        st.success(f"✅ {len(matches)}개 관련 섹션 발견")
                        for i, m in enumerate(matches):
                            score_bar = "🟦" * min(m["score"], 8)
                            with st.expander(
                                f"#{i+1} {score_bar} | {m['title'][:55]} ({m['year']}) — 매칭 점수: {m['score']}"
                            ):
                                st.markdown(f"**매칭 키워드:** {', '.join(m['matched_terms'])}")
                                st.markdown(f"**관련 구절:**")
                                st.markdown(f"> {m['chunk'][:400]}")

            # ════════════════════════════════════════
            # 📊 논문 비교 뷰어
            # ════════════════════════════════════════
            with deep_tabs[2]:
                st.markdown("#### 📊 논문 간 기술 비교 뷰어")
                st.caption("2~3편의 논문을 선택하여 핵심 기술 요소를 테이블로 자동 비교합니다.")

                paper_options = {
                    f"{p.get('title','')[:60]} ({p.get('year','')})": doi
                    for doi, p in st.session_state.papers.items()
                }
                selected_labels = st.multiselect(
                    "비교할 논문 선택 (2~3편)",
                    list(paper_options.keys()),
                    max_selections=3,
                )

                # 비교 측면 커스터마이징
                default_aspects = [
                    "주요 기여/Novelty", "사용 기술/공정", "노드(nm)",
                    "핵심 파라미터(Vmin/SNM)", "실험 방법론", "주요 결과",
                    "한계점", "향후 과제",
                ]
                custom_aspects = st.text_area(
                    "비교 측면 (한 줄에 하나씩)",
                    value="\n".join(default_aspects),
                    height=120,
                    label_visibility="collapsed",
                )
                aspects = [a.strip() for a in custom_aspects.split("\n") if a.strip()]

                compare_btn = st.button("📊 비교 분석 시작", use_container_width=False,
                                        disabled=len(selected_labels) < 2)

                if compare_btn and len(selected_labels) >= 2:
                    selected_dois = [paper_options[l] for l in selected_labels]
                    papers_for_compare = []
                    for doi in selected_dois:
                        paper = st.session_state.papers.get(doi, {})
                        chunks = [d for d, m in zip(all_chunks, all_metas) if m.get("doi") == doi]
                        papers_for_compare.append({
                            "doi": doi,
                            "title": paper.get("title",""),
                            "chunks": chunks[:4],
                        })

                    with st.spinner("LLM 비교 분석 중 (30초~1분)..."):
                        result = di.compare_papers(papers_for_compare, aspects)
                    st.session_state["compare_result"] = result

                if "compare_result" in st.session_state:
                    r = st.session_state["compare_result"]
                    titles = r.get("paper_titles", [])

                    # 비교 테이블
                    if r.get("table"):
                        import pandas as pd
                        rows = []
                        for row in r["table"]:
                            entry = {"비교 항목": row["aspect"]}
                            for i, val in enumerate(row.get("papers",[])):
                                label = titles[i][:25] if i < len(titles) else f"논문{i+1}"
                                entry[label] = val
                            rows.append(entry)
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # 핵심 차이점
                    if r.get("key_differences"):
                        st.markdown("---")
                        st.markdown("**🔍 핵심 차이점**")
                        st.info(r["key_differences"])
                    if r.get("recommendation"):
                        st.markdown("**💡 활용 권장**")
                        st.success(r["recommendation"])

                    # 엑셀 다운로드
                    if r.get("table"):
                        import pandas as pd
                        import io
                        rows = []
                        for row in r["table"]:
                            entry = {"비교 항목": row["aspect"]}
                            for i, val in enumerate(row.get("papers",[])):
                                label = titles[i][:25] if i < len(titles) else f"논문{i+1}"
                                entry[label] = val
                            rows.append(entry)
                        buf = io.BytesIO()
                        pd.DataFrame(rows).to_excel(buf, index=False)
                        st.download_button("📥 Excel 다운로드", buf.getvalue(),
                                           "paper_comparison.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ════════════════════════════════════════
            # ⚔️ Debate Mode
            # ════════════════════════════════════════
            with deep_tabs[3]:
                st.markdown("#### ⚔️ Debate Mode — 기술 토론 분석")
                st.caption("상반된 결론의 두 논문을 공정 엔지니어 관점에서 토론 분석합니다.")

                topic_input = st.text_input(
                    "토론 주제",
                    placeholder="예: GAA FET의 SRAM Vmin 개선 효과는 FinFET 대비 충분한가?",
                )
                db1, db2 = st.columns(2)
                paper_options = {
                    f"{p.get('title','')[:55]} ({p.get('year','')})": doi
                    for doi, p in st.session_state.papers.items()
                }
                with db1:
                    st.markdown("**📗 찬성/주장 논문**")
                    pro_label = st.selectbox("찬성 논문", list(paper_options.keys()), key="pro_paper")
                with db2:
                    st.markdown("**📕 반대/비판 논문**")
                    con_label = st.selectbox("반대 논문", list(paper_options.keys()), key="con_paper",
                                              index=min(1, len(paper_options)-1))

                user_fab = st.text_area(
                    "사내 공정 환경 (선택)",
                    placeholder="예: 현재 7nm FinFET 기반, Metal Pitch 36nm, 수율 목표 99.9%...",
                    height=60,
                )

                debate_btn = st.button("⚔️ 토론 시작", use_container_width=False,
                                       disabled=not topic_input.strip())

                if debate_btn and topic_input:
                    pro_doi = paper_options.get(pro_label,"")
                    con_doi = paper_options.get(con_label,"")
                    pro_chunks = [d for d,m in zip(all_chunks,all_metas) if m.get("doi")==pro_doi][:3]
                    con_chunks = [d for d,m in zip(all_chunks,all_metas) if m.get("doi")==con_doi][:3]

                    with st.spinner("공정 엔지니어 관점 토론 분석 중 (1~2분)..."):
                        result = di.run_debate(
                            topic_input, pro_chunks, pro_label[:60],
                            con_chunks, con_label[:60], user_fab,
                        )
                    st.session_state["debate_result"] = result

                if "debate_result" in st.session_state:
                    r = st.session_state["debate_result"]

                    st.markdown(f"### 🎯 {r.get('topic_summary','')}")
                    st.markdown("---")

                    dc1, dc2 = st.columns(2)
                    with dc1:
                        pro = r.get("pro_argument",{})
                        st.markdown("##### 📗 찬성 측")
                        if pro:
                            st.markdown(f"**주장:** {pro.get('main_claim','')}")
                            st.markdown("**근거:**")
                            for ev in pro.get("key_evidence",[]):
                                st.markdown(f"- {ev}")
                            st.success(f"💪 강점: {pro.get('strength','')}")
                            st.error(f"⚠️ 약점: {pro.get('weakness','')}")

                    with dc2:
                        con = r.get("con_argument",{})
                        st.markdown("##### 📕 반대/비판 측")
                        if con:
                            st.markdown(f"**주장:** {con.get('main_claim','')}")
                            st.markdown("**근거:**")
                            for ev in con.get("key_evidence",[]):
                                st.markdown(f"- {ev}")
                            st.success(f"💪 강점: {con.get('strength','')}")
                            st.error(f"⚠️ 약점: {con.get('weakness','')}")

                    st.markdown("---")
                    if r.get("hidden_assumptions"):
                        st.markdown("#### 🕵️ Hidden Assumptions — 저자들이 놓친 것")
                        for i, ha in enumerate(r["hidden_assumptions"],1):
                            st.warning(f"**{i}.** {ha}")

                    if r.get("engineer_verdict"):
                        st.markdown("#### 🏭 공정 엔지니어 최종 의견")
                        st.info(r["engineer_verdict"])

                    if r.get("practical_risk"):
                        st.markdown("#### ⚠️ 현장 적용 리스크")
                        for risk in r["practical_risk"]:
                            st.error(f"🚨 {risk}")

            # ════════════════════════════════════════
            # 🩺 증상 기반 역추적
            # ════════════════════════════════════════
            with deep_tabs[4]:
                st.markdown("#### 🩺 증상 기반 역추적 (Symptom-to-Theory)")
                st.caption("현재 겪고 있는 SRAM/공정 이상 증상을 입력하면 원인 메커니즘과 관련 논문을 역추적합니다.")

                symptom_input = st.text_area(
                    "증상 설명",
                    placeholder=(
                        "예시 1: FS 코너에서 Vmin이 급격히 증가하고 Read Disturb 불량이 발생\n"
                        "예시 2: 저온(-40°C)에서 write failure rate가 예상보다 10배 높음\n"
                        "예시 3: 특정 Metal Layer 변경 후 SRAM Yield가 급락"
                    ),
                    height=120,
                )

                st.markdown("**공정 컨텍스트 (선택)**")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    ctx_tech = st.selectbox("기술", ["FinFET","GAA","Planar","FDSOI","기타"], key="sym_tech")
                with sc2:
                    ctx_node = st.number_input("노드(nm)", 1, 130, 7, key="sym_node")
                with sc3:
                    ctx_temp = st.selectbox("온도", ["상온","저온(-40°C)","고온(125°C)","전범위"], key="sym_temp")

                symp_btn = st.button("🩺 진단 시작", use_container_width=False,
                                     disabled=not symptom_input.strip())

                if symp_btn and symptom_input.strip():
                    full_symptom = f"{symptom_input} (기술: {ctx_tech}, {ctx_node}nm, {ctx_temp})"
                    with st.spinner("① 물리 메커니즘 추론 → ② 논문 역추적 중..."):
                        result = di.symptom_to_theory(
                            full_symptom, all_chunks, all_metas,
                            engine._embedder, engine._collection,
                        )
                    st.session_state["symp_result"] = result

                if "symp_result" in st.session_state:
                    r = st.session_state["symp_result"]
                    mech = r.get("mechanisms",{})

                    st.markdown("---")
                    if mech.get("differential_diagnosis"):
                        st.markdown("#### 🔬 감별 진단")
                        st.info(mech["differential_diagnosis"])

                    if mech.get("root_causes"):
                        st.markdown("#### 📐 추정 근본 원인")
                        for i, rc in enumerate(mech["root_causes"], 1):
                            with st.expander(f"**원인 {i}: {rc.get('mechanism','')}**"):
                                st.markdown(f"**물리적 설명:** {rc.get('physics','')}")
                                kws = rc.get("keywords",[])
                                if kws:
                                    st.markdown(f"**관련 키워드:** `{'` `'.join(kws)}`")

                    if mech.get("recommended_experiments"):
                        st.markdown("#### 🧪 권장 실험")
                        for exp in mech["recommended_experiments"]:
                            st.markdown(f"- {exp}")

                    related = r.get("related_papers",[])
                    if related:
                        st.markdown("---")
                        st.markdown(f"#### 📚 역추적된 관련 논문 ({len(related)}편)")
                        for p in related:
                            with st.expander(f"📄 {p['title'][:65]} ({p['year']})"):
                                st.markdown(f"**DOI:** `{p['doi']}`")
                                st.markdown(f"**관련 구절:**")
                                st.markdown(f"> {p['relevant_chunk']}")

            # ════════════════════════════════════════
            # 🌳 기술 계통도
            # ════════════════════════════════════════
            with deep_tabs[5]:
                st.markdown("#### 🌳 기술 진화 계통도")
                st.caption("수집된 논문들의 기술 계승 관계를 계통도로 시각화합니다.")

                tree_btn = st.button("🌳 계통도 생성", use_container_width=False)

                if tree_btn:
                    with st.spinner("LLM 기술 계보 분석 중 (1~2분)..."):
                        tree_data = di.build_evolution_tree(
                            st.session_state.papers, all_chunks, all_metas
                        )
                    st.session_state["tree_data"] = tree_data

                if "tree_data" in st.session_state:
                    tree = st.session_state["tree_data"]
                    nodes = tree.get("nodes",[])
                    edges = tree.get("edges",[])
                    clusters = tree.get("clusters",[])

                    # 클러스터 범례
                    if clusters:
                        st.markdown("**기술 클러스터:**")
                        cluster_cols = st.columns(min(len(clusters), 4))
                        CLUSTER_COLORS = {
                            "blue":"#4f46e5","teal":"#0d9488","purple":"#7c3aed",
                            "green":"#16a34a","amber":"#d97706","coral":"#e11d48",
                            "gray":"#6b7280",
                        }
                        for i, cl in enumerate(clusters):
                            color = CLUSTER_COLORS.get(cl.get("color","blue"), "#4f46e5")
                            with cluster_cols[i % 4]:
                                st.markdown(
                                    f'<div style="background:{color}22;border-left:3px solid {color};'
                                    f'padding:6px 10px;border-radius:4px;font-size:12px;">'
                                    f'<b>{cl["name"]}</b><br><span style="color:#888">{cl.get("description","")[:50]}</span></div>',
                                    unsafe_allow_html=True
                                )

                    # vis.js 계통도 렌더링
                    import json as _json
                    CLUSTER_COLORS_VIS = {
                        "blue":"#4f46e5","teal":"#0d9488","purple":"#7c3aed",
                        "green":"#16a34a","amber":"#d97706","coral":"#e11d48","gray":"#6b7280",
                    }
                    cluster_color_map = {
                        cl["id"]: CLUSTER_COLORS_VIS.get(cl.get("color","blue"),"#4f46e5")
                        for cl in clusters
                    }

                    vis_nodes = []
                    for n in nodes:
                        color = cluster_color_map.get(n.get("cluster","c1"), "#4f46e5")
                        size  = 10 + n.get("importance",2) * 5
                        vis_nodes.append({
                            "id": n["doi"],
                            "label": f"{n.get('label','')[:22]}\n({n.get('year','')})",
                            "title": f"<b>{n.get('label','')}</b><br>{n.get('innovation','')}",
                            "color": {"background": color+"33", "border": color,
                                      "highlight": {"background": color+"66", "border": color}},
                            "size": size,
                            "font": {"color": "#e2e8f0", "size": 11},
                            "borderWidth": 2,
                        })

                    RELATION_COLORS = {
                        "extends":     "#818cf8",
                        "improves":    "#4ade80",
                        "contradicts": "#f87171",
                        "applies":     "#fbbf24",
                    }
                    vis_edges = []
                    for e in edges:
                        ec = RELATION_COLORS.get(e.get("relation","extends"), "#818cf8")
                        vis_edges.append({
                            "from": e["from"], "to": e["to"],
                            "title": e.get("description",""),
                            "color": {"color": ec, "opacity": 0.8, "highlight": ec},
                            "width": 2,
                            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
                            "dashes": e.get("relation") == "contradicts",
                            "smooth": {"type": "curvedCW", "roundness": 0.15},
                        })

                    nodes_json = _json.dumps(vis_nodes)
                    edges_json = _json.dumps(vis_edges)

                    tree_html = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.css"/>
<style>* {{margin:0;padding:0;box-sizing:border-box;}} body {{background:#0a0e1a;}} #net {{width:100%;height:560px;}}</style>
</head><body>
<div id="net"></div>
<div style="position:absolute;bottom:10px;left:10px;background:rgba(17,24,39,.9);border:1px solid #1f2937;border-radius:8px;padding:8px 12px;font-family:sans-serif;font-size:11px;color:#94a3b8;">
  <div style="margin-bottom:4px;color:#e2e8f0;font-weight:600;">관계 유형</div>
  <div><span style="color:#818cf8">━</span> 확장/계승 (extends)</div>
  <div><span style="color:#4ade80">━</span> 개선 (improves)</div>
  <div><span style="color:#f87171">┅</span> 반박 (contradicts)</div>
  <div><span style="color:#fbbf24">━</span> 적용 (applies)</div>
</div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const net = new vis.Network(document.getElementById('net'), {{nodes, edges}}, {{
  layout: {{hierarchical: {{enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 220, nodeSpacing: 80}}}},
  physics: {{enabled: false}},
  interaction: {{hover: true, tooltipDelay: 100, zoomView: true, dragView: true}},
  nodes: {{shape: "dot", scaling: {{min: 12, max: 40}}}},
  edges: {{width: 2}},
}});
net.once('stabilizationIterationsDone', () => net.setOptions({{physics: {{enabled: false}}}}));
</script></body></html>"""

                    components.html(tree_html, height=580, scrolling=False)

                    # 엣지 목록 (텍스트)
                    if edges:
                        st.markdown("---")
                        st.markdown("**기술 계승 관계 목록**")
                        rel_labels = {"extends":"→ 확장","improves":"→ 개선",
                                      "contradicts":"⟷ 반박","applies":"→ 적용"}
                        for e in edges[:20]:
                            fn = next((n.get("label","") for n in nodes if n["doi"]==e["from"]), e["from"][:30])
                            tn = next((n.get("label","") for n in nodes if n["doi"]==e["to"]),   e["to"][:30])
                            rel = rel_labels.get(e.get("relation","extends"), "→")
                            desc = e.get("description","")
                            st.markdown(f"- `{fn[:28]}` **{rel}** `{tn[:28]}`{f' — {desc}' if desc else ''}")
