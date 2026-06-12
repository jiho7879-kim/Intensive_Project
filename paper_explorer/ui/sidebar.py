"""
사이드바 설정 UI — 오픈소스 LLM 백엔드 선택 지원
"""
import streamlit as st

BACKEND_INFO = {
    "ollama": {
        "label": "🦙 Ollama (로컬, 완전 무료)",
        "desc": "PC에서 직접 실행. 인터넷 불필요. GPU 권장.",
        "needs_key": False,
        "models": ["llama3.2", "llama3.1", "mistral", "qwen2.5", "gemma3", "phi4"],
        "install": "https://ollama.com/download",
    },
    "groq": {
        "label": "⚡ Groq (클라우드, 무료 티어)",
        "desc": "무료 API 키 발급 후 사용. 매우 빠름.",
        "needs_key": True,
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "install": "https://console.groq.com",
    },
    "lmstudio": {
        "label": "🖥️ LM Studio (로컬 서버)",
        "desc": "LM Studio 앱으로 모델 로드 후 사용.",
        "needs_key": False,
        "models": ["local-model"],
        "install": "https://lmstudio.ai",
    },
}


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1f2937; }
        .backend-badge {
            display:inline-block; padding:2px 8px; border-radius:12px;
            font-size:11px; font-weight:600; margin-bottom:6px;
        }
        .badge-local  { background:#052e16; color:#4ade80; border:1px solid #166534; }
        .badge-cloud  { background:#1e1b4b; color:#818cf8; border:1px solid #3730a3; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ 설정")
        st.markdown("---")

        # ── LLM 백엔드 선택 ──────────────────────────────────────────────────
        st.markdown("**🤖 LLM 백엔드**")
        backend = st.selectbox(
            "백엔드 선택",
            options=list(BACKEND_INFO.keys()),
            format_func=lambda x: BACKEND_INFO[x]["label"],
            index=0,
            label_visibility="collapsed",
        )
        info = BACKEND_INFO[backend]
        badge_cls = "badge-cloud" if backend == "groq" else "badge-local"
        badge_txt = "클라우드" if backend == "groq" else "로컬 실행"
        st.markdown(
            f'<span class="backend-badge {badge_cls}">{badge_txt}</span> '
            f'<span style="font-size:12px;color:#64748b;">{info["desc"]}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"📎 설치/발급: {info['install']}")

        # 모델 선택
        model_options = info["models"]
        if backend == "lmstudio":
            model = st.text_input("모델명 (LM Studio 로드된 모델)", value="local-model",
                                  label_visibility="collapsed")
        else:
            model = st.selectbox("모델", model_options, label_visibility="collapsed")

        # Groq API 키
        api_key = ""
        if info["needs_key"]:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.get("groq_key", ""),
                placeholder="gsk_...",
                help="groq.com/console 에서 무료 발급 (신용카드 불필요)",
            )
            st.session_state.groq_key = api_key

        # Ollama / LM Studio 서버 주소
        base_url = ""
        if backend in ("ollama", "lmstudio"):
            default_url = "http://localhost:11434" if backend == "ollama" else "http://localhost:1234/v1"
            base_url = st.text_input(
                "서버 주소",
                value=st.session_state.get(f"{backend}_url", default_url),
                help="원격 서버 사용 시 변경",
            )
            st.session_state[f"{backend}_url"] = base_url

        st.markdown("---")

        # ── 이메일 (OpenAlex polite pool) ────────────────────────────────────
        st.markdown("**📧 OpenAlex Polite Pool 이메일** (권장)")
        oa_email = st.text_input(
            "이메일",
            value=st.session_state.get("oa_email", "researcher@example.com"),
            placeholder="your@email.com",
            label_visibility="collapsed",
            help="API 키 불필요. 이메일 입력 시 더 빠른 응답 (Polite Pool)",
        )
        st.session_state.oa_email = oa_email
        st.caption("🔓 OpenAlex는 완전 무료 · 인증 불필요 · 2억+ 논문 보유")

        st.markdown("---")

        # ── RAG 고급 옵션 ─────────────────────────────────────────────────────
        st.markdown("**🧠 RAG 고급 옵션**")
        use_hyde = st.toggle("HyDE 쿼리 강화", value=True,
                             help="가상 문서 생성으로 임베딩 품질 향상 (LLM 1회 추가 호출)")
        use_multiquery = st.toggle("멀티쿼리 확장", value=True,
                                   help="질문을 3개로 확장해 검색 재현율 향상 (LLM 1회 추가 호출)")
        use_reranker = st.toggle("Cross-encoder 재순위화", value=True,
                                 help="sentence-transformers 설치 시 활성화. 정밀도 향상")
        st.session_state.rag_options = {
            "use_hyde": use_hyde,
            "use_multiquery": use_multiquery,
            "use_reranker": use_reranker,
        }

        st.markdown("---")

        # ── 그래프 설정 ───────────────────────────────────────────────────────
        st.markdown("**🕸️ 그래프 탐색**")
        depth = st.slider("탐색 깊이", 1, 3, 2,
                          help="1=직접 인용, 2=인용의 인용, 3=3단계")
        max_papers = st.slider("최대 논문 수", 10, 200, 50, step=10)
        direction = st.selectbox(
            "탐색 방향",
            ["references", "citations", "both"],
            format_func=lambda x: {
                "references": "📖 참고문헌 (선행연구)",
                "citations":  "📣 피인용 (후속연구)",
                "both":       "🔄 양방향",
            }[x],
        )

        st.markdown("---")

        # ── PDF 설정 ──────────────────────────────────────────────────────────
        st.markdown("**📁 PDF 설정**")
        pdf_dir = st.text_input("저장 폴더", value="./downloaded_pdfs")
        email   = st.text_input(
            "이메일 (Unpaywall용)",
            value=st.session_state.get("user_email", "researcher@example.com"),
        )
        st.session_state.user_email = email

        st.markdown("---")

        # ── 도움말 ────────────────────────────────────────────────────────────
        with st.expander("🦙 Ollama 빠른 시작"):
            st.code("""# 1) 설치 (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2) 모델 다운로드 (약 2GB)
ollama pull llama3.2

# 3) 서버 실행 (자동 시작됨)
ollama serve""", language="bash")

        with st.expander("⚡ Groq 무료 발급"):
            st.markdown("""
1. [console.groq.com](https://console.groq.com) 회원가입
2. API Keys 메뉴 → Create API Key
3. 무료 분당 30 요청, 하루 1000 요청
""")

        with st.expander("🔬 SRAM 예시 DOI"):
            for name, doi in [
                ("FinFET SRAM Stability",  "10.1109/IEDM.2012.6479083"),
                ("6T SRAM Vmin Analysis",  "10.1109/JSSC.2013.2252394"),
                ("SRAM Write Assist",      "10.1109/ISSCC.2019.8662503"),
            ]:
                if st.button(f"📄 {name}", key=f"ex_{doi}"):
                    st.session_state.doi_input = doi
                    st.rerun()

    return {
        "backend":   backend,
        "model":     model,
        "api_key":   api_key,
        "base_url":  base_url,
        "oa_email":   oa_email,
        "depth":     depth,
        "max_papers": max_papers,
        "direction": direction,
        "pdf_dir":   pdf_dir,
        "email":     email,
    }
