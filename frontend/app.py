"""
StoxAI — Streamlit frontend, fully wired to the real backend
================================================================

Nothing in this file is mocked or hardcoded. Every screen renders
whatever the backend actually returns:

  POST /auth/signup         -> real account creation (email/username/password)
  POST /auth/login          -> real credential check, returns a JWT
  GET  /me                  -> real user profile
  POST /chat                -> real LangGraph response

AUTH NOTE: switched from Google OAuth to manual email/username/password
because OAuth was erroring during testing — see main.py for the matching
backend change. There's no query-param token handoff anymore; /auth/login
and /auth/signup return the JWT directly in the JSON response.

The "Agent run details" panel does NOT guess which agent ran — the
backend's ChatResponse includes `intent` directly (see main.py's
ChatResponse model / core.py's AgentState), so the panel just reads that
field and renders whichever result key came back non-empty:
tax_result, stock_prediction_result, stock_info_result,
save_interest_result, planning_result, report_result.

Known shapes (tax_result, stock_prediction_result) are rendered with a
proper layout because their Pydantic models are defined in core.py
(TaxSavingResponse, StockPredictionResult). Shapes not pinned down in
core.py (planning_result, report_result, stock_info_result) fall back to
a generic dynamic renderer — whatever keys/values the backend actually
sends, displayed as-is, so the UI never lies about data it hasn't seen.

BEFORE RUNNING:
  1. Set BACKEND_URL below or via .streamlit/secrets.toml.
  2. Backend's FRONTEND_URL must point back at wherever this runs.

KNOWN LIMITATION — the watchlist shown in the sidebar is NOT fetched from
a database on load, because no such endpoint exists yet: main.py has no
GET /watchlist route, and GET /me deliberately doesn't include `stocks`
(checked directly in main.py's get_me()). So the sidebar's watchlist is
only ever populated from real save_interest_result payloads returned
during THIS session's /chat calls — it resets on page reload even though
the data is still sitting in Mongo. If persistence across reloads
matters before the demo, the real fix is a small addition to main.py:

    @app.get("/watchlist")
    def get_watchlist(current_user: dict = Depends(get_current_user)):
        return agents.get_user_stocks(current_user["user_id"])

That's a one-function addition using a function that already exists in
agents.py — ask whoever owns main.py before adding it, same as the
news_agent.py situation earlier.
"""

import datetime as dt

import requests
import streamlit as st

#st.set_page_config(page_title="StoxAI", page_icon="S", layout="wide")
st.set_page_config(page_title="StoxAI", page_icon="S", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# BACKEND URL
# ---------------------------------------------------------------------------

try:
    BACKEND_URL = st.secrets["API_BASE_URL"]
except Exception:
    BACKEND_URL = "http://localhost:8000"  # local dev fallback

REQUEST_TIMEOUT = 45  # Render free tier can cold-start slowly


# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1220;
        background-image: radial-gradient(circle, #16213A 1px, transparent 1px);
        background-size: 24px 24px;
        color: #E4E8F1;
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar: native collapse/expand behavior left intact on purpose —
       do NOT override display/visibility/transform/width here, and do NOT
       hide [data-testid="collapsedControl"]. Doing either breaks the
       expand arrow (it has nowhere to reappear) and breaks mobile (the
       sidebar can no longer slide off-screen, so it just eats the
       viewport). Only cosmetic, non-layout rules belong in this block. */
    section[data-testid="stSidebar"] {
        background-color: #0D1524;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 21rem;
    }
    .sx-logo { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .sx-logo-mark {
        width: 26px; height: 26px; border-radius: 7px; background: #E8A33D;
        display: flex; align-items: center; justify-content: center;
        font-weight: 600; font-size: 13px; color: #1A1204;
    }
    .sx-logo-text { font-weight: 600; font-size: 16px; color: #F3F5FA; }
    .sx-badge {
        display: inline-block; font-size: 0.72rem; font-weight: 500;
        padding: 3px 10px; border-radius: 20px;
    }
    .badge-tax_optimization { background: rgba(52,211,153,0.15); color: #34D399; border: 1px solid rgba(52,211,153,0.35);}
    .badge-stock_prediction { background: rgba(96,165,250,0.15); color: #60A5FA; border: 1px solid rgba(96,165,250,0.35);}
    .badge-stock_info { background: rgba(96,165,250,0.15); color: #60A5FA; border: 1px solid rgba(96,165,250,0.35);}
    .badge-save_interest { background: rgba(232,163,61,0.15); color: #E8A33D; border: 1px solid rgba(232,163,61,0.35);}
    .badge-portfolio_planning { background: rgba(167,139,250,0.15); color: #A78BFA; border: 1px solid rgba(167,139,250,0.35);}
    .badge-report { background: rgba(167,139,250,0.15); color: #A78BFA; border: 1px solid rgba(167,139,250,0.35);}
    .badge-general { background: rgba(107,118,144,0.15); color: #9AA4BD; border: 1px solid rgba(107,118,144,0.35);}
    .sx-panel {
        border: 1px solid #223052; border-radius: 10px; padding: 0.9rem 1rem;
        background: #111A2C; margin-bottom: 0.6rem;
    }
    .sx-meta { font-size: 0.72rem; color: #6B7690; margin-top: 4px; }
    div.stButton > button, div.stLinkButton > a {
        background-color: #E8A33D !important; color: #1A1204 !important; border: none !important;
        border-radius: 8px; font-weight: 600;
    }
    .stMarkdown table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 12px 0 !important;
        table-layout: auto !important;
    }
    .stMarkdown th, .stMarkdown td {
        border: 1px solid #223052 !important;
        padding: 8px 12px !important;
        text-align: left !important;
        vertical-align: top !important;
        line-height: 1.5 !important;
        font-size: 0.88rem !important;
        word-break: break-word !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

INTENT_LABELS = {
    "tax_optimization": "Tax agent",
    "stock_prediction": "Prediction agent",
    "stock_info": "Stock info agent",
    "save_interest": "Watchlist agent",
    "portfolio_planning": "Planning agent",
    "report": "Report agent",
    "general": "General knowledge",
}


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "page": "home",      # "home" | "login" | (dashboard shows automatically once user is set)
        "token": None,
        "user": None,
        "chat_history": [],
        "run_log": [],       # every real /chat response, in full, newest first
        "tax_profile": None,
        "watchlist": {},     # {SYMBOL: {live_price, predicted_price, updated_at}} — derived
                              # ONLY from real save_interest_result.watchlist payloads seen
                              # this session. No GET /watchlist endpoint exists yet (see
                              # note in render_dashboard), so this resets on page reload.
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# BACKEND CALLS
# ---------------------------------------------------------------------------

def fetch_me(token: str):
    try:
        res = requests.get(
            f"{BACKEND_URL}/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        st.error(f"Couldn't reach the backend at {BACKEND_URL} — is it running/deployed? ({e})")
        return None
    return res.json() if res.status_code == 200 else None


def send_chat(user_query: str) -> dict:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    payload = {
        "user_query": user_query,
        "chat_history": st.session_state.chat_history,
        "portfolio": [],
        "tax_profile": st.session_state.tax_profile,
    }
    res = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    return res.json()


if st.session_state.token and not st.session_state.user:
    # Defensive fallback only — normal login/signup sets both token and
    # user together via _apply_auth_response(). This path only matters if
    # a token somehow exists without user data already attached.
    me = fetch_me(st.session_state.token)
    if me:
        st.session_state.user = me
    else:
        st.session_state.token = None
        st.warning("Your session expired. Please log in again.")


def process_chat_turn(user_message: str, echo_in_chat: bool = True):
    """
    Sends a real message through /chat and updates every piece of state
    that depends on the response — chat transcript, run log, AND the
    cached watchlist (any turn can return save_interest_result, not just
    ones sent from the dedicated widget below, since a normal chat
    message like "my favorite stock is TSLA" triggers the same node).
    """
    response = send_chat(user_message)
    reply = response.get("final_response", "(no response)")

    stock_info = response.get("stock_info_result") or {}
    graph_b64 = stock_info.get("graph_image_b64")

    if echo_in_chat:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        asst_msg = {
            "role": "assistant",
            "content": reply,
            "image_b64": graph_b64,
            "agents_executed": response.get("agents_executed"),
            "intent": response.get("intent", "general"),
        }
        st.session_state.chat_history.append(asst_msg)

    st.session_state.run_log.insert(
        0, {"response": response, "query": user_message, "ts": dt.datetime.now().strftime("%H:%M:%S")}
    )

    save_interest = response.get("save_interest_result") or {}
    if isinstance(save_interest, dict) and save_interest.get("watchlist"):
        st.session_state.watchlist = save_interest["watchlist"]  # real data straight from Mongo, via the backend

    return response


# ---------------------------------------------------------------------------
# DYNAMIC RENDERERS — every function here only shows what the backend
# actually sent. No field is assumed to exist beyond what core.py defines;
# anything not pinned down there is displayed generically.
# ---------------------------------------------------------------------------

def render_generic(data):
    """Fallback for shapes not pinned down as a Pydantic model in core.py
    (planning_result, report_result, stock_info_result) — shows exactly
    what came back, nothing invented."""
    if data is None:
        return
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                st.markdown(f"**{k}**")
                st.json(v)
            else:
                st.markdown(f"**{k}:** {v}")
    elif isinstance(data, list):
        st.json(data)
    else:
        st.write(data)


import re

def clean_markdown_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Convert raw html break tags inside tables/markdown to standard newlines
    cleaned = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return cleaned


def render_tax_result(data: dict):
    """Matches core.py's TaxSavingResponse: suggestions[], total_estimated_savings, explanation."""
    total = data.get("total_estimated_savings")
    if total is not None:
        st.markdown(f"**Estimated savings: {total:,.2f}**")
    for s in data.get("suggestions", []) or []:
        with st.container():
            st.markdown(f"**{clean_markdown_text(s.get('title', s.get('type', 'Suggestion')))}**")
            if s.get("detail"):
                st.markdown(clean_markdown_text(s["detail"]))
            if s.get("estimated_savings"):
                st.caption(f"Est. savings: {s['estimated_savings']:,.2f}")
            if s.get("source_snippets"):
                with st.expander("Source snippets (RAG)"):
                    for sn in s["source_snippets"]:
                        st.caption(f"• {clean_markdown_text(sn)}")
    if data.get("explanation"):
        st.markdown("---")
        st.markdown(clean_markdown_text(data["explanation"]))


def render_stock_prediction_result(data: list):
    """Matches core.py's StockPredictionResult: symbol, predicted_direction,
    predicted_range, confidence, explanation."""
    for item in data or []:
        symbol = item.get("symbol", "?")
        direction = item.get("predicted_direction", "?")
        confidence = item.get("confidence")
        st.markdown(f"**{symbol}** — {direction}" + (f" ({confidence:.0%} confidence)" if confidence is not None else ""))
        if item.get("predicted_range"):
            st.caption(f"Predicted range: {item['predicted_range']}")
        if item.get("explanation"):
            st.caption(item["explanation"])


import base64


def render_graph_b64(graph_b64: str):
    if not graph_b64:
        return
    try:
        raw_b64 = graph_b64.split(",", 1)[1] if "," in graph_b64 else graph_b64
        img_bytes = base64.b64decode(raw_b64)
        st.image(img_bytes, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")


def render_stock_info_result(data: dict):
    """Renders stock info including live price, predicted price (non-negative), and graph."""
    if not data or data.get("error"):
        st.warning(data.get("error", "No stock information available."))
        return

    symbol = data.get("symbol", "?")
    live_price = data.get("live_price")
    predicted_price = data.get("predicted_price")
    if predicted_price == -1 or predicted_price is None:
        predicted_price = live_price

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{symbol} Live Price", value=f"${live_price:,.2f}" if live_price is not None else "N/A")
    with col2:
        st.metric(label=f"{symbol} Predicted Price", value=f"${predicted_price:,.2f}" if predicted_price is not None else "N/A")

    graph_b64 = data.get("graph_image_b64")
    if graph_b64:
        st.markdown("**30-Day Performance History**")
        render_graph_b64(graph_b64)


def render_agent_pipeline(agents_executed: list = None, intent: str = "general"):
    if not agents_executed:
        label = INTENT_LABELS.get(intent, intent)
        badge_class = f"badge-{intent}" if intent in INTENT_LABELS else "badge-general"
        st.markdown(f"<span class='sx-badge {badge_class}'>{label}</span>", unsafe_allow_html=True)
        return

    html_parts = []
    for name in agents_executed:
        badge_cls = "badge-general"
        if "News" in name:
            badge_cls = "badge-stock_prediction"
        elif "Prediction" in name:
            badge_cls = "badge-stock_info"
        elif "Stock Info" in name or "Info" in name:
            badge_cls = "badge-save_interest"
        elif "Graph" in name or "Visualization" in name:
            badge_cls = "badge-tax_optimization"
        elif "Tax" in name:
            badge_cls = "badge-tax_optimization"
        elif "Watchlist" in name:
            badge_cls = "badge-save_interest"
        elif "Planning" in name:
            badge_cls = "badge-portfolio_planning"

        html_parts.append(f"<span class='sx-badge {badge_cls}'>{name}</span>")

    separator = " <span style='color:#6B7690; font-size:0.75rem;'>➔</span> "
    pipeline_html = separator.join(html_parts)
    st.markdown(f"<div style='margin-top:6px; margin-bottom:6px;'>{pipeline_html}</div>", unsafe_allow_html=True)


def render_planning_result(data: dict):
    if not data or data.get("status") == "need_stock_list":
        st.caption("No stocks evaluated yet.")
        return
    score = data.get("portfolio_score", "N/A")
    rating = data.get("score_rating", "")
    st.metric(label="Portfolio Score", value=f"{score} / 100", delta=rating)
    st.write("**Per-Stock Target Predictions:**")
    for item in data.get("symbols", []):
        sym = item.get("symbol")
        live = item.get("live_price")
        pred = item.get("predicted_price")
        pct = item.get("predicted_change_pct")
        st.markdown(f"**{sym}**: `{live}` ➔ `{pred}` ({pct})")
        if item.get("explanation"):
            st.caption(item["explanation"])


def render_agent_details(response: dict):
    intent = response.get("intent") or "general"
    agents_executed = response.get("agents_executed") or []
    render_agent_pipeline(agents_executed, intent)
    st.write("")

    if intent == "tax_optimization" and response.get("tax_result"):
        render_tax_result(response["tax_result"])
    elif intent == "stock_prediction" and response.get("stock_prediction_result"):
        render_stock_prediction_result(response["stock_prediction_result"])
    elif intent == "stock_info" and response.get("stock_info_result"):
        render_stock_info_result(response["stock_info_result"])
    elif intent == "save_interest" and response.get("save_interest_result"):
        render_generic(response["save_interest_result"])
    elif intent == "portfolio_planning" and response.get("planning_result"):
        render_planning_result(response["planning_result"])
    elif intent == "report" and response.get("report_result"):
        render_generic(response["report_result"])
    else:
        st.caption("No specialized agent data returned for this turn.")


# ---------------------------------------------------------------------------
# LOGIN SCREEN
# ---------------------------------------------------------------------------

def signup_request(username: str, email: str, password: str):
    res = requests.post(
        f"{BACKEND_URL}/auth/signup",
        json={"username": username, "email": email, "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    return res


def login_request(email: str, password: str):
    res = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"email": email, "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    return res


def _apply_auth_response(data: dict):
    st.session_state.token = data["token"]
    st.session_state.user = {
        "user_id": data["user_id"],
        "username": data["username"],
        "email": data["email"],
        "photo": None,  # manual signup has no avatar — real value, not a placeholder image
    }


def render_home():
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown(
            '<div class="sx-logo"><div class="sx-logo-mark">S</div>'
            '<div class="sx-logo-text">StoxAI</div></div>',
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("Login", use_container_width=True, key="home_login_nav"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown(
        "<h1 style='color:#F3F5FA; font-size:2.2rem; font-weight:600; max-width:640px; "
        "line-height:1.3;'>One dashboard for the news, the numbers, and the tax bill</h1>"
        "<p style='color:#9AA4BD; font-size:1rem; max-width:520px; line-height:1.65;'>"
        "StoxAI reads today's news, checks a prediction model's odds, and helps with your "
        "taxes and portfolio — all through one chat, routed to the right agent automatically.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Get started", key="home_cta"):
        st.session_state.page = "login"
        st.rerun()

    st.write("")

    features = [
        ("Tax agent", "#34D399", "Answers capital-gains and deduction questions, grounded in your actual profile."),
        ("Prediction agent", "#60A5FA", "Runs the LSTM+GRU models on your watchlist to estimate direction and confidence."),
        ("Stock info agent", "#60A5FA", "Ask how any stock is doing right now, in plain language."),
        ("Watchlist agent", "#E8A33D", "Add stocks by just mentioning them — no separate form needed."),
        ("Planning agent", "#A78BFA", "Scores your portfolio and suggests rebalancing moves."),
        ("Report agent", "#A78BFA", "Pulls everything into a summary when you ask for one."),
    ]

    for row_start in range(0, len(features), 3):
        cols = st.columns(3)
        for col, (title, color, desc) in zip(cols, features[row_start:row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="sx-panel" style="border-left:3px solid {color};">
                        <div style="font-size:0.95rem; font-weight:600; color:#F3F5FA; margin-bottom:6px;">{title}</div>
                        <div style="font-size:0.85rem; color:#9AA4BD; line-height:1.5;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")


def render_login():
    st.markdown(
        '<div class="sx-logo"><div class="sx-logo-mark">S</div>'
        '<div class="sx-logo-text">StoxAI</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("← Back to home", key="login_back_home"):
        st.session_state.page = "home"
        st.rerun()
    st.write("")

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            with st.form(key="login_form"):
                login_email = st.text_input("Email Address", key="login_email_val")
                login_password = st.text_input("Password", type="password", key="login_password_val")
                submitted_login = st.form_submit_button("Log in", use_container_width=True)

            if submitted_login:
                if not login_email or not login_password:
                    st.warning("Enter both email and password.")
                else:
                    try:
                        res = login_request(login_email, login_password)
                    except requests.RequestException as e:
                        st.error(f"Couldn't reach the backend at {BACKEND_URL}: {e}")
                        res = None
                    if res is not None:
                        if res.status_code == 200:
                            _apply_auth_response(res.json())
                            st.rerun()
                        else:
                            detail = res.json().get("detail", "Login failed") if res.headers.get("content-type", "").startswith("application/json") else "Login failed"
                            st.error(detail)

        with tab_signup:
            with st.form(key="signup_form"):
                signup_username = st.text_input("Username", key="signup_username_val")
                signup_email = st.text_input("Email Address", key="signup_email_val")
                signup_password = st.text_input("Password", type="password", key="signup_password_val", help="At least 6 characters.")
                submitted_signup = st.form_submit_button("Sign up", use_container_width=True)

            if submitted_signup:
                if not signup_username or not signup_email or not signup_password:
                    st.warning("Fill in username, email, and password.")
                else:
                    try:
                        res = signup_request(signup_username, signup_email, signup_password)
                    except requests.RequestException as e:
                        st.error(f"Couldn't reach the backend at {BACKEND_URL}: {e}")
                        res = None
                    if res is not None:
                        if res.status_code == 200:
                            _apply_auth_response(res.json())
                            st.rerun()
                        else:
                            detail = res.json().get("detail", "Sign up failed") if res.headers.get("content-type", "").startswith("application/json") else "Sign up failed"
                            st.error(detail)


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

def render_dashboard():
    user = st.session_state.user

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown(
            '<div class="sx-logo"><div class="sx-logo-mark">S</div>'
            '<div class="sx-logo-text">StoxAI</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"Welcome back, **{user.get('username') or user.get('email')}**")
    with top_r:
        if st.button("Log out", use_container_width=True):
            for k in ("token", "user", "chat_history", "run_log"):
                st.session_state[k] = None if k in ("token", "user") else []
            st.session_state.page = "home"
            st.rerun()

    with st.sidebar:
        if user.get("photo"):
            st.image(user["photo"], width=64)
        st.write(f"**{user.get('username', 'User')}**")
        st.caption(user.get("email", ""))
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.session_state.run_log = []
            st.rerun()

        st.divider()

        with st.expander("Tax profile (optional)"):
            st.caption("Sent with every message so the tax agent has real context.")
            filing_status = st.selectbox("Filing status", ["single", "married_joint", "married_separate", "hoi"])
            country = st.selectbox("Country", ["IN", "US"])
            annual_income = st.number_input("Annual income", min_value=0.0, step=1000.0)
            if st.button("Save tax profile", use_container_width=True):
                st.session_state.tax_profile = {
                    "filing_status": filing_status,
                    "country": country,
                    "annual_income": annual_income,
                }
                st.success("Saved for this session.")

        with st.expander("Ingest document to RAG (PDF, DOCX, CSV, TXT)"):
            st.caption("Upload your portfolio, tax docs, or financial notes. The RAG agent will read and answer queries about them!")
            uploaded_doc = st.file_uploader(
                "Upload file",
                type=["pdf", "docx", "txt", "md", "csv", "json"],
                key="rag_file_uploader",
            )
            if uploaded_doc is not None:
                if st.button("Ingest Document", use_container_width=True, key="btn_ingest_doc"):
                    with st.spinner("Embedding and ingesting document into RAG corpus..."):
                        try:
                            files = {"file": (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type or "application/octet-stream")}
                            headers = {"Authorization": f"Bearer {st.session_state.token}"}
                            res = requests.post(f"{BACKEND_URL}/ingest-user-file", files=files, headers=headers, timeout=60)
                            if res.status_code == 200:
                                data = res.json()
                                st.success(data.get("message", "Document ingested successfully!"))
                            else:
                                err = res.json().get("detail", "Ingestion failed") if res.headers.get("content-type", "").startswith("application/json") else res.text
                                st.error(err)
                        except Exception as e:
                            st.error(f"Error connecting to backend: {e}")

        st.divider()
        st.markdown("**Your watchlist**")
        st.caption("Predictions run nightly for whatever's added here.")

        new_tickers = st.text_input("Add tickers (comma-separated)", placeholder="e.g. AAPL, TCS, INFY", key="wl_input")
        if st.button("Add to watchlist", use_container_width=True):
            if not new_tickers.strip():
                st.warning("Enter at least one ticker first.")
            else:
                message = f"Add {new_tickers.strip()} to my watchlist"
                with st.spinner("Updating your watchlist..."):
                    try:
                        response = process_chat_turn(message, echo_in_chat=True)
                    except requests.HTTPError as e:
                        st.error(f"Backend error: {e.response.status_code} — {e.response.text}")
                        response = None
                    except requests.RequestException as e:
                        st.error(f"Couldn't reach the backend: {e}")
                        response = None

                if response:
                    result = response.get("save_interest_result") or {}
                    if result.get("error"):
                        st.error(f"Couldn't add that: {result['error']}")
                    else:
                        st.success(f"Added: {', '.join(result.get('saved', []))}")
                        st.rerun()

        if st.session_state.watchlist:
            for symbol, info in st.session_state.watchlist.items():
                predicted = info.get("predicted_price")
                live = info.get("live_price")
                predicted_display = live if (predicted == -1 or predicted is None) else predicted
                st.caption(
                    f"**{symbol}** — live: {live} · predicted: {predicted_display}"
                )
        else:
            st.caption("No stocks added yet this session. Add one above, or ask in chat "
                       "(e.g. \"my favorite stock is TSLA\") — either way triggers the same agent.")

    col_chat, col_details = st.columns([1.5, 1])

    with col_chat:
        st.markdown("##### Chat")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if isinstance(msg, dict) and msg.get("image_b64"):
                    render_graph_b64(msg["image_b64"])
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    render_agent_pipeline(msg.get("agents_executed"), msg.get("intent", "general"))

        prompt = st.chat_input("Ask about a stock, your portfolio, or your taxes...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking... (first request can be slow if the backend is waking up)"):
                    try:
                        response = process_chat_turn(prompt)
                    except requests.HTTPError as e:
                        st.error(f"Backend error: {e.response.status_code} — {e.response.text}")
                        st.stop()
                    except requests.RequestException as e:
                        st.error(f"Couldn't reach the backend: {e}")
                        st.stop()

                st.write(response.get("final_response", "(no response)"))
                stock_info = response.get("stock_info_result") or {}
                if stock_info.get("graph_image_b64"):
                    render_graph_b64(stock_info["graph_image_b64"])

                render_agent_pipeline(response.get("agents_executed"), response.get("intent", "general"))
            st.rerun()  # refresh so the details panel + watchlist reflect this turn immediately

    with col_details:
        st.markdown("##### Agent run details")
        if not st.session_state.run_log:
            st.caption("Ask something in chat — the agent that handles it, and its actual output, will show here.")
        else:
            latest = st.session_state.run_log[0]
            st.caption(f'"{latest["query"]}" · {latest["ts"]}')
            render_agent_details(latest["response"])

            if len(st.session_state.run_log) > 1:
                st.markdown("---")
                st.caption("Earlier runs")
                for entry in st.session_state.run_log[1:8]:
                    resp = entry["response"]
                    st.markdown(
                        f"""
                        <div class="sx-panel">
                            <div class="sx-meta">"{entry['query']}" · {entry['ts']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_agent_pipeline(resp.get("agents_executed"), resp.get("intent", "general"))


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

if st.session_state.user:
    render_dashboard()
elif st.session_state.page == "login":
    render_login()
else:
    render_home()
