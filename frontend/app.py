"""
StoxAI — Streamlit frontend

Flow: Home -> (Login/Sign up button -> modal) -> Dashboard (Chat / Portfolio / Agent activity)

Wired for real against main.py's actual contract:
  - Google OAuth redirect + JWT capture via ?token= query param
  - /me to fetch the logged-in user
  - /chat for every agent query

Run:
    pip install -r requirements.txt
    streamlit run app.py

Config:
    Create .streamlit/secrets.toml with:
        API_BASE_URL = "http://localhost:8000"
"""
from datetime import datetime

import streamlit as st

from api_client import google_login_url, fetch_me, send_chat, INTENT_TO_AGENT

st.set_page_config(page_title="StoxAI", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# Theme (matches your Nivesh AI mockups: dark navy + orange accent)
# --------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0b1120; color: #e2e8f0; }
.hero-title { font-size: 3rem; font-weight: 800; line-height: 1.15; color: #f8fafc; }
.hero-sub { color: #94a3b8; font-size: 1.05rem; max-width: 640px; }
.badge-pill {
    display: inline-block; background: rgba(148,163,184,0.12); color: #cbd5e1;
    padding: 4px 14px; border-radius: 999px; font-size: 0.85rem; margin-bottom: 1rem;
}
.feature-card {
    background: #131c31; border-left: 4px solid #f5a623; border-radius: 10px;
    padding: 1.1rem 1.3rem; height: 100%;
}
.feature-card h4 { margin: 0 0 .4rem 0; color: #f8fafc; }
.feature-card p { color: #94a3b8; font-size: 0.9rem; margin: 0; }
.run-card {
    background: #131c31; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
div.stButton > button[kind="primary"] {
    background-color: #f5a623; color: #111827; border: none; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    ("#f5a623", "News agent", "Fuses today's headlines with the model's odds into one explained call."),
    ("#a78bfa", "Planning agent", "Scores diversification and volatility, then suggests rebalancing."),
    ("#60a5fa", "Market prediction", "Estimates tomorrow's direction, retrained every night."),
    ("#f87171", "Short-selling agent", "Flags bearish calls and checks margin, entry, and stop levels."),
    ("#2dd4bf", "Tax agent", "Answers capital-gains questions grounded in current tax rules."),
    ("#f5a623", "Email digest", "A daily summary of verdicts, delivered before markets open."),
]

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("token", None)
st.session_state.setdefault("user", None)
st.session_state.setdefault("chat_history", [])   # [{role, content}]
st.session_state.setdefault("agent_runs", [])      # [{agent, color, headline, detail, ts}]
st.session_state.setdefault("portfolio", [])       # [PortfolioHolding dicts]

# Capture ?token=... coming back from /auth/google/callback
qp = st.query_params
if "token" in qp and not st.session_state.token:
    candidate = qp["token"]
    user = fetch_me(candidate)
    if user:
        st.session_state.token = candidate
        st.session_state.user = user
        st.query_params.clear()
        st.rerun()
    else:
        st.query_params.clear()
        st.error("Login didn't go through — token was rejected by /me. Check JWT_SECRET_KEY / clock skew on the backend.")


# --------------------------------------------------------------------------
# Login / sign up modal
# --------------------------------------------------------------------------
@st.dialog("Log in")
def login_dialog():
    st.caption("Welcome back to StoxAI.")
    st.link_button("Continue with Google", google_login_url(), use_container_width=True, type="primary")
    st.divider()
    st.caption(
        "The backend currently only supports Google sign-in (see auth/google/* "
        "routes in main.py). If you want username/password too, that route "
        "needs to be added server-side first."
    )
    with st.expander("Dev / demo login (no backend needed)"):
        st.caption("Useful for building the UI before your backend is deployed.")
        demo_name = st.text_input("Demo username", value="rahul_k")
        if st.button("Continue with demo user"):
            st.session_state.token = "DEV-DEMO-TOKEN"
            st.session_state.user = {"username": demo_name, "email": f"{demo_name}@demo.local", "photo": None}
            st.rerun()


# --------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------
def render_home():
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown("### 📈 **StoxAI**")
    with top_r:
        if st.button("Login / Sign up", type="primary", use_container_width=True):
            login_dialog()

    st.markdown('<div class="badge-pill">6 agents, one wishlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">One dashboard for the news,<br>the numbers, and the tax bill</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">StoxAI reads today\'s news, checks a prediction model\'s odds, '
        'and scores your diversification every morning — so all you have to do is ask.</p>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Get started", type="primary"):
        login_dialog()

    st.write("")
    st.write("")
    cols = st.columns(3)
    for i, (color, title, desc) in enumerate(FEATURES):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="feature-card" style="border-left-color:{color}">
                        <h4>{title}</h4><p>{desc}</p>
                    </div>""",
                unsafe_allow_html=True,
            )
            st.write("")


# --------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------
def log_run(intent: str, headline: str, detail: str):
    agent_name, color = INTENT_TO_AGENT.get(intent, ("Assistant", "#94a3b8"))
    st.session_state.agent_runs.insert(0, {
        "agent": agent_name, "color": color, "headline": headline,
        "detail": detail, "ts": datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state.agent_runs = st.session_state.agent_runs[:20]


def render_dashboard():
    user = st.session_state.user or {}
    header_l, header_r = st.columns([5, 1])
    with header_l:
        st.markdown(f"### Welcome back, **{user.get('username', 'trader')}**")
    with header_r:
        if st.button("Log out", use_container_width=True):
            for k in ("token", "user", "chat_history", "agent_runs"):
                st.session_state[k] = None if k in ("token", "user") else []
            st.rerun()

    tab_chat, tab_portfolio, tab_activity = st.tabs(["Chat", "Portfolio", "Agent activity"])

    # --- CHAT TAB ---
    with tab_chat:
        col_chat, col_side = st.columns([2, 1])

        with col_chat:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("agent"):
                        st.caption(f"🏷️ {msg['agent']}")

            prompt = st.chat_input("Ask about a stock, or how to improve your portfolio...")
            if prompt:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.spinner("Routing to the right agent..."):
                    try:
                        result = send_chat(
                            token=st.session_state.token,
                            user_query=prompt,
                            portfolio=st.session_state.portfolio,
                            tax_profile=None,
                            chat_history=st.session_state.chat_history[:-1],
                        )
                        answer = result.get("final_response", "(no response)")
                        intent = result.get("intent", "general")
                        agent_name, _ = INTENT_TO_AGENT.get(intent, ("Assistant", "#94a3b8"))
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "agent": agent_name}
                        )
                        log_run(intent, headline=answer[:60], detail=f"intent: {intent}")
                    except RuntimeError as e:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": f"⚠️ {e}"}
                        )
                st.rerun()

        with col_side:
            st.markdown("**RECENT RUNS**")
            if not st.session_state.agent_runs:
                st.caption("No agent runs yet — ask something in the chat.")
            for run in st.session_state.agent_runs[:5]:
                st.markdown(
                    f"""<div class="run-card">
                            <span style="color:{run['color']}; font-weight:700;">{run['agent']}</span><br>
                            <span style="font-size:0.85rem;">{run['headline']}</span><br>
                            <span style="color:#64748b; font-size:0.75rem;">{run['ts']} · {run['detail']}</span>
                        </div>""",
                    unsafe_allow_html=True,
                )

    # --- PORTFOLIO TAB ---
    with tab_portfolio:
        st.caption("Holdings here are sent along with every chat query as `portfolio` (see ChatRequest in main.py).")
        with st.form("add_holding", clear_on_submit=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            symbol = c1.text_input("Symbol", placeholder="TCS.NS")
            qty = c2.number_input("Quantity", min_value=0.0, step=1.0)
            buy_price = c3.number_input("Buy price", min_value=0.0, step=1.0)
            current_price = c4.number_input("Current price", min_value=0.0, step=1.0)
            asset_type = c5.selectbox("Type", ["equity", "mutual_fund", "etf", "bond", "crypto"])
            buy_date = st.date_input("Buy date")
            if st.form_submit_button("Add holding", type="primary") and symbol:
                st.session_state.portfolio.append({
                    "symbol": symbol.upper(), "quantity": qty, "buy_price": buy_price,
                    "buy_date": str(buy_date), "current_price": current_price, "asset_type": asset_type,
                })
        if st.session_state.portfolio:
            st.table(st.session_state.portfolio)
        else:
            st.info("No holdings added yet.")

    # --- AGENT ACTIVITY TAB ---
    with tab_activity:
        st.caption(
            "This log is built client-side from each /chat response's `intent`. "
            "It resets on refresh — persist it server-side (e.g. a `runs` "
            "collection in Mongo) if you want history across sessions/devices."
        )
        if not st.session_state.agent_runs:
            st.info("Nothing has run yet.")
        for run in st.session_state.agent_runs:
            st.markdown(
                f"""<div class="run-card">
                        <span style="color:{run['color']}; font-weight:700;">{run['agent']}</span>
                        <span style="color:#64748b; font-size:0.8rem;"> · {run['ts']}</span><br>
                        <b>{run['headline']}</b><br>
                        <span style="color:#94a3b8; font-size:0.85rem;">{run['detail']}</span>
                    </div>""",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
if st.session_state.token and st.session_state.user:
    render_dashboard()
else:
    render_home()
