"""
Thin client around the FastAPI backend in Files_required_for_scheduler.zip.
Matches main.py exactly:
  GET  /auth/google/login          -> browser redirect to Google, ends up back
                                       at FRONTEND_URL?token=<jwt>
  GET  /me            (Bearer JWT) -> {user_id, username, email, photo}
  POST /chat          (Bearer JWT) -> ChatResponse (see main.py ChatResponse)

Nothing here is invented — if a backend route doesn't exist yet (e.g. a
persisted agent-run log), it is NOT called from this file.
"""
import requests
import streamlit as st

API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30


def google_login_url() -> str:
    return f"{API_BASE_URL}/auth/google/login"


def fetch_me(token: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


def send_chat(
    token: str,
    user_query: str,
    portfolio: list[dict],
    tax_profile: dict | None,
    chat_history: list[dict],
) -> dict:
    """
    Mirrors ChatRequest / ChatResponse in main.py exactly.
    Raises RuntimeError with a readable message on failure so the UI can
    show it instead of crashing.
    """
    payload = {
        "user_query": user_query,
        "portfolio": portfolio,
        "tax_profile": tax_profile,
        "chat_history": chat_history,
    }
    try:
        r = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {API_BASE_URL}: {e}")

    if r.status_code == 401:
        raise RuntimeError("Session expired — please log in again.")
    if r.status_code != 200:
        raise RuntimeError(f"Backend error {r.status_code}: {r.text[:300]}")

    return r.json()


# Friendly display names for the `intent` field the router returns.
# This mapping is a UI-only guess to match your "agent" mockups — the
# backend itself just calls these LangGraph node names.
INTENT_TO_AGENT = {
    "tax_optimization": ("Tax agent", "#2dd4bf"),
    "save_interest": ("Watchlist agent", "#a78bfa"),
    "portfolio_planning": ("Planning agent", "#a78bfa"),
    "stock_prediction": ("Market prediction", "#60a5fa"),
    "stock_info": ("News agent", "#f5a623"),
    "report": ("Report agent", "#f87171"),
    "general": ("Assistant", "#94a3b8"),
}
