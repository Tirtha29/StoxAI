"""
Single FastAPI app for the whole merged product: Google OAuth + JWT auth
(from Hackathon/backend/main.py) and the multi-agent /chat endpoint (from
tax_saving_agent/api.py), mounted together.

WHY ONE SERVICE INSTEAD OF TWO (task 4's "or tell me clearly"):
  - You're deploying to Render as a hackathon/small-team project, and the
    Streamlit frontend just needs one base URL + one CORS origin to talk
    to. Two services means two Render web services, two sets of secrets,
    and CORS/cookie wiring between them for basically no benefit here -
    /chat already re-checks the JWT itself, there's no separate trust
    boundary being enforced by splitting them.
  - The tradeoff you're accepting: /chat pulls in heavy deps (keras/
    tensorflow, yfinance, sentence-transformers) into the SAME process as
    auth, so a cold start on Render's free/small tiers will be slower for
    login too, and a crash in model loading takes auth down with it.
  - Split them again later (same api.py you already have, unchanged) if
    /chat's traffic or resource needs grow enough to want independent
    scaling - nothing here prevents that, call_remote_agent() in graph.py
    already exists for exactly that kind of split.

Run:
    uvicorn main:app --reload --port 8000
"""

from typing import List, Optional

import logging

import httpx
from urllib.parse import urlencode
from bson import ObjectId
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import config
from database import users_collection
from auth_utils import create_jwt_token, get_current_user
from graph import build_graph
from core import PortfolioHolding, TaxProfile
from news_agent import ingest_news_for_symbols

logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Multi-Agent API")

# --- CORS -------------------------------------------------------------
# config.API_CORS_ORIGINS is built from FRONTEND_URL / API_CORS_ORIGINS
# and already strips trailing slashes, which is the #1 cause of a CORS
# failure that looks identical to a misconfigured origin (Render URL vs
# Streamlit Cloud URL mismatch). allow_credentials=True means the origin
# list can NEVER contain "*" - browsers reject that combination outright,
# so on Render set API_CORS_ORIGINS (or FRONTEND_URL) to the EXACT
# https://your-app.streamlit.app URL, no trailing slash.
if not config.API_CORS_ORIGINS:
    logger.warning(
        "No FRONTEND_URL / API_CORS_ORIGINS configured - the browser will "
        "block every request from your Streamlit frontend until you set "
        "one of these env vars to your frontend's exact origin."
    )
elif "*" in config.API_CORS_ORIGINS:
    logger.warning(
        "API_CORS_ORIGINS contains '*' - this is ignored by browsers when "
        "allow_credentials=True. Set it to your exact frontend origin(s) instead."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# compiled once at startup with a checkpointer attached - see graph.py's
# _make_checkpointer() for the in-memory-vs-sqlite/postgres tradeoff
_graph_app = build_graph()


# ===========================================================================
# Auth routes (unchanged behavior from Hackathon/backend/main.py)
# ===========================================================================

@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/google/login")
def google_login():
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)


@app.get("/auth/google/callback")
def google_callback(code: str = None, error: str = None):
    if error:
        return RedirectResponse(f"{config.FRONTEND_URL}?error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    token_data = {
        "code": code,
        "client_id": config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    token_res = httpx.post(GOOGLE_TOKEN_URL, data=token_data)
    if token_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to get token from Google")
    token_json = token_res.json()
    google_access_token = token_json.get("access_token")
    google_refresh_token = token_json.get("refresh_token")

    userinfo_res = httpx.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {google_access_token}"},
    )
    if userinfo_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to get user info from Google")
    userinfo = userinfo_res.json()

    email = userinfo.get("email")
    if not email or not email.endswith("@gmail.com"):
        return RedirectResponse(f"{config.FRONTEND_URL}?error=gmail_only")

    google_id = userinfo.get("id")
    username = userinfo.get("name")
    photo = userinfo.get("picture")

    existing_user = users_collection.find_one({"google_id": google_id})

    update_fields = {
        "username": username,
        "email": email,
        "photo": photo,
        "google_access_token": google_access_token,
    }
    if google_refresh_token:
        update_fields["google_refresh_token"] = google_refresh_token

    if existing_user:
        users_collection.update_one({"google_id": google_id}, {"$set": update_fields})
        user_id = str(existing_user["_id"])
    else:
        update_fields["google_id"] = google_id
        result = users_collection.insert_one(update_fields)
        user_id = str(result.inserted_id)

    jwt_token = create_jwt_token(user_id=user_id, email=email)

    return RedirectResponse(f"{config.FRONTEND_URL}?token={jwt_token}")


@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    user = users_collection.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": str(user["_id"]),
        "username": user.get("username"),
        "email": user.get("email"),
        "photo": user.get("photo"),
    }


# ===========================================================================
# /chat route (was tax_saving_agent/api.py - mounted directly here)
# ===========================================================================

class ChatRequest(BaseModel):
    user_query: str
    portfolio: List[PortfolioHolding] = []
    tax_profile: Optional[TaxProfile] = None
    chat_history: List[dict] = []


class ChatResponse(BaseModel):
    intent: Optional[str] = None
    final_response: str
    tax_result: Optional[dict] = None
    stock_prediction_result: Optional[list] = None
    stock_info_result: Optional[dict] = None
    save_interest_result: Optional[dict] = None
    planning_result: Optional[dict] = None
    report_result: Optional[dict] = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Requires a valid JWT (same Bearer token /me uses) - user_id/email come
    from the token, not the request body, so the watchlist and portfolio-
    planning flow are always scoped to the caller. thread_id = user_id, so
    LangGraph's checkpointer keeps one continuous conversation per user
    (this is what makes the "list your stocks" -> next-turn save_interest
    hand-off in graph.py work across separate HTTP requests).
    """
    initial_state = {
        "user_query": req.user_query,
        "chat_history": req.chat_history,
        "portfolio": [p.model_dump(mode="json") for p in req.portfolio],
        "tax_profile": req.tax_profile.model_dump(mode="json") if req.tax_profile else {},
        "user_id": current_user["user_id"],
        "user_email": current_user.get("email"),
    }

    try:
        result = _graph_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": current_user["user_id"]}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"graph execution failed: {e}")

    return ChatResponse(
        intent=result.get("intent"),
        final_response=result.get("final_response", ""),
        tax_result=result.get("tax_result"),
        stock_prediction_result=result.get("stock_prediction_result"),
        stock_info_result=result.get("stock_info_result"),
        save_interest_result=result.get("save_interest_result"),
        planning_result=result.get("planning_result"),
        report_result=result.get("report_result"),
    )


# ===========================================================================
# Admin-only route: manually trigger a news fetch + RAG ingestion.
#
# This is the ONLY way (besides the news_agent.py CLI / a Render Cron Job
# running that same CLI) that a news fetch happens - it is never called
# from /chat or from any user-facing node in graph.py. It does not accept
# a user JWT; it's guarded by a separate shared secret so it isn't
# reachable by a logged-in end user at all.
# ===========================================================================

class IngestNewsRequest(BaseModel):
    symbols: List[str]


@app.post("/admin/ingest-news")
def admin_ingest_news(req: IngestNewsRequest, x_admin_key: str = Header(None)):
    if not config.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="ADMIN_API_KEY is not configured on the server")
    if x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header")

    return ingest_news_for_symbols(req.symbols)
