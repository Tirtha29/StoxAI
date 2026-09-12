"""
Single FastAPI app for the whole merged product: manual email/username/
password auth + JWT, and the multi-agent /chat endpoint, mounted together.

AUTH CHANGE (from Google OAuth to manual signup/login):
  Google OAuth was throwing errors during testing, so this now uses plain
  email + username + password instead. auth_utils.py (JWT creation and
  verification) is COMPLETELY UNCHANGED — it only ever worked off a
  user_id + email, never cared how the user got authenticated, so nothing
  there needed to change. /me, /chat, and /admin/ingest-news are also
  unchanged below — they only consume the JWT, same as before.

  Passwords are hashed with bcrypt before storage — never store or log a
  raw password. Existing users created via the old Google flow (if any
  are in your Mongo already) have no password_hash and can't log in
  through this new flow; they'd need to sign up fresh.

WHY ONE SERVICE INSTEAD OF TWO:
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

Run:
    uvicorn main:app --reload --port 8000
"""

from typing import List, Optional

import logging

import bcrypt
from bson import ObjectId
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
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

# compiled once at startup with a checkpointer attached - see graph.py's
# _make_checkpointer() for the in-memory-vs-sqlite/postgres tradeoff
_graph_app = build_graph()


# ===========================================================================
# Password hashing helpers
# ===========================================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # malformed/missing hash on the user doc (e.g. a leftover Google-only
        # account with no password_hash at all) - treat as "wrong password"
        return False


# ===========================================================================
# Auth routes — manual email + username + password
# ===========================================================================

@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


class SignupRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    email: str


@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    username = req.username.strip()

    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if not username:
        raise HTTPException(status_code=400, detail="Username can't be empty")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user_doc = {
        "email": email,
        "username": username,
        "password_hash": hash_password(req.password),
        "photo": None,
    }
    result = users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)

    token = create_jwt_token(user_id=user_id, email=email)
    return AuthResponse(token=token, user_id=user_id, username=username, email=email)


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    email = req.email.strip().lower()
    user = users_collection.find_one({"email": email})

    if not user or not user.get("password_hash") or not verify_password(req.password, user["password_hash"]):
        # deliberately the same error for "no such user" and "wrong password" -
        # don't leak which one it was
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    user_id = str(user["_id"])
    token = create_jwt_token(user_id=user_id, email=email)
    return AuthResponse(token=token, user_id=user_id, username=user.get("username", ""), email=email)


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
