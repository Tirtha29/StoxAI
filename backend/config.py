"""
Single config module for the whole merged app (auth + tax-saving agent +
news ingestion + RAG + stock prediction + email). Every tunable lives
here, read from environment variables only - nothing is hardcoded, so
this file is safe to commit even though the merged app talks to Google,
MongoDB, Groq, Brevo, etc.

Deploying on Render: set ALL of these as Environment Variables in the
Render dashboard for the service (do not commit a real .env). Locally,
copy `.env.example` to `.env`.

SECURITY NOTE: earlier values pasted in chat for MONGODB_URI,
GOOGLE_CLIENT_SECRET, JWT_SECRET_KEY and BREVO_API_KEY should be treated
as compromised (they were shared in plaintext) - rotate all four before
deploying, then set the *new* values here via env vars only.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _strip_slash(url: str) -> str:
    return url.rstrip("/") if url else url


def _list(name: str, default: str = "") -> list:
    """Comma-separated env list, trimmed and with trailing slashes removed
    from anything that looks like a URL - avoids the classic
    'https://foo.onrender.com/' != 'https://foo.onrender.com' CORS bug."""
    raw = os.getenv(name, default)
    return [_strip_slash(x.strip()) for x in raw.split(",") if x.strip()]


# ===========================================================================
# Auth / Google OAuth / JWT  (formerly Hackathon/backend/config.py)
# ===========================================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
FRONTEND_URL = _strip_slash(os.getenv("FRONTEND_URL"))  # Streamlit app URL

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))

# ===========================================================================
# MongoDB - one cluster, one DB. Per your request, the per-user stock
# watchlist now lives INSIDE the same `users` document as the rest of the
# user's profile (a "stocks" field, side by side with everything else)
# instead of a separate collection - see database.py.
# ===========================================================================

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "auth_app")

# RAG vector-search collection/index (rag.py talks to these directly)
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "agenticvectordb")
MONGODB_VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")

# ===========================================================================
# LLM providers - Groq (fast, used by the react agents / planning synth) +
# Anthropic (used by core.py's grounded explanation and the
# general-intelligence fallback in graph.py's chatbot node)
# ===========================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # core.py / graph.py call this LLM_API_KEY
LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ===========================================================================
# Email - Brevo ONLY. No SMTP/SendGrid/Mailgun fallback chain anymore.
# `agents.send_email_api` is the single place that sends mail, and it is
# NOT called from any user-facing chat node - it stays available in the
# codebase for whoever wires up a notification later (see agents.py).
# ===========================================================================

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_NAME = os.getenv("SENDER_NAME", "App Notifier")

# ===========================================================================
# Tax-saving rules engine
# ===========================================================================

MIN_HOLDING_DAYS_BEFORE_HARVEST_SUGGESTION = int(os.getenv("MIN_HOLDING_DAYS_BEFORE_HARVEST_SUGGESTION", "30"))
LTCG_HOLDING_DAYS = int(os.getenv("LTCG_HOLDING_DAYS", "365"))
SECTION_80C_LIMIT = float(os.getenv("SECTION_80C_LIMIT", "150000"))
US_401K_LIMIT = float(os.getenv("US_401K_LIMIT", "23000"))
US_IRA_LIMIT = float(os.getenv("US_IRA_LIMIT", "7000"))

# ===========================================================================
# RAG (rag.py) - untouched, still the tax-doc corpus. news_agent.py also
# writes into this same corpus (TAX_DOCS_DIR) when it's manually run, so
# ingested news becomes retrievable the same way tax docs are.
# ===========================================================================

TAX_DOCS_DIR = os.getenv("TAX_DOCS_DIR", os.path.join(os.path.dirname(__file__), "data", "tax_docs"))
RAG_DENSE_BACKEND = os.getenv("RAG_DENSE_BACKEND", "none")  # none | faiss | mongodb
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(os.path.dirname(__file__), "data", "faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ===========================================================================
# News agent (news_agent.py) - finance-news fetch + RAG ingestion.
# IMPORTANT: this is NEVER triggered by a user chat message. It only runs
# when *we* invoke it: `python news_agent.py TICKER [TICKER ...]`, a
# Render Cron Job, or the admin-only /admin/ingest-news route in main.py
# (guarded by ADMIN_API_KEY below). No email is sent from this path.
# ===========================================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_API_BASE_URL = os.getenv("FINNHUB_API_BASE_URL", "https://finnhub.io/api/v1")
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))
NEWS_ARTICLES_PER_SYMBOL = int(os.getenv("NEWS_ARTICLES_PER_SYMBOL", "10"))

# Shared secret required in an `X-Admin-Key` header to hit /admin/* routes
# in main.py. Leave unset locally if you only ever trigger news_agent.py
# from the CLI/cron and never expose the HTTP route.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# ===========================================================================
# Stock prediction (LSTM/GRU models)
# ===========================================================================

STOCK_MODEL_DIR = os.getenv("STOCK_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
STOCK_PREDICTION_WINDOW = int(os.getenv("STOCK_PREDICTION_WINDOW", "99"))
# Sentinel meaning "no prediction cached yet". Any code that reads a
# cached predicted_price MUST check for this and fall back to a live
# price lookup instead (see agents.resolve_price()). Keep as -1 (never a
# valid price).
NO_PREDICTION_SENTINEL = -1

# ===========================================================================
# Remote-agent URLs (only used if you split an agent into its own service)
# ===========================================================================

AGENT_CALL_TIMEOUT_SECONDS = int(os.getenv("AGENT_CALL_TIMEOUT_SECONDS", "20"))
AGENT_STOCK_PREDICTION_URL = os.getenv("AGENT_STOCK_PREDICTION_URL")
AGENT_REPORT_GENERATION_URL = os.getenv("AGENT_REPORT_GENERATION_URL")

# ===========================================================================
# API / CORS
# ===========================================================================

# Comma-separated list of exact origins allowed to call this API (no
# trailing slashes, no wildcards - wildcards silently break because
# allow_credentials=True is also set, which browsers reject when paired
# with Access-Control-Allow-Origin: *). On Render, set this to your
# Streamlit app's exact URL(s), e.g.:
#   API_CORS_ORIGINS=https://your-app.streamlit.app,http://localhost:8501
API_CORS_ORIGINS = _list("API_CORS_ORIGINS", FRONTEND_URL or "")

# ===========================================================================
# LangGraph checkpointing
# ===========================================================================

# "memory" = in-process MemorySaver (fine for a demo, does NOT survive a
# Render dyno restart / redeploy - all in-flight threads are lost). Set to
# "sqlite" once you add `langgraph-checkpoint-sqlite` and a persistent
# disk, or "postgres" with `langgraph-checkpoint-postgres` for real
# durability across restarts. On Render's free tier there is no
# persistent disk, so "memory" is the practical default there - the
# `pending_action` "list your stocks" hand-off just resets on redeploy,
# which is an acceptable tradeoff for the current scope.
CHECKPOINTER_BACKEND = os.getenv("CHECKPOINTER_BACKEND", "memory")
CHECKPOINTER_SQLITE_PATH = os.getenv("CHECKPOINTER_SQLITE_PATH", "checkpoints.sqlite")
