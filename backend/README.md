# Merged Multi-Agent Financial Assistant

Hackathon auth/stock/email/db codebase + tax_saving_agent's LangGraph
flow, merged into one deployable FastAPI service. `rag.py` is untouched,
exactly as instructed.

## ⚠️ Rotate your secrets first

The `.env` values pasted into chat (Mongo URI+password, Google client
secret, JWT secret, Brevo key) are now exposed and should be treated as
compromised. Before deploying: rotate the Mongo Atlas DB user password,
regenerate the Google OAuth client secret, pick a new random
`JWT_SECRET_KEY`, and regenerate the Brevo API key. Put the _new_ values
only in Render's environment variable settings, never in a committed file.
`.env.example` in this bundle has placeholders only.

## What changed in this pass

- **Short selling is removed completely** - `short_selling.py`, its node,
  its `AgentState` fields, and its router keywords are gone.
- **Email is Brevo-only.** `agents.send_email_api` (Brevo's transactional
  email HTTP API, not SMTP) is the only mailer left in the codebase.
  SendGrid/Mailgun/SMTP and the multi-provider fallback chain are gone.
- **Email and the news agent are not called from any user chat flow.**
  Both still work, but only if _you_ invoke them (see "News agent"
  below). No user-facing node sends an email or fetches news anymore.
- **News agent (`news_agent.py`)** fetches recent finance news for a
  symbol and ingests it straight into the same RAG corpus `rag.py`
  builds from `data/tax_docs/*.md`, via the unmodified `rag.add_doc()`.
  It is triggered manually or on a schedule - never by a user prompt.
- **User questions that don't match a specialized agent** are now
  answered by the model's own general knowledge (`chatbot_response_node`
  calls the Anthropic model directly with a "you don't have live data"
  system prompt), instead of a flat "I don't understand" message.
- **The per-user stock watchlist now lives on the same `users` Mongo
  document as the rest of the profile** (a `stocks` field, side by side
  with `email`/`username`/etc.), instead of a separate `user_stocks`
  collection - see `database.py` / `agents.py` Section 3.
- **The `-1` "not predicted yet" sentinel is resolved in one place**:
  `agents.resolve_price(symbol, cached)`. Every agent that needs "the
  price to use" for a watchlist symbol goes through it - if
  `predicted_price` is the sentinel, it fetches a live price instead of
  treating `-1` as a real number.

## File layout

```
config.py         every env-driven setting, both codebases merged
database.py       one MongoClient, one `users` collection (profile + watchlist)
auth_utils.py     JWT create/verify (unchanged)
agents.py         stock prediction (LSTM+GRU) + Brevo email + watchlist
                   (was stock_agent.py + email_agent.py + db_agent.py)
core.py           schemas (AgentState) + tax rules + LLM explain
news_agent.py     finance-news fetch -> RAG ingestion. Manual/cron only.
rag.py            UNCHANGED - tax-doc (+ ingested news) retrieval
graph.py          all LangGraph nodes + router + checkpointer + build_graph()
main.py           FastAPI app: Google OAuth/JWT + POST /chat +
                   POST /admin/ingest-news, one service
models/           lstm_model.keras, gru_model.keras (renamed/moved)
data/tax_docs/    markdown knowledge base for rag.py (news gets added here too)
sample_data/      sample_portfolio.json, for graph.py's __main__ smoke test
requirements.txt  merged dependencies
.env.example      placeholders only
```

## The conversation flow

`router_node` (in `graph.py`) checks, in order:

0. **`state["pending_action"] == "await_stock_list"`** - set by the
   planning node when a user has no watchlist yet. The very next message
   is routed straight to `save_interest_node`, whatever it says.
1. **Tax keywords** (tax, 80c, ltcg, deduction, harvest) → `tax_saving_node`
   - unchanged: rules engine + tax-doc RAG + LLM explanation.
2. **"How's X doing" / a bare ticker** → `stock_info_node` - live price
   (yfinance) + a prediction (cached watchlist price via
   `agents.resolve_price`, or a fresh LSTM/GRU run if it's not on the
   watchlist). No news fetch.
3. **"My favorite stocks are..." / "I'm interested in..."** →
   `save_interest_node` - regex-extracts tickers, upserts them onto the
   user's own document with `predicted_price = -1` (sentinel: "not
   predicted yet").
4. **"How do I improve my portfolio" / "which stocks should I profit
   from"** → `planning_node`:
   - no watchlist → sets `pending_action="await_stock_list"` and asks the
     user to list stocks; **you then re-ask the portfolio question
     yourself** on your next turn.
   - watchlist exists → per symbol: `agents.resolve_price()` (cached
     prediction, or live price if it's still the `-1` sentinel) → an LLM
     pass turns that into a short plain-language note. No email, no news.
5. **"predict" / "forecast"** → `stock_prediction_node` - runs
   `agents.predict_stock()` (LSTM+GRU) for every symbol in
   `state["portfolio"]`. If a predicted symbol is also on the user's
   watchlist, the fresh prediction is cached back onto their `users`
   document (clearing the `-1` sentinel for next time).
6. **"report"** → `report_generation_node` (still a stub, out of scope
   here).
7. anything else → `chatbot_response_node`, which answers from the
   model's own general knowledge - it never triggers a news fetch or any
   other side effect.

## News agent - manual/scheduled only, never from user prompts

`news_agent.py` fetches recent company news for one or more symbols from
Finnhub's `/api/v1/company-news` endpoint and writes it into the RAG corpus
via the unmodified `rag.add_doc()`. There is no email step
and no `AgentState` field for it - it is completely decoupled from `/chat`.

Trigger it in one of three ways:

```bash
# 1. CLI - run locally, or as a one-off Render Shell command
python news_agent.py AAPL TSLA INFY

# 2. Render Cron Job - a separate Render service type that runs the same
#    command on a schedule (e.g. daily). This is the recommended way to
#    keep the RAG corpus fresh in production without touching user traffic.

# 3. Admin-only HTTP route (does NOT use the user JWT at all):
curl -X POST https://your-backend.onrender.com/admin/ingest-news \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "TSLA"]}'
```

Set `ADMIN_API_KEY` in your environment to enable route #3; leave it
blank to have that route return 403 and only use #1/#2.

Set `FINNHUB_API_KEY` in the environment. The client sends the ticker,
UTC `from`/`to` dates, and Finnhub token, then maps Finnhub's `headline`,
`summary`, `source`, `url`, and Unix `datetime` fields into the RAG document.

## Checkpointer / threads

`build_graph()` compiles with a checkpointer (`config.CHECKPOINTER_BACKEND`,
default `memory` = LangGraph's `MemorySaver`). `main.py`'s `/chat` route
uses the authenticated user's Mongo `_id` as `thread_id`, so
`pending_action` and prior turns persist **within a running process**.

**Important limitation:** `MemorySaver` is in-process memory - a Render
restart/redeploy wipes every in-flight thread (a user mid-way through
"list your stocks" would need to start that turn over). This is expected
and fine for Render's free/small tiers, which have no persistent disk
anyway. For real persistence across restarts, add
`langgraph-checkpoint-sqlite` (with a Render persistent disk) or
`langgraph-checkpoint-postgres`, and set
`CHECKPOINTER_BACKEND=sqlite`/`postgres` - `graph.py`'s
`_make_checkpointer()` already has the sqlite branch stubbed in, swap the
import when you add the package.

## What I could and couldn't verify here

This sandbox has no network access and doesn't have `fastapi`, `pymongo`,
`langgraph`, `keras`/`tensorflow`, `yfinance`, etc. installed, so I
**could not** actually run `python graph.py` or hit `/chat` with FastAPI's
TestClient. I ran `python -m py_compile` on every changed file (all pass)
and manually traced every `config.<X>` reference back to a definition in
`config.py`. Please run these two locally before deploying:

```bash
pip install -r requirements.txt
cp _env.example .env   # fill in real (rotated) values
python graph.py                       # smoke test, prints the final state as JSON
uvicorn main:app --reload --port 8000 # then POST /chat with a Bearer token from /auth/google/callback
```

## Deploying: Render (backend) + Streamlit (frontend), no CORS/domain surprises

1. **Deploy the backend to Render first** (Web Service,
   `uvicorn main:app --host 0.0.0.0 --port $PORT`). Note the resulting URL,
   e.g. `https://your-backend.onrender.com`.
2. **Deploy the Streamlit frontend** (Render or Streamlit Community
   Cloud). Note its exact URL, e.g. `https://your-app.streamlit.app` -
   **no trailing slash**.
3. Back in the Render backend's Environment tab, set:
   - `FRONTEND_URL=https://your-app.streamlit.app` (no trailing slash -
     `config.py` also strips one automatically as a safety net, but don't
     rely on that).
   - `GOOGLE_REDIRECT_URI=https://your-backend.onrender.com/auth/google/callback`
     - and add that exact URI to the "Authorized redirect URIs" list in
       Google Cloud Console for this OAuth client, or the OAuth callback
       will fail with `redirect_uri_mismatch`.
   - Every other var from `.env.example` (rotated values).
   - If your frontend needs to be reachable from more than one origin
     (e.g. a local dev URL too), set `API_CORS_ORIGINS` explicitly as a
     comma-separated list instead of relying on `FRONTEND_URL` alone.
4. **Never set `API_CORS_ORIGINS` (or leave `FRONTEND_URL`) as `*`.** The
   backend sends `allow_credentials=True` for the Bearer-token flow, and
   browsers silently reject a wildcard origin combined with credentials -
   requests will fail with an opaque CORS error in devtools, not a clear
   4xx. `main.py` logs a warning at startup if it detects this
   misconfiguration.
5. In the Streamlit frontend, point every API call at the Render backend
   URL from step 1 (e.g. via a `BACKEND_URL` secret/env var in Streamlit's
   own settings) - don't hardcode `localhost`.
6. `models/*.keras` (≈5MB total) should ship in the repo/image - they're
   loaded lazily on first prediction (`agents._get_models()`), not at
   import time, so cold start isn't paying for them until the first
   stock-info/prediction request.
7. If you want the news agent to run automatically in production, add a
   **Render Cron Job** service that runs
   `python news_agent.py TICKER1 TICKER2 ...` on a schedule, pointed at
   the same repo/env vars as the backend - it does not need to be part of
   the web service.
8. If you want persistence for the checkpointer across restarts, add a
   Render persistent disk and use the sqlite checkpointer (see above).
