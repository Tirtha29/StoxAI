# StoxAI — Multi-Agent Financial Assistant

StoxAI is a merged financial-assistant application combining the hackathon
authentication, stock, watchlist, database, and email components with the
`tax_saving_agent` LangGraph flow.

The application is split into:

- **FastAPI backend** — authentication, MongoDB, LangGraph agents, stock
  prediction, RAG, news ingestion, and API routes.
- **Streamlit frontend** — Home, Dashboard, Chat, Google login, and the
  user-facing interface.

`rag.py` remains untouched as required.

---

## ⚠️ Security — Rotate Secrets Before Deployment

Any real secrets previously shared in plaintext should be treated as
compromised.

Before deploying:

- Rotate the MongoDB Atlas database-user password.
- Regenerate the Google OAuth client secret.
- Generate a new random `JWT_SECRET_KEY`.
- Regenerate the Groq API key if it was exposed.
- Regenerate the Brevo API key.
- Regenerate the Finnhub API key if it was exposed.
- Update all rotated values in your local `.env` and/or Render Environment
  Variables.

**Never commit `.env` to GitHub.**

Use `.env.example` only for placeholders.

---

# 1. Project Structure

Recommended repository structure:

```text
StoxAI/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── auth_utils.py
│   ├── agents.py
│   ├── core.py
│   ├── graph.py
│   ├── rag.py
│   ├── news_agent.py
│   ├── scheduler.py
│   ├── requirements.txt
│   ├── .env                  # local only — DO NOT COMMIT
│   ├── models/
│   │   ├── lstm_model.keras
│   │   └── gru_model.keras
│   ├── data/
│   │   └── tax_docs/
│   └── sample_data/
│       └── sample_portfolio.json
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── secrets.toml      # local only — DO NOT COMMIT
│
├── .gitignore
└── README.md
```

### Main backend files

| File | Purpose |
|---|---|
| `config.py` | Central environment-variable configuration |
| `database.py` | MongoDB client and `users` collection |
| `auth_utils.py` | JWT creation and verification |
| `agents.py` | Stock prediction, watchlist, and Brevo email |
| `core.py` | Agent state, schemas, tax rules, and LLM explanation |
| `graph.py` | LangGraph nodes, router, checkpointer, and graph |
| `rag.py` | Tax-document and ingested-news retrieval |
| `news_agent.py` | Finnhub finance-news ingestion into RAG |
| `scheduler.py` | Scheduled/background tasks |
| `main.py` | FastAPI application and API routes |

---

# 2. Backend Features

## Authentication

- Google OAuth login.
- JWT-based authentication after successful Google login.
- User profile stored in MongoDB.
- Google profile information can include:
  - Google ID
  - Email
  - Username
  - Photo
  - OAuth tokens

## Stock Watchlist

The per-user watchlist is stored directly inside the user's MongoDB
document in the `stocks` field.

Example:

```json
{
  "email": "user@example.com",
  "username": "User",
  "stocks": {
    "AAPL": {
      "live_price": 227.5,
      "predicted_price": 231.2,
      "updated_at": "2026-09-11T12:00:00+00:00"
    }
  }
}
```

There is no separate `user_stocks` collection.

## Stock Prediction

The application uses LSTM + GRU models for stock-price prediction.

Models:

```text
models/lstm_model.keras
models/gru_model.keras
```

Models are loaded lazily when prediction is required.

A cached `predicted_price` of `-1` means:

> prediction has not been generated yet.

`agents.resolve_price()` handles this sentinel centrally and falls back to
a live price instead of treating `-1` as a real price.

## Tax-Saving Agent

Tax-related questions are handled through the LangGraph tax flow using:

- Tax rules.
- Tax-document RAG.
- LLM-based explanation.

Relevant examples include:

- Tax deductions
- Section 80C
- LTCG
- Tax harvesting

## RAG

The existing `rag.py` implementation is kept unchanged.

The main knowledge base is:

```text
data/tax_docs/
```

The news agent can also add finance-news documents to the same RAG corpus.

## Finance News

`news_agent.py` uses Finnhub's company-news endpoint to fetch recent
finance news for selected stock symbols.

News ingestion is intentionally decoupled from user chat.

It can be triggered:

1. Manually from the CLI.
2. Through a scheduled job.
3. Through the admin-only `/admin/ingest-news` endpoint.

## Email

Email is **Brevo-only**.

The application does not use:

- SMTP
- SendGrid
- Mailgun
- Multi-provider fallback chains

The Brevo email function remains available for future notification wiring,
but email is not triggered by normal user chat.

---

# 3. Conversation / Agent Flow

The LangGraph router checks user requests in this order:

### 0. Pending stock-list action

If:

```text
state["pending_action"] == "await_stock_list"
```

the next message is routed to `save_interest_node`.

This occurs when the planning agent needs the user's watchlist.

### 1. Tax questions

Keywords such as:

```text
tax
80c
ltcg
deduction
harvest
```

route to:

```text
tax_saving_node
```

### 2. Stock information

Requests such as:

```text
How's AAPL doing?
AAPL
```

route to:

```text
stock_info_node
```

This uses:

- Live price from `yfinance`.
- Cached prediction when available.
- A fresh LSTM/GRU prediction when necessary.

News is not fetched as part of this user flow.

### 3. Save stock interests

Requests such as:

```text
My favorite stocks are AAPL, TSLA
I'm interested in INFY
```

route to:

```text
save_interest_node
```

The detected tickers are saved to the user's MongoDB document with:

```text
predicted_price = -1
```

### 4. Portfolio planning

Requests such as:

```text
How do I improve my portfolio?
Which stocks should I profit from?
```

route to:

```text
planning_node
```

If the user has no watchlist, the node asks them to provide stocks and sets:

```text
pending_action = await_stock_list
```

If a watchlist exists, the agent resolves the appropriate price for each
symbol and generates a short plain-language note.

### 5. Predictions

Requests containing:

```text
predict
forecast
```

route to:

```text
stock_prediction_node
```

The LSTM/GRU prediction is generated for the portfolio.

If the symbol is also in the user's watchlist, the fresh prediction is cached
back into the MongoDB user document.

### 6. Reports

Requests containing:

```text
report
```

route to:

```text
report_generation_node
```

The report-generation implementation is currently a stub.

### 7. General questions

Everything else is routed to:

```text
chatbot_response_node
```

The model answers using its general knowledge and does not trigger news
ingestion or other side effects.

---

# 4. News Agent

Run the news agent manually:

```bash
python news_agent.py AAPL TSLA INFY
```

The agent:

1. Fetches recent company news from Finnhub.
2. Converts the returned information into RAG documents.
3. Adds the documents using the existing `rag.add_doc()` flow.

The admin HTTP endpoint is:

```text
POST /admin/ingest-news
```

Example:

```bash
curl -X POST https://your-backend.onrender.com/admin/ingest-news \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "TSLA"]}'
```

`ADMIN_API_KEY` must be configured for this route. If it is not configured,
the route remains disabled.

---

# 5. Checkpointer / Conversation Threads

The graph uses a configurable checkpointer.

Default:

```env
CHECKPOINTER_BACKEND=memory
```

`MemorySaver` keeps conversation state within the running process.

The `/chat` route uses the authenticated user's MongoDB `_id` as the
LangGraph `thread_id`.

### Important limitation

`MemorySaver` is in-process memory.

A Render restart or redeploy clears in-flight conversation threads.

For persistent state across restarts, use:

```text
langgraph-checkpoint-sqlite
```

with a persistent Render disk, or:

```text
langgraph-checkpoint-postgres
```

and configure the corresponding backend.

---

# 6. Backend Environment Variables

Create a local:

```text
backend/.env
```

Do not commit it.

The backend configuration reads environment variables through `config.py`.

Important variables include:

```env
# Google OAuth / JWT
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=
FRONTEND_URL=

JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080

# MongoDB
MONGODB_URI=
MONGODB_DB_NAME=auth_app
MONGODB_COLLECTION=agenticvectordb
MONGODB_VECTOR_INDEX=vector_index

# LLM
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=

# Brevo
BREVO_API_KEY=
SENDER_EMAIL=
SENDER_NAME=App Notifier

# Finnhub
FINNHUB_API_KEY=
FINNHUB_API_BASE_URL=https://finnhub.io/api/v1
NEWS_LOOKBACK_DAYS=7
NEWS_ARTICLES_PER_SYMBOL=10

# Admin
ADMIN_API_KEY=

# RAG
RAG_DENSE_BACKEND=none

# Checkpointer
CHECKPOINTER_BACKEND=memory
```

Additional optional configuration variables are defined in `config.py`.

---

# 7. Run the Backend Locally

Open a terminal in:

```text
backend/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure `.env` contains the required local values.

Start FastAPI:

```bash
python -m uvicorn main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

---

# 8. Run the Frontend Locally

Open a second terminal in:

```text
frontend/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create:

```text
frontend/.streamlit/secrets.toml
```

with:

```toml
API_BASE_URL = "http://localhost:8000"
```

Then run:

```bash
streamlit run app.py
```

The Streamlit frontend normally runs at:

```text
http://localhost:8501
```

For UI-only development, the frontend also includes a **Dev / demo login**
flow that allows the Home → Dashboard → Chat experience to be tested without
the live backend.

---

# 9. Connect Google OAuth Locally

For local development:

```env
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:8501
```

The Google login flow is:

```text
Streamlit
    ↓
/auth/google/login
    ↓
Google OAuth
    ↓
/auth/google/callback
    ↓
FRONTEND_URL?token=...
    ↓
Streamlit reads JWT
    ↓
/me
```

The Google OAuth redirect URI must exactly match the URI configured in
Google Cloud Console.

---

# 10. GitHub Setup

Before pushing the project:

Create a root `.gitignore` containing at least:

```gitignore
.env
*.env
.venv/
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.DS_Store
```

Then:

```bash
git init
git add .
git status
```

**Check that `.env` and `secrets.toml` are NOT being committed.**

Commit:

```bash
git commit -m "Initial StoxAI application"
```

Connect your GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git branch -M main
git push -u origin main
```

---

# 11. Deploy Backend to Render

Deploy the backend **before** deploying the Streamlit frontend.

Create a new Render **Web Service** connected to the GitHub repository.

If the repository is structured as a monorepo:

### Root Directory

```text
backend
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Do **not** upload `.env` to Render.

Instead, add the environment variables through Render's Environment Variables
section.

After deployment, Render will provide a URL such as:

```text
https://your-backend.onrender.com
```

Test:

```text
https://your-backend.onrender.com/health
```

and:

```text
https://your-backend.onrender.com/docs
```

---

# 12. Deploy the Streamlit Frontend

The frontend can be deployed separately using Render or Streamlit Community
Cloud.

After deployment, note the exact frontend URL, for example:

```text
https://your-app.streamlit.app
```

Do not include a trailing slash.

Configure the frontend's API URL to point to the deployed backend:

```text
https://your-backend.onrender.com
```

Do not leave production API calls pointing to:

```text
http://localhost:8000
```

---

# 13. Production Google OAuth Configuration

Once the backend is deployed, change the backend environment variables to:

```env
FRONTEND_URL=https://your-app.streamlit.app

GOOGLE_REDIRECT_URI=https://your-backend.onrender.com/auth/google/callback
```

Then add this **exact** callback URI to the Google OAuth client's
**Authorized redirect URIs**:

```text
https://your-backend.onrender.com/auth/google/callback
```

The redirect URI must match exactly, otherwise Google will return:

```text
redirect_uri_mismatch
```

After obtaining the Render backend URL, send it to the person managing the
Google Cloud OAuth configuration so they can add the callback URI.

---

# 14. CORS

The backend uses `FRONTEND_URL` as the default allowed origin.

For multiple frontend origins, configure:

```env
API_CORS_ORIGINS=https://your-app.streamlit.app,http://localhost:8501
```

Do **not** use:

```env
API_CORS_ORIGINS=*
```

The application uses credentialed requests, so wildcard CORS is not
appropriate here.

---

# 15. MongoDB Atlas

The production backend should use MongoDB Atlas rather than a local:

```text
mongodb://localhost:27017
```

Use an Atlas URI:

```text
mongodb+srv://...
```

and configure it as:

```env
MONGODB_URI=...
```

Make sure the Atlas database user and network-access settings permit the
deployed Render service to connect.

---

# 16. Models

The trained models should be included in the repository/image:

```text
models/lstm_model.keras
models/gru_model.keras
```

They are loaded lazily by `agents._get_models()` when a prediction is
requested.

---

# 17. Scheduled Tasks

News ingestion and other scheduled tasks are separate from normal user chat.

For production, scheduled tasks can be run through a separate Render
scheduled service/job using the appropriate script and the same required
environment variables.

For example, a news ingestion command can be:

```bash
python news_agent.py AAPL TSLA INFY
```

Keep scheduled/background work separate from the FastAPI web service unless
there is a specific reason to combine them.

---

# 18. Current Open Items

The following are areas that can be extended later.

### Agent run history

The frontend's **Recent Runs / Agent Activity** display is currently
client-side and does not persist across refreshes.

A persistent implementation would require:

1. Writing agent-run records to MongoDB.
2. Adding a backend route such as:
   ```text
   GET /agent-runs
   ```
3. Updating the frontend to retrieve those records.

### Tax profile

`/chat` accepts an optional `tax_profile`, but it is currently `None`.

For more personalized tax responses, a frontend form can collect fields such
as:

- Filing status
- Annual income
- Country
- Other fields defined by `TaxProfile` in `core.py`

### Report generation

`report_generation_node` is currently a stub and remains outside the current
scope.

### Checkpointer persistence

The default `memory` checkpointer does not survive Render restarts.
Persistent SQLite or PostgreSQL checkpointing can be added later.

---

# 19. API Overview

The FastAPI service includes authentication, user, chat, and administrative
functionality.

Important routes include:

```text
GET  /
GET  /health

GET  /auth/google/login
GET  /auth/google/callback

GET  /me
POST /chat

POST /admin/ingest-news
```

For the complete interactive API specification during development, open:

```text
http://localhost:8000/docs
```

or, after deployment:

```text
https://your-backend.onrender.com/docs
```

---

# 20. Development Checklist

Before GitHub:

- [ ] `.env` exists locally.
- [ ] `.env` is in `.gitignore`.
- [ ] `.streamlit/secrets.toml` is in `.gitignore`.
- [ ] No API keys or passwords are hardcoded.
- [ ] `requirements.txt` is present.
- [ ] Model files are present.
- [ ] Backend starts successfully.
- [ ] `/health` works.
- [ ] `/docs` works.
- [ ] Frontend starts successfully.

Before Render:

- [ ] Secrets have been rotated.
- [ ] GitHub repository does not contain `.env`.
- [ ] Render Root Directory is `backend`.
- [ ] Render Build Command is correct.
- [ ] Render Start Command is correct.
- [ ] All required Render environment variables are configured.
- [ ] MongoDB Atlas accepts the Render connection.
- [ ] Backend `/health` works on the Render URL.

Before production OAuth:

- [ ] Render backend URL is known.
- [ ] `GOOGLE_REDIRECT_URI` uses the Render backend URL.
- [ ] Google Cloud Console contains the exact callback URI.
- [ ] `FRONTEND_URL` is the deployed Streamlit URL.
- [ ] CORS contains the correct frontend origin.
- [ ] Frontend API URL points to the Render backend.

---

# 21. Quick Deployment Flow

```text
                    ┌─────────────────────┐
                    │      GitHub Repo    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌────────────────────┐
       │ Render Backend   │        │ Streamlit Frontend │
       │    FastAPI       │◄───────│                    │
       └────────┬─────────┘        └────────────────────┘
                │
       ┌────────┼─────────┐
       │        │         │
       ▼        ▼         ▼
   MongoDB   Google     LLM APIs
    Atlas     OAuth    / Finnhub /
                        Brevo
```

Recommended order:

```text
1. Run backend locally
        ↓
2. Run frontend locally
        ↓
3. Test Google OAuth locally
        ↓
4. Verify .gitignore
        ↓
5. Push to GitHub
        ↓
6. Deploy backend to Render
        ↓
7. Test /health and /docs
        ↓
8. Configure Google OAuth redirect URI
        ↓
9. Deploy Streamlit frontend
        ↓
10. Point frontend to Render backend
        ↓
11. Test complete Google → Dashboard → Chat flow
        ↓
12. Configure scheduled news/background jobs
```

---

## License

Add the project's license here if/when one is selected.
