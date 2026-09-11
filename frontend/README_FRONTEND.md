# StoxAI frontend — setup & open items

## Run it
```
pip install -r requirements.txt
mkdir .streamlit
echo 'API_BASE_URL = "http://localhost:8000"' > .streamlit/secrets.toml
streamlit run app.py
```
Right now, without touching your backend, you can already click "Get started"
→ "Dev / demo login" and walk through the whole Home → Dashboard → Chat flow
with fake data. That's how to build/test the UI before the backend is live.

## To connect it for real
1. Run your backend: `uvicorn main:app --reload --port 8000`
2. Fill in `.env` for the backend at minimum: `GOOGLE_CLIENT_ID`,
   `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` (must point at
   `http://localhost:8000/auth/google/callback` while testing locally),
   `FRONTEND_URL` (must be the exact URL Streamlit is running on, e.g.
   `http://localhost:8501`, no trailing slash — this is also what
   `API_CORS_ORIGINS` defaults to), `JWT_SECRET_KEY`, `MONGODB_URI`.
3. Click "Continue with Google" in the app — it hits `/auth/google/login`,
   Google redirects back to `FRONTEND_URL?token=...`, and `app.py` reads
   that token and calls `/me`.

## Decisions I need from you
- **Auth**: Google-only (as built) or do you also want username/password?
  If the latter, that's a new backend route — tell me and I'll adjust the
  login modal to match once it exists.
- **Agent run history**: currently faked client-side from `intent` on each
  `/chat` reply, and it's lost on refresh. If you want the "Recent runs" /
  "Agent activity" panel to persist and show what ran even when the user
  isn't in an active chat (e.g. the nightly retrain, the email digest job),
  the scheduler needs to write rows somewhere (Mongo collection is simplest)
  and main.py needs a `GET /agent-runs` route to read them back.
- **Tax profile**: `/chat` accepts an optional `tax_profile` — I left it as
  `None` for now. If you want tax questions answered accurately, add a form
  in the sidebar (filing_status, annual_income, country, etc. — schema is in
  `core.py`'s `TaxProfile`) and I'll wire it in.
- **Deployment**: once this moves off localhost, I need the deployed API URL
  to put in `secrets.toml`, and it must exactly match `FRONTEND_URL` on the
  backend for CORS + the OAuth redirect to work.

## Files still missing from what you sent
- `agents.py` — listed in your folder screenshot, not present in the zip.
- The actual scheduler script (for the nightly retrain / news ingestion cron).
