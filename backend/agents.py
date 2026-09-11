"""
The "hackathon-side" agents, merged into one file since they're small and
the LangGraph nodes in graph.py call straight into them:

  SECTION 1: stock prediction   (formerly Agents/stock_agent.py)
  SECTION 2: email sending      (formerly Agents/email_agent.py) - Brevo ONLY
  SECTION 3: per-user watchlist (formerly Agents/db_agent.py) - now stored
             inside the same `users` document as the rest of the user's
             profile (see database.py), not a separate collection.

Each section also keeps its original optional standalone LangChain
ReAct-agent wrapper (run_stock_agent / run_email_agent) in case you want
to expose either as a free-form tool-calling agent later - but graph.py's
nodes call the plain functions directly, which is faster and doesn't need
a Groq round-trip just to run a lookup.

Neither the email agent nor the news-ingestion pipeline (see
news_agent.py) is called from any user-facing chat node. They stay here,
fully working, for you to trigger manually / on a schedule - see
news_agent.py's module docstring and main.py's /admin/ingest-news route.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from bson import ObjectId
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import config

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Stock prediction (formerly Agents/stock_agent.py)
# ===========================================================================

_lstm_model = None
_gru_model = None


def _get_models():
    global _lstm_model, _gru_model
    if _lstm_model is None:
        _lstm_model = load_model(f"{config.STOCK_MODEL_DIR}/lstm_model.keras")
    if _gru_model is None:
        _gru_model = load_model(f"{config.STOCK_MODEL_DIR}/gru_model.keras")
    return _lstm_model, _gru_model


def _fetch_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="2y")
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.reset_index()
    data["Close_Pct"] = data["Close"].pct_change()
    data.dropna(inplace=True)

    if len(data) < config.STOCK_PREDICTION_WINDOW + 1:
        raise ValueError(f"Not enough history for {ticker} to make a prediction")

    return data


def predict_stock(ticker: str) -> dict:
    """
    Returns {ticker, last_close, predicted_price, predicted_pct_change,
    direction}. Raises ValueError if the ticker has no/insufficient
    history - callers (graph.py nodes) should catch this and degrade
    gracefully rather than crash the whole chat turn.
    """
    ticker = ticker.strip().upper()
    data = _fetch_data(ticker)

    scaler_pct = MinMaxScaler()
    scaler_vol = MinMaxScaler()

    pct_scaled = scaler_pct.fit_transform(data[["Close_Pct"]])
    vol_scaled = scaler_vol.fit_transform(data[["Volume"]])

    scaled = np.hstack((pct_scaled, vol_scaled))
    x_input = scaled[-config.STOCK_PREDICTION_WINDOW:]
    x_input = np.expand_dims(x_input, axis=0)

    lstm_model, gru_model = _get_models()

    pred_pct_lstm_scaled = lstm_model.predict(x_input, verbose=0)
    pred_pct_gru_scaled = gru_model.predict(x_input, verbose=0)

    pred_pct_lstm = scaler_pct.inverse_transform(pred_pct_lstm_scaled).flatten()[0]
    pred_pct_gru = scaler_pct.inverse_transform(pred_pct_gru_scaled).flatten()[0]

    pred_pct = (pred_pct_lstm + pred_pct_gru) / 2

    last_close = data["Close"].iloc[-1]
    predicted_price = last_close * (1 + pred_pct)

    return {
        "ticker": ticker,
        "last_close": round(float(last_close), 2),
        "predicted_price": round(float(predicted_price), 2),
        "predicted_pct_change": round(float(pred_pct * 100), 2),
        "direction": "up" if pred_pct > 0 else "down",
    }


def get_live_price(ticker: str) -> float:
    """Cheap current-price lookup (no model inference) - used by
    resolve_price() below whenever a cached prediction is missing, and as
    a fallback anywhere a fast price is needed without running the
    LSTM/GRU models."""
    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)
    try:
        price = stock.fast_info["last_price"]
    except Exception:
        price = stock.history(period="1d")["Close"].iloc[-1]
    return round(float(price), 2)


def resolve_price(symbol: str, cached: dict) -> dict:
    """
    THE single place every agent should go through to get "the price to
    use" for a watchlist symbol. `cached` is one entry from a user's
    watchlist: {"live_price": ..., "predicted_price": ..., "updated_at": ...}.

    Rule: if predicted_price is config.NO_PREDICTION_SENTINEL (-1), it
    means "not predicted yet" - fetch a fresh live price instead of
    treating -1 as a real number. Otherwise trust the cached prediction
    (and cached live_price alongside it) so we're not re-running the
    LSTM/GRU models or hitting yfinance on every turn.

    Returns {"price": float|None, "predicted_price": float|None,
    "source": "cached_prediction"|"live_price_only", "live_price": float|None}.
    """
    cached = cached or {}
    predicted_price = cached.get("predicted_price", config.NO_PREDICTION_SENTINEL)

    if predicted_price is not None and predicted_price != config.NO_PREDICTION_SENTINEL:
        return {
            "price": predicted_price,
            "predicted_price": predicted_price,
            "live_price": cached.get("live_price"),
            "source": "cached_prediction",
        }

    try:
        live_price = get_live_price(symbol)
    except Exception as e:
        logger.warning("resolve_price: get_live_price(%s) failed: %s", symbol, e)
        live_price = cached.get("live_price")

    return {
        "price": live_price,
        "predicted_price": None,
        "live_price": live_price,
        "source": "live_price_only",
    }


def build_stock_agent():
    """Optional: wraps predict_stock as a Groq ReAct tool-agent, for a
    free-form 'ask about a stock in plain English' entry point instead of
    calling predict_stock() directly."""
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_stock_prediction(ticker: str) -> str:
        """Predict tomorrow's closing price and percentage change for a given stock ticker.
        ticker: stock ticker symbol, e.g. AAPL, GOOG, TSLA
        """
        result = predict_stock(ticker)
        return (
            f"{result['ticker']}: last close ${result['last_close']}, "
            f"predicted to go {result['direction']} by {abs(result['predicted_pct_change'])}% "
            f"to a predicted price of ${result['predicted_price']} tomorrow."
        )

    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0)
    return create_react_agent(llm, tools=[get_stock_prediction])


def run_stock_agent(instruction: str) -> str:
    agent = build_stock_agent()
    result = agent.invoke({"messages": [("user", instruction)]})
    return result["messages"][-1].content


# ===========================================================================
# SECTION 2: Email sending via Brevo ONLY (formerly Agents/email_agent.py)
#
# This is the ONLY email-sending code in the whole codebase. No SMTP, no
# SendGrid, no Mailgun. It is kept fully working but is NOT called from
# any user chat flow - wire it up yourself (e.g. a scheduled job, or your
# own admin route) if/when you want notifications again.
# ===========================================================================

import requests

BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def send_email_api(to: str, subject: str, body: str) -> str:
    """Low-level Brevo send via the Brevo transactional email HTTP API
    (NOT SMTP). Returns a human-readable status string rather than
    raising, so callers can log/display it without a try/except."""
    if not config.BREVO_API_KEY or not config.SENDER_EMAIL:
        return "Failed to send email: BREVO_API_KEY / SENDER_EMAIL not configured"

    headers = {
        "accept": "application/json",
        "api-key": config.BREVO_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "sender": {"name": config.SENDER_NAME, "email": config.SENDER_EMAIL},
        "to": [{"email": to}],
        "subject": subject,
        "textContent": body,
    }
    res = requests.post(BREVO_URL, headers=headers, json=payload, timeout=10)
    if res.status_code >= 300:
        return f"Failed to send email: {res.status_code} {res.text}"
    return f"Email sent to {to}"


def build_email_agent_for_user(user_email: str):
    """Optional: wraps send_email_api as a Groq ReAct tool-agent scoped to
    one recipient, for free-form 'send me a summary' instructions. Still
    not called anywhere automatically - available if you want it."""
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    @tool
    def send_email(subject: str, body: str) -> str:
        """Send an email to the logged-in user's own email address (via Brevo).
        subject: email subject line
        body: plain text email body
        """
        return send_email_api(user_email, subject, body)

    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0)
    return create_react_agent(llm, tools=[send_email])


def run_email_agent(user_email: str, instruction: str) -> str:
    agent = build_email_agent_for_user(user_email)
    result = agent.invoke({"messages": [("user", instruction)]})
    return result["messages"][-1].content


# ===========================================================================
# SECTION 3: Per-user stock watchlist (formerly Agents/db_agent.py)
#
# Now stored on the SAME `users` document as the rest of the profile
# (database.users_collection), in a "stocks" field, instead of a separate
# `user_stocks` collection. user_id is the string form of the Mongo
# ObjectId that auth_utils puts in the JWT.
# ===========================================================================

from database import users_collection


def get_user_stocks(user_id: str) -> dict:
    """{SYMBOL: {"live_price": float, "predicted_price": float, "updated_at": str}}"""
    try:
        doc = users_collection.find_one({"_id": ObjectId(user_id)}, {"stocks": 1})
    except Exception as e:
        logger.warning("get_user_stocks: bad user_id %s: %s", user_id, e)
        return {}
    return (doc or {}).get("stocks", {}) or {}


def store_user_stocks(user_id: str, stock_list: list) -> dict:
    """Adds any new tickers to the user's watchlist (on their own `users`
    document) with a live price and predicted_price =
    config.NO_PREDICTION_SENTINEL (-1), meaning 'not predicted yet' -
    stock_prediction_node fills that in on first use. Existing tickers are
    left untouched (their cached prediction stays)."""
    existing = get_user_stocks(user_id)
    updates = {}

    for ticker in stock_list:
        ticker = ticker.strip().upper()
        if ticker not in existing:
            try:
                price = get_live_price(ticker)
            except Exception as e:
                logger.warning("Couldn't fetch live price for %s: %s", ticker, e)
                price = config.NO_PREDICTION_SENTINEL
            updates[f"stocks.{ticker}"] = {
                "live_price": price,
                "predicted_price": config.NO_PREDICTION_SENTINEL,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    if updates:
        users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": updates})

    return get_user_stocks(user_id)


def store_predicted_price(user_id: str, ticker: str, predicted_price: float, live_price: float = None) -> None:
    """Called by stock_prediction_node after it runs the LSTM/GRU models
    for a watchlist symbol, so next time the planning flow doesn't need to
    re-run inference for the same symbol."""
    ticker = ticker.strip().upper()
    field = {
        f"stocks.{ticker}.predicted_price": predicted_price,
        f"stocks.{ticker}.updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if live_price is not None:
        field[f"stocks.{ticker}.live_price"] = live_price
    users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": field})
