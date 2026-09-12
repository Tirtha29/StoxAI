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
from typing import List, Optional, Literal, Dict, Any

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
    means "not predicted yet" - send current live price as predicted price
    so -1 is NEVER sent to the frontend.
    """
    cached = cached or {}
    predicted_price = cached.get("predicted_price", config.NO_PREDICTION_SENTINEL)

    try:
        live_price = get_live_price(symbol)
    except Exception as e:
        logger.warning("resolve_price: get_live_price(%s) failed: %s", symbol, e)
        live_price = cached.get("live_price")

    if predicted_price is not None and predicted_price != config.NO_PREDICTION_SENTINEL:
        eff_predicted = predicted_price
        source = "cached_prediction"
    else:
        # User requirement: if predicted price is -1, send the current live price!
        eff_predicted = live_price
        source = "live_price_fallback"

    return {
        "price": eff_predicted,
        "predicted_price": eff_predicted,
        "live_price": live_price,
        "source": source,
    }


# ===========================================================================
# SECTION 1B: Pydantic Query Parsing, Tavily Symbol Extraction & Graphing Agent
# ===========================================================================

import io
import base64
from core import ParsedUserQuery


def parse_user_query_with_pydantic_llm(user_query: str) -> ParsedUserQuery:
    """
    Uses LLM with Pydantic output parsing to extract query intent, stock symbols,
    company names, or tax saving intent from the user prompt.
    """
    if not user_query or not user_query.strip():
        return ParsedUserQuery(intent="general")

    if config.GROQ_API_KEY:
        models_to_try = [config.GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
        for model in models_to_try:
            if not model:
                continue
            try:
                from langchain_groq import ChatGroq
                from langchain_core.output_parsers import PydanticOutputParser
                from langchain_core.prompts import PromptTemplate

                parser = PydanticOutputParser(pydantic_object=ParsedUserQuery)
                llm = ChatGroq(api_key=config.GROQ_API_KEY, model=model, temperature=0.0)

                prompt = PromptTemplate(
                    template=(
                        "Classify the following user query for a financial assistant app and extract any ticker symbols or company names.\n"
                        "Query: {query}\n\n"
                        "Intent mapping guide:\n"
                        "- 'stock_info': asking about price, current performance, or prediction for a stock (e.g., 'how is AAPL doing', 'price of Reliance')\n"
                        "- 'save_interest': adding/tracking watchlist or favorite stocks (e.g., 'add TSLA to my watchlist', 'favorite stocks are AAPL, MSFT')\n"
                        "- 'portfolio_planning': asking for portfolio evaluation, scoring, or advice on stocks held\n"
                        "- 'stock_prediction': asking specifically for stock predictions or price targets\n"
                        "- 'tax_optimization': asking about saving taxes, 80C, LTCG, capital gains, tax loss harvesting\n"
                        "- 'report': asking for summary report, pdf, or dashboard\n"
                        "- 'general': general questions, news, general finance concepts\n\n"
                        "{format_instructions}\n"
                    ),
                    input_variables=["query"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )

                chain = prompt | llm | parser
                res = chain.invoke({"query": user_query})
                if res.symbol and not res.symbols:
                    res.symbols = [res.symbol]
                return res
            except Exception as e:
                logger.warning("parse_user_query_with_pydantic_llm (%s) failed: %s", model, e)

    # Heuristic fallback if LLM parser is unavailable
    query_lower = user_query.lower()
    symbols = []
    import re
    tokens = re.findall(r"\b[A-Za-z0-9\.]{1,10}\b", user_query)
    for tok in tokens:
        up = tok.upper()
        if tok.isupper() and len(tok) <= 6 and up not in {"I", "A", "AN", "THE", "MY", "IS", "ARE", "TO", "IN", "ON", "FOR", "AND", "OR", "ADD", "GET"}:
            symbols.append(up)

    if any(k in query_lower for k in ["tax", "80c", "deduction", "harvest", "ltcg", "capital gain"]):
        intent = "tax_optimization"
    elif any(k in query_lower for k in ["favorite stock", "favourite stock", "interested in", "watch ", "track ", "watchlist"]):
        intent = "save_interest"
    elif any(k in query_lower for k in ["portfolio score", "improve my portfolio", "more profit"]):
        intent = "portfolio_planning"
    elif any(k in query_lower for k in ["predict", "forecast", "price target"]):
        intent = "stock_prediction"
    elif any(k in query_lower for k in ["how's", "how is", "what about", "price of", "stock"]) or symbols:
        intent = "stock_info"
    else:
        intent = "general"

    return ParsedUserQuery(
        intent=intent,
        symbol=symbols[0] if symbols else None,
        symbols=symbols,
        is_stock_query=(intent in ["stock_info", "stock_prediction", "save_interest"]),
    )


COMMON_COMPANY_TICKERS = {
    "TESLA": "TSLA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
}


def resolve_stock_symbol_via_tavily(query_or_name: str) -> List[str]:
    """
    If user prompt asks about a stock (e.g. 'Tesla', 'Reliance', 'Tata Motors')
    without providing an exact ticker symbol, use company map + Tavily web search + LLM to lookup the exact symbol.
    """
    if not query_or_name or not query_or_name.strip():
        return []

    cleaned_name = query_or_name.strip().upper()
    for name, ticker in COMMON_COMPANY_TICKERS.items():
        if name in cleaned_name:
            return [ticker]

    search_query = f"{query_or_name} stock ticker symbol Yahoo Finance"
    results_text = ""

    # Attempt Tavily search
    if config.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            res = client.search(query=search_query, max_results=3)
            results_text = "\n".join([r.get("content", "") for r in res.get("results", [])])
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)

    if not results_text:
        # Fallback to direct HTTP search if Tavily key is missing/failed
        try:
            resp = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query_or_name, "quotesCount": 3},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", [])
                found = [q.get("symbol") for q in quotes if q.get("symbol")]
                if found:
                    return found[:2]
        except Exception as e:
            logger.warning("Yahoo Finance search fallback failed: %s", e)

    # Use LLM to extract ticker from Tavily results or query
    if config.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0.0)
            sys = (
                "You extract official stock ticker symbols from user text and search results. "
                "For example: 'Tesla' -> TSLA, 'Apple' -> AAPL, 'Reliance' -> RELIANCE.NS. "
                "Return ONLY a comma-separated list of valid upper-case stock symbols (e.g. TSLA, RELIANCE.NS, AAPL). "
                "If no stock symbol is found, return empty string."
            )
            prompt = f"User Query: {query_or_name}\nSearch Context:\n{results_text}"
            resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
            extracted = resp.content.strip()
            import re
            raw_syms = [s.strip().upper() for s in re.split(r"[\s,]+", extracted) if s.strip()]
            syms = [COMMON_COMPANY_TICKERS.get(s, s) for s in raw_syms]
            return [s for s in syms if len(s) <= 12]
        except Exception as e:
            logger.warning("LLM symbol extraction from Tavily failed: %s", e)

    return []


def generate_stock_graph(symbol: str, days: int = 30) -> dict:
    """
    Downloads historical performance for `symbol` over past `days` via yfinance,
    creates a base64 dark-themed PNG graph image and structured data points.
    """
    symbol = symbol.strip().upper()
    try:
        data = yf.download(symbol, period=f"{days}d")
        if data.empty:
            return {"symbol": symbol, "data_points": [], "graph_image_b64": None, "error": "No data returned"}

        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-index columns: keep first level
            data.columns = [col[0] for col in data.columns]

        data = data.reset_index()

        # Find date column and close column
        date_col = next((c for c in data.columns if str(c).lower() in ("date", "index")), data.columns[0])
        close_col = next((c for c in data.columns if "close" in str(c).lower()), "Close")

        data_points = []
        for _, row in data.iterrows():
            val = row[date_col]
            d_str = val.strftime("%Y-%m-%d") if hasattr(val, "strftime") else str(val)[:10]
            close_val = round(float(row[close_col]), 2)
            data_points.append({"date": d_str, "close": close_val})

        # Render dark-themed chart using matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3.5), facecolor="#0B1220")
        ax = plt.gca()
        ax.set_facecolor("#111A2C")

        dates = [dp["date"] for dp in data_points]
        closes = [dp["close"] for dp in data_points]

        plt.plot(dates, closes, color="#E8A33D", linewidth=2, label=f"{symbol} Close")
        plt.fill_between(dates, closes, min(closes)*0.98, color="#E8A33D", alpha=0.15)

        plt.title(f"{symbol} Performance (Past {days} Days)", color="#F3F5FA", fontsize=11, fontweight="bold", pad=10)
        plt.xlabel("Date", color="#9AA4BD", fontsize=8)
        plt.ylabel("Price", color="#9AA4BD", fontsize=8)

        # Style ticks and grid
        ax.tick_params(colors="#9AA4BD", labelsize=7)
        plt.xticks(rotation=30)
        ax.spines['bottom'].set_color('#223052')
        ax.spines['top'].set_color('#223052')
        ax.spines['right'].set_color('#223052')
        ax.spines['left'].set_color('#223052')
        plt.grid(True, linestyle="--", alpha=0.2, color="#223052")

        # Set x-ticks to reasonable frequency
        if len(dates) > 10:
            step = max(1, len(dates) // 6)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)])

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, facecolor=plt.gcf().get_facecolor(), edgecolor="none")
        plt.close()
        buf.seek(0)
        b64_str = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        return {
            "symbol": symbol,
            "data_points": data_points,
            "graph_image_b64": b64_str,
            "error": None,
        }

    except Exception as e:
        logger.warning("generate_stock_graph failed for %s: %s", symbol, e)
        return {"symbol": symbol, "data_points": [], "graph_image_b64": None, "error": str(e)}


def tavily_general_search_answer(user_query: str, chat_history: list = None) -> Optional[str]:
    """
    Fallback for general knowledge queries when no specialized agent handles it.
    Uses Tavily search to fetch real-time web info and synthesizes response with LLM.
    """
    search_context = ""
    if config.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            res = client.search(query=user_query, max_results=4)
            search_context = "\n\n".join([f"Source: {r.get('title')}\n{r.get('content')}" for r in res.get("results", [])])
        except Exception as e:
            logger.warning("Tavily general search failed: %s", e)

    # Use LLM (Groq or Anthropic) to produce answer
    prompt = f"User Question: {user_query}\n\n"
    if search_context:
        prompt += f"Real-time Web Search Results:\n{search_context}\n\nSummarize the answer clearly based on search results."
    else:
        prompt += "Answer concisely based on general knowledge."

    sys_prompt = "You are an intelligent financial and general assistant. Provide accurate, helpful, plain-language answers."

    # Try Groq first for speed
    if config.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0.3)
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            logger.warning("Groq general synthesis failed: %s", e)

    return _general_llm_answer(user_query, chat_history)


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
