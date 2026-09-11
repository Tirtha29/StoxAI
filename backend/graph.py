"""
Orchestration layer for the whole merged app: LangGraph node functions,
the compiled StateGraph (with a checkpointer so multi-turn conversations
persist across calls), and the remote-agent HTTP adapter.

Node naming convention: `<name>_node(state: AgentState) -> dict`. Each
node returns ONLY the key(s) it owns - LangGraph merges the rest.

--------------------------------------------------------------------------
CONVERSATION FLOW (what router_node decides, in order):

  0. state["pending_action"] == "await_stock_list"
       -> straight to save_interest_node, no keyword matching. This is
          set by planning_node when the user's watchlist was empty; the
          very next message is treated as "here are my tickers", however
          it's phrased.

  1. tax keywords ("tax", "80c", "deduction", "harvest", "ltcg", ...)
       -> tax_saving_node (rules engine + tax-doc RAG + LLM explanation)

  2. "how's X doing" / "what about <TICKER>" / a bare ticker-looking query
       -> stock_info_node (live price + cached/fresh prediction for ONE symbol)

  3. "my favorite stocks are ..." / "I'm interested in ..." / "track/watch ..."
       -> save_interest_node (parses tickers, upserts the user's watchlist)

  4. "how can I improve my portfolio" / "which stocks should I profit from" /
     "portfolio score" -> planning_node:
       - no watchlist yet -> ask the user to list stocks (pending_action set,
         see #0) instead of guessing
       - watchlist exists -> for each symbol: resolve a price (cached
         prediction if it isn't the -1 "not predicted yet" sentinel,
         otherwise a live price - see agents.resolve_price), then an LLM
         pass turns that into a plain-language note. No email is sent.

  5. "predict" / "forecast" / "price target" -> stock_prediction_node
     (batch LSTM+GRU prediction over state["portfolio"], NOT the watchlist)
  6. "report" / "summary" / "pdf" / "dashboard" -> report_generation_node (stub)
  7. anything else -> chatbot_response_node directly ("general") - answered
     from the model's own general knowledge, not from any specialized
     agent. Nothing here ever triggers a news fetch: news_agent.py is only
     ever run by us (CLI / cron / the admin-only route in main.py), never
     from a user chat turn.
--------------------------------------------------------------------------

Run standalone for a quick smoke test:  python graph.py
"""

import json
import logging
import os
import re
from typing import List, Optional

import requests
from langgraph.graph import StateGraph, END

import config
import agents
from core import (
    AgentState,
    PortfolioHolding,
    TaxProfile,
    TaxSavingResponse,
    StockPredictionResult,
    generate_all_suggestions,
    generate_grounded_explanation,
)
from rag import TaxRAGRetriever

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Remote-agent HTTP adapter (unchanged from the original)
# ===========================================================================

class RemoteAgentError(Exception):
    pass


def call_remote_agent(url: str, payload: dict, timeout: Optional[int] = None) -> dict:
    timeout = timeout or config.AGENT_CALL_TIMEOUT_SECONDS
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise RemoteAgentError(f"call to {url} failed: {e}")
    except ValueError as e:
        raise RemoteAgentError(f"call to {url} returned non-JSON response: {e}")


# ===========================================================================
# SECTION 2: Small shared helpers
# ===========================================================================

_retriever = None


def _get_retriever() -> TaxRAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TaxRAGRetriever()
    return _retriever


# Words that look like tickers (all-caps, 1-6 chars) but are common enough
# in ordinary chat that we should never treat them as a symbol.
_TICKER_STOPWORDS = {
    "I", "A", "AN", "THE", "MY", "IS", "ARE", "TO", "IN", "ON", "OF", "AND",
    "OR", "FOR", "ADD", "GET", "SET", "HOW", "WHAT", "ABOUT", "STOCK",
    "STOCKS", "PLEASE", "ALSO", "WANT", "LIKE", "TRACK", "WATCH", "FAVORITE",
    "FAVOURITE", "INTERESTED", "PORTFOLIO", "SCORE", "PROFIT", "YES", "NO",
}


def _extract_symbols(text: str) -> List[str]:
    """
    Heuristic ticker extraction: comma/space separated all-caps tokens,
    1-6 letters, minus common stopwords. Good enough for a hackathon demo;
    swap for a Groq extraction call (build a one-line `_llm_extract_symbols`
    using agents' ChatGroq pattern) if you need to handle lowercase input
    or company names like "Tesla" reliably.
    """
    candidates = re.findall(r"\b[A-Za-z]{1,6}\b", text)
    seen, symbols = set(), []
    for tok in candidates:
        up = tok.upper()
        if up in _TICKER_STOPWORDS or up in seen:
            continue
        if tok.isupper() or len(tok) <= 5:  # allow "tsla" typed lowercase too
            seen.add(up)
            symbols.append(up)
    return symbols


def _llm_text(prompt: str, system: str = "", max_tokens: int = 300) -> Optional[str]:
    """One-shot Groq completion, used for the planning node's per-symbol
    synthesis. Returns None (never raises) if GROQ_API_KEY isn't set or the
    call fails - callers fall back to a template."""
    if not config.GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL, temperature=0.3)
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        logger.warning("Groq synthesis failed, falling back to template: %s", e)
        return None


GENERAL_SYSTEM_PROMPT = """You are the general-purpose fallback of a \
financial assistant app. The user's message didn't match any of the \
app's specialized agents (tax saving, stock prediction, watchlist, \
portfolio planning). Answer helpfully from your own general knowledge. \
If the question needs live/current data you don't have (today's exact \
price, breaking news, etc.), say so plainly and suggest the specific \
in-app command that would get them a live answer (e.g. "ask how's \
AAPL doing" for a live price/prediction). Never claim to have checked \
the news or the internet - you have not. Keep the answer concise and \
plain-language. This is not financial advice."""


def _general_llm_answer(user_query: str, chat_history: Optional[list] = None) -> Optional[str]:
    """
    General-knowledge fallback for anything that doesn't match a
    specialized agent's routing keywords - uses the same Anthropic model
    core.py's explanation layer uses. Returns None (never raises) if no
    API key is set or the call fails, so callers can fall back to a
    static message.
    """
    if not config.LLM_API_KEY:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.LLM_API_KEY)
        messages = []
        for turn in (chat_history or [])[-6:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_query})

        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=500,
            system=GENERAL_SYSTEM_PROMPT,
            messages=messages,
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as e:
        logger.warning("General LLM fallback failed: %s", e)
        return None


# ===========================================================================
# SECTION 3: Router
# ===========================================================================

def router_node(state: AgentState) -> dict:
    if state.get("pending_action") == "await_stock_list":
        return {"intent": "save_interest"}

    query = state.get("user_query", "").lower()

    if any(k in query for k in ["tax", "80c", "deduction", "harvest", "ltcg", "capital gain"]):
        intent = "tax_optimization"
    elif any(k in query for k in ["favorite stock", "favourite stock", "interested in", "watch ", "track ", "watchlist"]):
        intent = "save_interest"
    elif any(k in query for k in ["portfolio score", "improve my portfolio", "more profit", "which stocks should i", "how to profit"]):
        intent = "portfolio_planning"
    elif any(k in query for k in ["predict", "forecast", "price target"]):
        intent = "stock_prediction"
    elif any(k in query for k in ["how's", "how is", "what about", "doing today", "going up", "going down", "price of"]) or _extract_symbols(state.get("user_query", "")):
        intent = "stock_info"
    elif any(k in query for k in ["report", "summary", "pdf", "dashboard"]):
        intent = "report"
    else:
        intent = "general"

    return {"intent": intent}


def route_after_router(state: AgentState) -> str:
    return state.get("intent", "general")


# ===========================================================================
# SECTION 4: Tax-saving node (unchanged from the tax_saving_agent repo)
# ===========================================================================

def tax_saving_node(state: AgentState) -> dict:
    portfolio_raw = state.get("portfolio", [])
    tax_profile_raw = state.get("tax_profile", {})

    portfolio: List[PortfolioHolding] = [PortfolioHolding(**h) for h in portfolio_raw]
    profile = TaxProfile(**tax_profile_raw) if tax_profile_raw else TaxProfile()

    suggestions = generate_all_suggestions(portfolio, profile)

    retriever = _get_retriever()
    for s in suggestions:
        query = f"{s.title}. {s.detail}"
        hits = retriever.query(query, k=3, country=profile.country)
        s.source_snippets = [h["text"] for h in hits]

    total_savings = round(sum(s.estimated_savings for s in suggestions), 2)
    explanation = generate_grounded_explanation(
        [s.model_dump() for s in suggestions], total_savings
    )

    response = TaxSavingResponse(
        suggestions=suggestions,
        total_estimated_savings=total_savings,
        explanation=explanation,
    )
    return {"tax_result": json.loads(response.model_dump_json())}


# ===========================================================================
# SECTION 5: Stock prediction node - batch LSTM+GRU over state["portfolio"]
# ===========================================================================

def stock_prediction_node(state: AgentState) -> dict:
    portfolio_raw = state.get("portfolio", [])
    symbols = [h["symbol"] for h in portfolio_raw] if portfolio_raw else _extract_symbols(state.get("user_query", ""))

    results = []
    for symbol in symbols:
        try:
            pred = agents.predict_stock(symbol)
        except Exception as e:
            logger.warning("stock_prediction_node: predict_stock(%s) failed: %s", symbol, e)
            results.append(StockPredictionResult(
                symbol=symbol,
                predicted_direction="flat",
                confidence=0.0,
                explanation=f"Prediction unavailable for {symbol}: {e}",
            ))
            continue

        results.append(StockPredictionResult(
            symbol=pred["ticker"],
            predicted_direction=pred["direction"],
            predicted_range=[pred["last_close"], pred["predicted_price"]],
            confidence=0.0,  # the LSTM/GRU models don't emit a calibrated confidence score
            explanation=(
                f"Last close {pred['last_close']}, model predicts {pred['direction']} "
                f"{abs(pred['predicted_pct_change'])}% to {pred['predicted_price']}."
            ),
        ))

        # if this symbol is on the user's watchlist, refresh the cached
        # prediction so planning_node/stock_info_node stop hitting the -1
        # sentinel for it next time
        user_id = state.get("user_id")
        if user_id:
            try:
                agents.store_predicted_price(user_id, pred["ticker"], pred["predicted_price"], pred["last_close"])
            except Exception as e:
                logger.warning("stock_prediction_node: couldn't cache prediction for %s: %s", symbol, e)

    return {"stock_prediction_result": [json.loads(r.model_dump_json()) for r in results]}


# ===========================================================================
# SECTION 6: Stock info node - "how's TICKER doing" for one symbol
#
# No news fetch here anymore - live price + prediction only. If the
# symbol is already on the user's watchlist, the -1 sentinel rule is
# honored via agents.resolve_price(); otherwise this runs a fresh
# prediction, same as before.
# ===========================================================================

def stock_info_node(state: AgentState) -> dict:
    symbols = _extract_symbols(state.get("user_query", ""))
    if not symbols:
        return {"stock_info_result": {"error": "no ticker found in the message"}}

    symbol = symbols[0]
    info: dict = {"symbol": symbol}

    user_id = state.get("user_id")
    cached = agents.get_user_stocks(user_id).get(symbol) if user_id else None

    if cached:
        resolved = agents.resolve_price(symbol, cached)
        info["live_price"] = resolved["live_price"]
        info["predicted_price"] = resolved["predicted_price"]
        info["source"] = resolved["source"]
    else:
        try:
            info["live_price"] = agents.get_live_price(symbol)
        except Exception as e:
            info["live_price"] = None
            info["price_error"] = str(e)

        try:
            pred = agents.predict_stock(symbol)
            info["predicted_price"] = pred["predicted_price"]
            info["direction"] = pred["direction"]
        except Exception as e:
            info["predicted_price"] = None
            info["prediction_error"] = str(e)

    return {"stock_info_result": info}


# ===========================================================================
# SECTION 7: Save-interest node - persists the user's watchlist
# (now stored on their own `users` document - see agents.py / database.py)
# ===========================================================================

def save_interest_node(state: AgentState) -> dict:
    user_id = state.get("user_id")
    if not user_id:
        return {"save_interest_result": {"error": "not authenticated - no user_id in state"}, "pending_action": None}

    symbols = _extract_symbols(state.get("user_query", ""))
    if not symbols:
        # user was asked for a list and didn't give one we could parse -
        # keep pending_action set so we ask again next turn
        return {"save_interest_result": {"error": "no tickers found"}}

    watchlist = agents.store_user_stocks(user_id, symbols)
    return {"save_interest_result": {"saved": symbols, "watchlist": watchlist}, "pending_action": None}


# ===========================================================================
# SECTION 8: Portfolio planning node
#
# No news fetch, no email. Per-symbol price comes from
# agents.resolve_price() (honors the -1 "not predicted yet" sentinel by
# falling back to a live price), then an LLM pass turns (symbol + price)
# into a short plain-language note.
# ===========================================================================

PLANNING_SYSTEM_PROMPT = """You are the planning layer of a stock-watchlist \
assistant. For ONE symbol you are given its current live price and, if \
available, a model-predicted price. Write 1-2 plain-language sentences on \
what's happening and what to watch for - never a firm buy/sell \
instruction, this is not investment advice. Keep it under 50 words."""


def _plan_for_symbol(symbol: str, cached: dict) -> dict:
    resolved = agents.resolve_price(symbol, cached)
    live_price = resolved["live_price"]
    predicted_price = resolved["predicted_price"]
    source = resolved["source"]

    prompt = (
        f"Symbol: {symbol}\nLive price: {live_price}\n"
        f"Predicted price: {predicted_price if predicted_price is not None else 'not available'}"
    )
    explanation = _llm_text(prompt, system=PLANNING_SYSTEM_PROMPT) or (
        f"{symbol}: live price {live_price}, predicted price "
        f"{predicted_price if predicted_price is not None else 'n/a'}."
    )

    return {
        "symbol": symbol,
        "live_price": live_price,
        "predicted_price": predicted_price,
        "source": source,
        "explanation": explanation,
    }


def planning_node(state: AgentState) -> dict:
    user_id = state.get("user_id")
    if not user_id:
        return {"planning_result": {"error": "not authenticated - no user_id in state"}}

    watchlist = agents.get_user_stocks(user_id)
    if not watchlist:
        return {
            "pending_action": "await_stock_list",
            "planning_result": {"status": "need_stock_list"},
        }

    per_symbol = [_plan_for_symbol(sym, cached) for sym, cached in watchlist.items()]
    result = {"status": "ok", "symbols": per_symbol}

    return {"planning_result": result}


# ===========================================================================
# SECTION 9: Report generation (still a stub - unchanged scope of this merge)
# ===========================================================================

def report_generation_node(state: AgentState) -> dict:
    return {"report_result": {"sections": [], "report_url": None}}


# ===========================================================================
# SECTION 10: Final response composition
# ===========================================================================

def chatbot_response_node(state: AgentState) -> dict:
    parts = []

    if state.get("tax_result"):
        parts.append(state["tax_result"]["explanation"])

    if state.get("stock_info_result"):
        info = state["stock_info_result"]
        if info.get("error"):
            parts.append("I couldn't find a ticker in that message - try something like \"how's AAPL doing?\".")
        else:
            line = f"{info['symbol']}: live price {info.get('live_price', 'n/a')}"
            if info.get("predicted_price"):
                line += f", model predicts {info.get('direction', '')} to {info['predicted_price']}".rstrip()
            parts.append(line)

    if state.get("save_interest_result"):
        si = state["save_interest_result"]
        if si.get("error") == "no tickers found":
            parts.append("I couldn't pick out any tickers from that - could you list them like \"AAPL, TSLA, INFY\"?")
        elif si.get("error"):
            parts.append("I couldn't save that - please sign in and try again.")
        else:
            parts.append(f"Added to your watchlist: {', '.join(si['saved'])}. Ask me about your portfolio score whenever you're ready.")

    if state.get("planning_result"):
        pr = state["planning_result"]
        if pr.get("status") == "need_stock_list":
            parts.append("You don't have any stocks saved yet - tell me which ones you're interested in (e.g. \"I'm interested in AAPL, TSLA\") and then ask me again.")
        elif pr.get("error"):
            parts.append("I couldn't build a portfolio plan - please sign in and try again.")
        else:
            lines = ["Here's what I'm seeing on your watchlist:"]
            for s in pr["symbols"]:
                lines.append(f"- {s['symbol']}: {s['explanation']}")
            parts.append("\n".join(lines))

    if state.get("stock_prediction_result"):
        lines = ["Predictions:"]
        for p in state["stock_prediction_result"]:
            lines.append(f"- {p['symbol']}: {p['explanation']}")
        parts.append("\n".join(lines))

    if state.get("report_result", {}).get("report_url"):
        parts.append(f"Full report: {state['report_result']['report_url']}")

    if not parts:
        # "general" intent (or every specialized node came back empty) -
        # answer from the model's own general knowledge instead of a flat
        # "I don't understand" message.
        answer = _general_llm_answer(state.get("user_query", ""), state.get("chat_history"))
        parts.append(answer or (
            "I don't have a live data source for that yet, so I can't verify current "
            "facts here - but generally speaking: "
            + (state.get("user_query") or "").strip()
        ))

    final = "\n\n".join(parts)
    return {"final_response": final}


# ===========================================================================
# SECTION 11: Checkpointer + graph builder
# ===========================================================================

def _make_checkpointer():
    """MemorySaver by default. See config.CHECKPOINTER_BACKEND for the
    tradeoff (in-memory does not survive a Render restart)."""
    backend = config.CHECKPOINTER_BACKEND

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver.from_conn_string(config.CHECKPOINTER_SQLITE_PATH)
        except ImportError:
            logger.warning("CHECKPOINTER_BACKEND=sqlite but langgraph-checkpoint-sqlite isn't "
                            "installed - falling back to in-memory. `pip install langgraph-checkpoint-sqlite`.")

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("tax_saving", tax_saving_node)
    graph.add_node("stock_prediction", stock_prediction_node)
    graph.add_node("stock_info", stock_info_node)
    graph.add_node("save_interest", save_interest_node)
    graph.add_node("portfolio_planning", planning_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("chatbot_response", chatbot_response_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "tax_optimization": "tax_saving",
            "stock_prediction": "stock_prediction",
            "stock_info": "stock_info",
            "save_interest": "save_interest",
            "portfolio_planning": "portfolio_planning",
            "report": "report_generation",
            "general": "chatbot_response",
        },
    )

    for node in (
        "tax_saving", "stock_prediction", "stock_info", "save_interest",
        "portfolio_planning", "report_generation",
    ):
        graph.add_edge(node, "chatbot_response")

    graph.add_edge("chatbot_response", END)

    return graph.compile(checkpointer=_make_checkpointer())


if __name__ == "__main__":
    app = build_graph()

    with open(os.path.join(os.path.dirname(__file__), "sample_data", "sample_portfolio.json")) as f:
        sample = json.load(f)

    initial_state: AgentState = {
        "user_query": "how much tax can I save right now?",
        "chat_history": [],
        "portfolio": sample["portfolio"],
        "tax_profile": sample["tax_profile"],
    }

    # thread_id groups turns into one persisted conversation - normally
    # this is the user_id (see main.py), a fixed id is fine for this smoke test
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "smoke-test"}})
    print(json.dumps(result, indent=2, default=str))
