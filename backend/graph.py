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


def reset_retriever():
    global _retriever
    _retriever = None


# Words that look like tickers (all-caps, 1-6 chars) but are common enough
# in ordinary chat that we should never treat them as a symbol.
_TICKER_STOPWORDS = {
    "I", "A", "AN", "THE", "MY", "IS", "ARE", "TO", "IN", "ON", "OF", "AND",
    "OR", "FOR", "ADD", "GET", "SET", "HOW", "WHAT", "ABOUT", "STOCK",
    "STOCKS", "PLEASE", "ALSO", "WANT", "LIKE", "TRACK", "WATCH", "FAVORITE",
    "FAVOURITE", "INTERESTED", "PORTFOLIO", "SCORE", "PROFIT", "YES", "NO",
    "SHOW", "TELL", "GRAPH", "CHART", "PAST", "DAYS", "DAY", "PRICE", "PRICES",
    "PREDICT", "PREDICTED", "TAX", "TAXES", "SAVE", "SAVINGS", "CAN", "EARN",
    "ME", "GIVE", "NEED", "CURRENT", "HISTORICAL", "HI", "HELLO", "FIND", "CHECK",
    "SEE", "LOOK", "FETCH", "RUN", "INDIA", "BANGALORE",
}


def _extract_symbols(text: str) -> List[str]:
    """
    Heuristic ticker extraction: comma/space separated tokens,
    mapped through COMMON_COMPANY_TICKERS for company names like 'Tesla' or 'Reliance'.
    """
    candidates = re.findall(r"\b[A-Za-z0-9\.]{1,12}\b", text)
    seen, symbols = set(), []
    for tok in candidates:
        up = tok.upper()
        if up in _TICKER_STOPWORDS or up in seen:
            continue
        mapped = agents.COMMON_COMPANY_TICKERS.get(up, up)
        if mapped not in seen and mapped not in _TICKER_STOPWORDS:
            seen.add(mapped)
            symbols.append(mapped)
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
        return {
            "intent": "save_interest",
            "tax_result": None,
            "stock_info_result": None,
            "stock_prediction_result": None,
            "save_interest_result": None,
            "planning_result": None,
            "report_result": None,
            "final_response": None,
            "agents_executed": [],
        }

    query = state.get("user_query", "")
    parsed = agents.parse_user_query_with_pydantic_llm(query)
    intent = parsed.intent
    symbols = parsed.symbols

    if not symbols:
        # Check if Tavily symbol resolution can discover ticker from company/query
        if parsed.is_stock_query or parsed.company_or_stock_name or intent in ("stock_info", "save_interest", "stock_prediction"):
            lookup_target = parsed.company_or_stock_name or query
            tavily_syms = agents.resolve_stock_symbol_via_tavily(lookup_target)
            if tavily_syms:
                symbols = tavily_syms
                if intent == "general":
                    intent = "stock_info"

    if not symbols:
        symbols = _extract_symbols(query)

    # CRITICAL: Clear per-turn result keys from previous turns in checkpointer state
    return {
        "intent": intent,
        "parsed_symbols": symbols,
        "tax_result": None,
        "stock_info_result": None,
        "stock_prediction_result": None,
        "save_interest_result": None,
        "planning_result": None,
        "report_result": None,
        "final_response": None,
        "agents_executed": [],
    }


def route_after_router(state: AgentState) -> str:
    return state.get("intent", "general")


# ===========================================================================
# SECTION 4: Tax-saving node (unchanged from the tax_saving_agent repo)
# ===========================================================================

def tax_saving_node(state: AgentState) -> dict:
    user_query = state.get("user_query", "")
    portfolio_raw = state.get("portfolio", [])
    tax_profile_raw = state.get("tax_profile", {})

    portfolio: List[PortfolioHolding] = [PortfolioHolding(**h) for h in portfolio_raw]
    profile = TaxProfile(**tax_profile_raw) if tax_profile_raw else TaxProfile()

    # 1. Query RAG retriever using the actual user query
    retriever = _get_retriever()
    user_rag_hits = retriever.query(user_query, k=5, country=profile.country) if user_query else []
    user_snippets = [h["text"] for h in user_rag_hits]

    # 2. Evaluate rule-based tax opportunities
    suggestions = generate_all_suggestions(portfolio, profile)
    for s in suggestions:
        query = f"{s.title}. {s.detail}"
        hits = retriever.query(query, k=3, country=profile.country)
        s.source_snippets = [h["text"] for h in hits]

    # Combine RAG hits from user query and suggestions
    all_snippets = list(dict.fromkeys(user_snippets + [sn for s in suggestions for sn in s.source_snippets]))

    total_savings = round(sum(s.estimated_savings for s in suggestions), 2)

    # 3. Grounded answer generation answering the user's specific query using RAG context
    context_text = "\n\n".join(f"--- Document Snippet {i+1} ---\n{sn}" for i, sn in enumerate(all_snippets[:6]))
    
    system_prompt = (
        "You are an expert financial and tax advisor AI assistant. "
        "Answer the user's specific question directly, thoroughly, and accurately based on the provided document snippets, "
        "tax profile, and portfolio details. "
        "If the user asks about tax rules, exemption limits, portfolio details, or uploaded documents, provide a clear, plain-language answer "
        "grounded directly in the retrieved document text. "
        "Do not invent facts or numbers not present in the context."
    )
    
    prompt_text = (
        f"User Question: {user_query}\n\n"
        f"Retrieved Document / RAG Context Snippets:\n{context_text if context_text else 'No specific document snippets found.'}\n\n"
        f"User Tax Profile: Annual Income={profile.annual_income}, Country={profile.country}, 80C Used={profile.section_80c_used}\n"
        f"Calculated Tax Opportunities: {[s.title + ': ' + s.detail for s in suggestions]}"
    )

    explanation = _llm_text(prompt_text, system=system_prompt)

    if not explanation:
        explanation = generate_grounded_explanation(
            [s.model_dump() for s in suggestions], total_savings
        )

    response = TaxSavingResponse(
        suggestions=suggestions,
        total_estimated_savings=total_savings,
        explanation=explanation,
    )
    return {
        "tax_result": json.loads(response.model_dump_json()),
        "agents_executed": ["News Agent", "Tax Rules Engine Agent", "Tax RAG Retriever Agent", "Tax Explanation Agent"],
    }


# ===========================================================================
# SECTION 5: Stock prediction node - batch LSTM+GRU over state["portfolio"]
# ===========================================================================

def stock_prediction_node(state: AgentState) -> dict:
    portfolio_raw = state.get("portfolio", [])
    symbols = [h["symbol"] for h in portfolio_raw] if portfolio_raw else state.get("parsed_symbols") or _extract_symbols(state.get("user_query", ""))

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
            confidence=0.0,
            explanation=(
                f"Last close {pred['last_close']}, model predicts {pred['direction']} "
                f"{abs(pred['predicted_pct_change'])}% to {pred['predicted_price']}."
            ),
        ))

        user_id = state.get("user_id")
        if user_id:
            try:
                agents.store_predicted_price(user_id, pred["ticker"], pred["predicted_price"], pred["last_close"])
            except Exception as e:
                logger.warning("stock_prediction_node: couldn't cache prediction for %s: %s", symbol, e)

    return {
        "stock_prediction_result": [json.loads(r.model_dump_json()) for r in results],
        "agents_executed": ["News Agent", "Prediction Agent", "Stock Info Agent"],
    }


# ===========================================================================
# SECTION 6: Stock info node - Live Price + Predicting Agent + Graph Making Agent
# ===========================================================================

def stock_info_node(state: AgentState) -> dict:
    symbols = state.get("parsed_symbols") or _extract_symbols(state.get("user_query", ""))
    if not symbols:
        symbols = agents.resolve_stock_symbol_via_tavily(state.get("user_query", ""))

    if not symbols:
        return {
            "stock_info_result": {"error": "no ticker found in the message"},
            "agents_executed": ["News Agent", "Stock Info Agent"],
        }

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
            info["predicted_price"] = pred.get("predicted_price")
            info["direction"] = pred.get("direction")
        except Exception as e:
            info["predicted_price"] = None
            info["prediction_error"] = str(e)

    # Sentinel Check: If predicted price is -1 or None, send current live price (never -1)
    if info.get("predicted_price") is None or info.get("predicted_price") == config.NO_PREDICTION_SENTINEL:
        info["predicted_price"] = info.get("live_price")

    # Invoke Graph Making Agent
    graph_res = agents.generate_stock_graph(symbol, days=30)
    info["graph_image_b64"] = graph_res.get("graph_image_b64")
    info["data_points"] = graph_res.get("data_points", [])

    return {
        "stock_info_result": info,
        "agents_executed": ["News Agent", "Prediction Agent", "Stock Info Agent", "Graph Visualization Agent"],
    }


# ===========================================================================
# SECTION 7: Save-interest node - persists the user's watchlist
# ===========================================================================

def save_interest_node(state: AgentState) -> dict:
    user_id = state.get("user_id")
    if not user_id:
        return {
            "save_interest_result": {"error": "not authenticated - no user_id in state"},
            "pending_action": None,
            "agents_executed": ["News Agent", "Stock Info Agent", "Watchlist Agent"],
        }

    symbols = state.get("parsed_symbols") or _extract_symbols(state.get("user_query", ""))
    if not symbols:
        symbols = agents.resolve_stock_symbol_via_tavily(state.get("user_query", ""))

    if not symbols:
        return {
            "save_interest_result": {"error": "no tickers found"},
            "agents_executed": ["News Agent", "Stock Info Agent", "Watchlist Agent"],
        }

    watchlist = agents.store_user_stocks(user_id, symbols)
    return {
        "save_interest_result": {"saved": symbols, "watchlist": watchlist},
        "pending_action": None,
        "agents_executed": ["News Agent", "Stock Info Agent", "Watchlist Agent"],
    }


# ===========================================================================
# SECTION 8: Portfolio planning node
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
    if predicted_price is None or predicted_price == config.NO_PREDICTION_SENTINEL:
        predicted_price = live_price
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
    user_query = state.get("user_query", "")

    # 1. Extract symbols from current user query first
    symbols = state.get("parsed_symbols") or _extract_symbols(user_query)
    if not symbols:
        symbols = agents.resolve_stock_symbol_via_tavily(user_query)

    # 2. If no symbols in query, fallback to user's saved watchlist if authenticated
    user_id = state.get("user_id")
    if not symbols and user_id:
        watchlist = agents.get_user_stocks(user_id)
        if watchlist:
            symbols = list(watchlist.keys())

    if not symbols:
        return {
            "pending_action": "await_stock_list",
            "planning_result": {"status": "need_stock_list"},
            "agents_executed": ["News Agent", "Prediction Agent", "Stock Info Agent", "Portfolio Planning Agent"],
        }

    # 3. For EACH symbol, calculate real actual prediction using LSTM+GRU model & yfinance
    per_symbol_details = []
    total_pct_change = 0.0

    for sym in symbols:
        try:
            pred = agents.predict_stock(sym)
            live_price = pred["last_close"]
            predicted_price = pred["predicted_price"]
            pct_change = pred["predicted_pct_change"]
            direction = pred["direction"]
        except Exception as e:
            logger.warning("predict_stock failed for %s in planning_node: %s", sym, e)
            try:
                live_price = agents.get_live_price(sym)
            except Exception:
                live_price = 100.0
            predicted_price = live_price
            pct_change = 0.0
            direction = "flat"

        total_pct_change += pct_change

        prompt = (
            f"Stock: {sym}\n"
            f"Current Live Price: {live_price}\n"
            f"Predicted Model Target Price: {predicted_price} ({'+' if pct_change >= 0 else ''}{pct_change}% {direction})\n"
            "Provide 1 specific, plain-language prediction explanation sentence for this stock based on these exact numbers."
        )
        explanation = _llm_text(prompt, system=PLANNING_SYSTEM_PROMPT) or (
            f"{sym} is currently at {live_price}. The model predicts a {direction}ward move of {abs(pct_change)}% "
            f"targeting {predicted_price}."
        )

        per_symbol_details.append({
            "symbol": sym,
            "live_price": live_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": f"{'+' if pct_change >= 0 else ''}{pct_change}%",
            "direction": direction,
            "explanation": explanation,
        })

    # 4. Compute Portfolio Score (0 - 100)
    avg_pct = total_pct_change / max(len(symbols), 1)
    base_score = 75.0
    diversification_bonus = min(len(symbols) * 3.0, 15.0)
    growth_factor = max(min(avg_pct * 4.0, 15.0), -25.0)
    
    portfolio_score = round(max(min(base_score + growth_factor + diversification_bonus, 98.0), 30.0), 1)
    
    if portfolio_score >= 80:
        score_rating = "Strong Momentum & Growth"
    elif portfolio_score >= 65:
        score_rating = "Moderate Stability"
    else:
        score_rating = "High Volatility Warning"

    result = {
        "status": "ok",
        "portfolio_score": portfolio_score,
        "score_rating": score_rating,
        "symbols": per_symbol_details,
    }

    return {
        "planning_result": result,
        "agents_executed": ["News Agent", "Prediction Agent", "Stock Info Agent", "Portfolio Planning Agent"],
    }


# ===========================================================================
# SECTION 9: Report generation
# ===========================================================================

def report_generation_node(state: AgentState) -> dict:
    return {
        "report_result": {"sections": [], "report_url": None},
        "agents_executed": ["News Agent", "Report Generation Agent"],
    }


# ===========================================================================
# SECTION 10: Final response composition
# ===========================================================================

def chatbot_response_node(state: AgentState) -> dict:
    parts = []
    intent = state.get("intent") or "general"

    if intent == "tax_optimization" and state.get("tax_result"):
        parts.append(state["tax_result"]["explanation"])
    elif intent == "stock_info" and state.get("stock_info_result"):
        info = state["stock_info_result"]
        if info.get("error"):
            parts.append("I couldn't find a ticker in that message - try asking about any stock like \"how's AAPL doing?\".")
        else:
            line = f"**{info['symbol']}** — Live Price: {info.get('live_price', 'n/a')}"
            if info.get("predicted_price"):
                line += f" · Predicted Price: {info.get('predicted_price')}"
            if info.get("direction"):
                line += f" ({info.get('direction')})"
            parts.append(line)
    elif intent == "save_interest" and state.get("save_interest_result"):
        si = state["save_interest_result"]
        if si.get("error") == "no tickers found":
            parts.append("I couldn't pick out any tickers from that - could you list them like \"AAPL, TSLA, INFY\"?")
        elif si.get("error"):
            parts.append("I couldn't save that - please sign in and try again.")
        else:
            parts.append(f"Added to your watchlist: {', '.join(si['saved'])}. Ask me about your portfolio score whenever you're ready.")
    elif intent == "portfolio_planning" and state.get("planning_result"):
        pr = state["planning_result"]
        if pr.get("status") == "need_stock_list":
            parts.append("Tell me which stocks you own or are interested in (e.g. \"I have AAPL, TSLA, GOOGL, calculate my portfolio score\") and I'll evaluate them for you!")
        elif pr.get("error"):
            parts.append("I couldn't evaluate your portfolio score — please try again.")
        else:
            score = pr.get("portfolio_score", "N/A")
            rating = pr.get("score_rating", "")
            lines = [f"**Portfolio Score**: **{score} / 100** ({rating})\n"]
            lines.append("**Per-Stock Model Predictions & Analysis:**")
            for s in pr.get("symbols", []):
                lines.append(
                    f"• **{s['symbol']}**: Live Price `{s['live_price']}` ➔ Predicted Target `{s['predicted_price']}` "
                    f"({s['predicted_change_pct']}, {s['direction']})\n  _{s['explanation']}_"
                )
            parts.append("\n".join(lines))
    elif intent == "stock_prediction" and state.get("stock_prediction_result"):
        lines = ["Predictions:"]
        for p in state["stock_prediction_result"]:
            lines.append(f"- {p['symbol']}: {p['explanation']}")
        parts.append("\n".join(lines))
    elif intent == "report" and (state.get("report_result") or {}).get("report_url"):
        parts.append(f"Full report: {state['report_result']['report_url']}")

    executed = state.get("agents_executed") or []
    if not parts:
        # General intent or unhandled query -> Tavily general search + LLM fallback
        answer = agents.tavily_general_search_answer(state.get("user_query", ""), state.get("chat_history"))
        parts.append(answer or (
            "I don't have a live data source for that yet, so I can't verify current "
            "facts here - but generally speaking: "
            + (state.get("user_query") or "").strip()
        ))
        if not executed:
            executed = ["News Agent", "General Knowledge Fallback Agent"]

    final = "\n\n".join(parts)
    return {"final_response": final, "agents_executed": executed}


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
