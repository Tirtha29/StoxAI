"""
Core domain layer for the tax-saving agent: data contracts, the
rules-based tax logic, and the LLM explanation step.

Merged from (formerly separate) schemas.py + rules_engine.py + llm.py so
the whole tax domain lives in one place. Sections below are still clearly
separated - split them back out if this ever needs to grow independently.

THE CONTRACT (read this first): every agent (stock prediction, tax
saving, report generation, chatbot) reads from / writes to the SAME
`AgentState` TypedDict defined below. Add your own fields to AgentState
instead of inventing a parallel state object - that's what keeps
LangGraph routing simple.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Data contracts (formerly schemas.py)
# ===========================================================================

# --- Portfolio / user profile - shared by tax and stock prediction ---

class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float
    buy_price: float
    buy_date: date
    current_price: float
    asset_type: Literal["equity", "mutual_fund", "etf", "bond", "crypto"] = "equity"

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.buy_price

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def holding_days(self) -> int:
        return (date.today() - self.buy_date).days


class TaxProfile(BaseModel):
    filing_status: Literal["single", "married_joint", "married_separate", "hoi"] = "single"
    annual_income: float = 0.0
    country: Literal["IN", "US"] = "IN"
    realized_gains_ytd: float = 0.0
    realized_losses_ytd: float = 0.0
    section_80c_used: float = 0.0          # India only
    retirement_contrib_ytd: float = 0.0    # US only (401k/IRA combined)


# --- Tax Saving agent - request / response contract ---

class TaxSavingRequest(BaseModel):
    portfolio: List[PortfolioHolding]
    tax_profile: TaxProfile


class TaxSuggestion(BaseModel):
    type: Literal[
        "tax_loss_harvest",
        "hold_for_ltcg",
        "deduction_80c",
        "deduction_retirement",
    ]
    symbol: Optional[str] = None
    title: str
    detail: str
    estimated_savings: float = 0.0
    source_snippets: List[str] = Field(default_factory=list)  # RAG citations


class TaxSavingResponse(BaseModel):
    suggestions: List[TaxSuggestion]
    total_estimated_savings: float
    explanation: str  # LLM-generated natural language summary, RAG-grounded


# --- Contracts for other agents (stock prediction / report) ---
# Field names are kept stable so graph.py doesn't need rewiring if these
# get filled in more fully later.

class StockPredictionResult(BaseModel):
    symbol: str
    predicted_direction: Literal["up", "down", "flat"]
    predicted_range: Optional[List[float]] = None
    confidence: float = 0.0
    explanation: str = ""


class ReportBundle(BaseModel):
    report_url: Optional[str] = None
    sections: List[str] = Field(default_factory=list)


# --- Shared LangGraph state - THE central object passed between every node ---

class AgentState(TypedDict, total=False):
    # input
    user_query: str
    chat_history: List[dict]
    user_id: Optional[str]      # from the JWT, set by api.py - keys the Mongo watchlist
    user_email: Optional[str]   # from the JWT, used as the alert-email recipient

    # routing
    # "tax_optimization" | "stock_info" | "save_interest" | "portfolio_planning"
    # | "stock_prediction" | "report" | "general"
    intent: Optional[str]
    # set by planning_node when it had to stop and ask the user for their
    # watchlist; router_node checks this BEFORE keyword-matching so the
    # very next message is treated as the stock list, not re-routed.
    pending_action: Optional[str]

    # shared data
    portfolio: List[dict]       # serialized PortfolioHolding list (tax / stock-prediction agents)
    tax_profile: dict           # serialized TaxProfile
    watchlist: Optional[dict]   # {SYMBOL: {live_price, predicted_price, updated_at}} from Mongo

    # per-agent outputs (each agent only writes its own key)
    tax_result: Optional[dict]                       # serialized TaxSavingResponse
    stock_prediction_result: Optional[List[dict]]
    stock_info_result: Optional[dict]                # single-symbol lookup for "how's X doing"
    save_interest_result: Optional[dict]             # watchlist after a save_interest turn
    planning_result: Optional[dict]                  # portfolio-score / "what should I do" suggestions
    report_result: Optional[dict]

    # final
    final_response: Optional[str]


# ===========================================================================
# SECTION 2: Rules-based tax logic (formerly rules_engine.py)
# Deliberately NOT a full tax-law model - a hackathon-scoped approximation.
# ===========================================================================

def find_tax_loss_harvest_candidates(portfolio: List[PortfolioHolding]) -> List[TaxSuggestion]:
    """Positions currently below cost basis that could be sold to realize a loss."""
    suggestions = []
    for h in portfolio:
        if h.unrealized_pnl < 0 and h.holding_days >= config.MIN_HOLDING_DAYS_BEFORE_HARVEST_SUGGESTION:
            loss = abs(h.unrealized_pnl)
            # crude marginal-rate assumption for the demo; refine with real slabs later
            assumed_rate = 0.20
            est_savings = round(loss * assumed_rate, 2)
            suggestions.append(TaxSuggestion(
                type="tax_loss_harvest",
                symbol=h.symbol,
                title=f"Harvest loss on {h.symbol}",
                detail=(
                    f"{h.symbol} is down {loss:,.2f} from cost basis "
                    f"({h.cost_basis:,.2f} -> {h.market_value:,.2f}). Selling now realizes "
                    f"the loss, which can offset capital gains elsewhere this year."
                ),
                estimated_savings=est_savings,
            ))
    return suggestions


def find_ltcg_hold_suggestions(portfolio: List[PortfolioHolding]) -> List[TaxSuggestion]:
    """Positions close to crossing the long-term capital gains threshold."""
    suggestions = []
    window_days = 30  # "close to" window
    for h in portfolio:
        if h.unrealized_pnl > 0:
            days_to_ltcg = config.LTCG_HOLDING_DAYS - h.holding_days
            if 0 < days_to_ltcg <= window_days:
                # rough rate delta between short-term and long-term treatment
                assumed_delta_rate = 0.10
                est_savings = round(h.unrealized_pnl * assumed_delta_rate, 2)
                suggestions.append(TaxSuggestion(
                    type="hold_for_ltcg",
                    symbol=h.symbol,
                    title=f"Wait {days_to_ltcg} days before selling {h.symbol}",
                    detail=(
                        f"{h.symbol} crosses the long-term holding threshold in "
                        f"{days_to_ltcg} day(s). Selling after that date can lower the "
                        f"tax rate applied to the {h.unrealized_pnl:,.2f} gain."
                    ),
                    estimated_savings=est_savings,
                ))
    return suggestions


def find_deduction_suggestions(profile: TaxProfile) -> List[TaxSuggestion]:
    """Remaining deduction/contribution room (rules differ by country)."""
    suggestions = []

    if profile.country == "IN":
        remaining = max(config.SECTION_80C_LIMIT - profile.section_80c_used, 0)
        if remaining > 0:
            assumed_rate = 0.20
            est_savings = round(remaining * assumed_rate, 2)
            suggestions.append(TaxSuggestion(
                type="deduction_80c",
                title="Unused Section 80C room",
                detail=(
                    f"You have {remaining:,.2f} of unused Section 80C limit "
                    f"(ELSS, PPF, life insurance, etc.). Investing the remainder "
                    f"reduces taxable income."
                ),
                estimated_savings=est_savings,
            ))

    elif profile.country == "US":
        limit = config.US_401K_LIMIT + config.US_IRA_LIMIT
        remaining = max(limit - profile.retirement_contrib_ytd, 0)
        if remaining > 0:
            assumed_rate = 0.22
            est_savings = round(remaining * assumed_rate, 2)
            suggestions.append(TaxSuggestion(
                type="deduction_retirement",
                title="Unused 401(k) / IRA contribution room",
                detail=(
                    f"You have {remaining:,.2f} of combined 401(k)/IRA room left "
                    f"this year. Contributing the remainder lowers taxable income."
                ),
                estimated_savings=est_savings,
            ))

    return suggestions


def generate_all_suggestions(portfolio: List[PortfolioHolding], profile: TaxProfile) -> List[TaxSuggestion]:
    suggestions: List[TaxSuggestion] = []
    suggestions += find_tax_loss_harvest_candidates(portfolio)
    suggestions += find_ltcg_hold_suggestions(portfolio)
    suggestions += find_deduction_suggestions(profile)
    return suggestions


# ===========================================================================
# SECTION 3: LLM explanation layer (formerly llm.py)
# Falls back to a deterministic template (no API call) if no API key is
# set, so the graph still runs for anyone who hasn't exported
# ANTHROPIC_API_KEY yet.
# ===========================================================================

EXPLANATION_SYSTEM_PROMPT = """You are the explanation layer of a tax-saving agent in a \
financial assistant app. You are given a list of tax-saving suggestions \
that were computed by a deterministic rules engine (NOT by you), each \
with retrieved snippets from authoritative tax-rule documents.

Rules:
- Do not invent numbers, thresholds, or rules that aren't in the provided \
suggestions or retrieved snippets.
- Every claim about a rule (holding periods, limits, wash-sale windows, \
contribution caps) must be traceable to a retrieved snippet - if a snippet \
doesn't cover something, don't state it as fact.
- Write for a retail investor, not a tax professional: plain language, \
short sentences, no jargon left unexplained.
- Lead with the single most valuable action, then the rest.
- End with one sentence on the total estimated savings.
- Keep it under 180 words.
- Do not add a disclaimer about consulting a tax professional unless the \
suggestions involve a genuinely ambiguous judgment call.
"""


def generate_grounded_explanation(suggestions: List[dict], total_savings: float) -> str:
    if not suggestions:
        return "No tax-saving opportunities found for the current portfolio and profile."

    if not config.LLM_API_KEY:
        return _template_fallback(suggestions, total_savings)

    try:
        return _call_anthropic(suggestions, total_savings)
    except Exception:
        # never let an LLM/network hiccup break the demo
        return _template_fallback(suggestions, total_savings)


def _call_anthropic(suggestions: List[dict], total_savings: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.LLM_API_KEY)

    context_blocks = []
    for s in suggestions:
        snippets = "\n".join(f"  - {sn}" for sn in s.get("source_snippets", [])) or "  (no supporting snippet retrieved)"
        context_blocks.append(
            f"Suggestion: {s['title']}\n"
            f"Type: {s['type']}\n"
            f"Computed detail: {s['detail']}\n"
            f"Estimated savings: {s['estimated_savings']}\n"
            f"Retrieved rule snippets:\n{snippets}"
        )

    user_content = (
        f"Total estimated savings across all suggestions: {total_savings}\n\n"
        + "\n\n".join(context_blocks)
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=400,
        system=EXPLANATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _template_fallback(suggestions: List[dict], total_savings: float) -> str:
    lines = [f"Found {len(suggestions)} tax-saving opportunity(ies), estimated total savings {total_savings}."]
    for s in suggestions:
        lines.append(f"- {s['title']}: {s['detail']}")
    return "\n".join(lines)
