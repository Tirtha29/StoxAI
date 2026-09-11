"""
scheduler.py — nightly refresh of stock predictions + news, for every
symbol any user has on their watchlist.

WHAT IT DOES (once per run) — two independent steps, order doesn't matter:

  NEWS: ingests broad finance/stock-market news into the RAG corpus -
  NOT scoped to any user's watchlist. Calls Finnhub's general market feed
  directly (see _fetch_general_market_news() below) rather than going
  through news_agent.py, since that file's fetch_articles() is hardwired
  to Finnhub's per-symbol endpoint and is kept untouched on purpose.
  Watchlist-only would only ever cover tickers someone happened to
  favorite; rag.py's retrieval already finds whatever's relevant to a
  given question at query time, so broad nightly coverage is more useful
  here. Always runs.

  PREDICTIONS: only runs at all if at least one user has at least one
  ticker stored (has_any_favorites() gate) — no point loading the
  LSTM/GRU models if no one has favorited anything yet. When it does
  run: for each distinct watchlist ticker, runs agents.predict_stock()
  once, then writes the fresh predicted_price back onto every user who
  has that ticker on their watchlist via agents.store_predicted_price().
  This clears the -1 "not predicted yet" sentinel for all of them at
  once per ticker, instead of re-running inference per-user.

HOW TO RUN IT:

  Default — self-scheduling, runs forever:
      python scheduler.py
    Starts APScheduler internally and fires the job every day at 19:00 in
    the timezone set by SCHEDULER_TIMEZONE (default Asia/Kolkata), then
    keeps the process alive waiting for the next 7pm. Needs `apscheduler`
    installed (see below).

  Testing / one-shot:
      python scheduler.py --once
    Runs the job immediately, once, and exits. No scheduling involved.

DEPLOYING THE DEFAULT (self-scheduling) MODE ON RENDER — IMPORTANT:
  Because this script now keeps its own clock, it must run as a service
  type that stays alive 24/7 — a Render **Background Worker**, not a free
  Web Service. Render's free/small web services can spin down when idle,
  and a spun-down process can't fire its internal 7pm timer. Set the
  Background Worker's start command to `python scheduler.py` and give it
  the same environment variables as your main backend (MONGODB_URI,
  FINNHUB_API_KEY, STOCK_MODEL_DIR, etc. from config.py).

  Alternative, if you'd rather not pay for an always-on worker: use
  `python scheduler.py --once` as the command on a Render **Cron Job**
  instead, and let Render's own scheduler (not this script) decide when
  it runs. Render Cron Jobs run in UTC — for 7:00 PM IST (UTC+5:30) that
  cron expression is `30 13 * * *`.

ADD TO requirements.txt (needed now — this is the default path, not optional):
    apscheduler
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import List

import requests

import config
from database import users_collection
from agents import predict_stock, store_predicted_price
from rag import add_doc  # only imported, not modified — same function news_agent.py already uses

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# STEP 1: figure out which symbols actually need refreshing
# ---------------------------------------------------------------------------

def get_all_watchlist_symbols() -> List[str]:
    """
    Distinct set of tickers across every user's `stocks` map. One user
    doc looks like {"stocks": {"AAPL": {...}, "TSLA": {...}}}; we just
    need the keys, deduplicated across all users, so predict_stock() runs
    once per ticker no matter how many users are watching it.
    """
    symbols = set()
    cursor = users_collection.find({}, {"stocks": 1})
    for doc in cursor:
        stocks = (doc or {}).get("stocks") or {}
        symbols.update(stocks.keys())
    return sorted(symbols)


def get_user_ids_watching(symbol: str) -> List[str]:
    """Every user_id (as a string) that has `symbol` on their watchlist."""
    cursor = users_collection.find({f"stocks.{symbol}": {"$exists": True}}, {"_id": 1})
    return [str(doc["_id"]) for doc in cursor]


def has_any_favorites() -> bool:
    """
    True if at least one user has at least one ticker stored on their
    `stocks` map. Used to gate the prediction step specifically — no
    point spinning up the LSTM/GRU models if nobody has favorited
    anything yet.
    """
    doc = users_collection.find_one(
        {"stocks": {"$exists": True, "$ne": {}}}, {"_id": 1}
    )
    return doc is not None


# ---------------------------------------------------------------------------
# STEP 2: refresh predictions
# ---------------------------------------------------------------------------

def refresh_predictions(symbols: List[str]) -> dict:
    results = {"succeeded": [], "failed": []}

    for symbol in symbols:
        try:
            prediction = predict_stock(symbol)  # runs LSTM + GRU once for this ticker
        except Exception as e:
            logger.error("predict_stock(%s) failed: %s", symbol, e)
            results["failed"].append({"symbol": symbol, "error": str(e)})
            continue

        user_ids = get_user_ids_watching(symbol)
        for user_id in user_ids:
            try:
                store_predicted_price(
                    user_id=user_id,
                    ticker=symbol,
                    predicted_price=prediction["predicted_price"],
                    live_price=prediction["last_close"],
                )
            except Exception as e:
                logger.error("store_predicted_price(%s, %s) failed: %s", user_id, symbol, e)

        logger.info(
            "%s: predicted %s (%.2f%%), pushed to %d user(s)",
            symbol, prediction["direction"], prediction["predicted_pct_change"], len(user_ids),
        )
        results["succeeded"].append({"symbol": symbol, "watchers_updated": len(user_ids), **prediction})

    return results


# ---------------------------------------------------------------------------
# STEP: refresh news — broad market/finance news, NOT scoped to any
# watchlist. Deliberately NOT routed through news_agent.py (kept
# byte-for-byte untouched) since its fetch_articles() is hardwired to
# Finnhub's per-symbol /company-news endpoint. This duplicates a small
# amount of Finnhub-calling + rag.add_doc() plumbing here instead of
# modifying that file. If news_agent.py ever grows a general-news
# function of its own, this can be deleted in favor of calling it.
# ---------------------------------------------------------------------------

def _fetch_general_market_news(category: str = "general") -> list:
    """Raw Finnhub /news (category feed) items — no ticker required."""
    if not config.FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set — skipping general news fetch")
        return []

    params = {"category": category, "token": config.FINNHUB_API_KEY}
    try:
        resp = requests.get(f"{config.FINNHUB_API_BASE_URL}/news", params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("General market news fetch failed: %s", e)
        return []

    if not isinstance(payload, list):
        logger.warning("General market news fetch returned an unexpected response shape")
        return []

    return payload[:config.NEWS_ARTICLES_PER_SYMBOL]


def _general_news_to_markdown(items: list) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"# General finance & market news ({today})", ""]
    if not items:
        lines.append("No recent general market articles were found at ingestion time.")
        return "\n".join(lines)

    for item in items:
        if not isinstance(item, dict):
            continue
        headline = item.get("headline") or "(untitled)"
        summary = item.get("summary") or ""
        source = item.get("source") or "unknown"
        url = item.get("url", "")
        published = item.get("datetime")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

        lines.append(f"## {headline}")
        lines.append(f"Source: {source} | Published: {published or 'unknown'}")
        if summary:
            lines.append(summary)
        if url:
            lines.append(f"Link: {url}")
        lines.append("")

    return "\n".join(lines)


def refresh_news() -> dict:
    """Fetch + ingest broad market news into the same RAG corpus rag.py
    already builds from data/tax_docs/*.md, via the unmodified add_doc()."""
    items = _fetch_general_market_news()
    markdown = _general_news_to_markdown(items)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    doc_name = f"news_general_{ts}"

    tmp_dir = tempfile.mkdtemp(prefix="news_ingest_")
    tmp_path = os.path.join(tmp_dir, f"{doc_name}.md")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    try:
        # country=None -> not tax-jurisdiction-specific, retrieved
        # regardless of the user's country filter in TaxRAGRetriever.query()
        dest_path = add_doc(tmp_path, country=None, name=doc_name)
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return {"articles_fetched": len(items), "ingested_path": dest_path}


# ---------------------------------------------------------------------------
# THE JOB — what actually runs, once, each time this fires
# ---------------------------------------------------------------------------

def run_daily_job():
    """
    Two independent steps — neither depends on the other's output, so the
    order they run in doesn't matter. They're just written top-to-bottom
    here for readability.
    """
    logger.info("=== Nightly refresh started ===")

    # --- News: always runs, broad market/finance coverage, not tied to
    # anyone's watchlist. ---
    news_results = refresh_news()
    logger.info(
        "News ingestion: %s article(s) ingested",
        news_results.get("articles_fetched", "?"),
    )

    # --- Predictions: only run if at least one user has stored favorites.
    # No watchlist entries anywhere -> no point loading the LSTM/GRU
    # models at all. ---
    if has_any_favorites():
        symbols = get_all_watchlist_symbols()
        logger.info("Refreshing predictions for %d distinct symbol(s): %s", len(symbols), ", ".join(symbols))
        prediction_results = refresh_predictions(symbols)
        logger.info(
            "Predictions: %d succeeded, %d failed",
            len(prediction_results["succeeded"]), len(prediction_results["failed"]),
        )
    else:
        logger.info("No user has any stored favorites yet — skipping prediction refresh.")
        prediction_results = {"succeeded": [], "failed": [], "skipped": True}

    logger.info("=== Nightly refresh finished ===")
    return {"predictions": prediction_results, "news": news_results}


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nightly stock prediction + news refresh")
    parser.add_argument(
        "--once", action="store_true",
        help="Run the job a single time and exit, instead of starting the daily 19:00 loop. "
             "Useful for testing, or if you're triggering this externally (e.g. Render Cron Job) "
             "instead of letting this script keep its own schedule.",
    )
    args = parser.parse_args()

    if args.once:
        run_daily_job()
        return

    # Default: start the built-in daily scheduler and keep running.
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "The daily scheduler needs APScheduler. Install it with:\n"
            "    pip install apscheduler\n"
            "and add `apscheduler` to requirements.txt.\n"
            "(Or run with --once to skip scheduling and just run the job now.)"
        )
        sys.exit(1)

    timezone = getattr(config, "SCHEDULER_TIMEZONE", "Asia/Kolkata")
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(run_daily_job, CronTrigger(hour=19, minute=0, timezone=timezone))

    logger.info("Scheduler started — will run daily at 19:00 (%s). Ctrl+C to stop.", timezone)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
