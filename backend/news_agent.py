"""
News agent: fetches recent finance news for a symbol and ingests it into
the SAME RAG corpus rag.py already builds from data/tax_docs/*.md (via
rag.add_doc, completely unchanged/untouched).

CRITICAL - how this is (and is NOT) wired up:
  - graph.py's router NEVER calls anything in this file. No user chat
    message triggers a news fetch. This is intentional per your
    instruction: news fetching only happens when *we* call it.
  - No email is sent from this module, ever.
  - You trigger ingestion in one of three ways:
      1. CLI, locally or as a one-off Render Shell command:
           python news_agent.py AAPL TSLA INFY
      2. A Render Cron Job service running the same command on a
         schedule (e.g. daily) - this is the recommended way to keep the
         RAG corpus fresh in production without touching user traffic.
      3. The admin-only HTTP route in main.py: POST /admin/ingest-news
         with header `X-Admin-Key: <config.ADMIN_API_KEY>` - use this if
         you want to trigger a refresh remotely (e.g. from a Render Cron
         Job that just curls your own service, or manually from Postman)
         without SSHing in. It is NOT reachable by a logged-in end user;
         it does not accept or use the user's JWT at all.

Once ingested, the news becomes retrievable the same way any tax doc is -
through rag.TaxRAGRetriever. Nothing in graph.py automatically queries
this news content for a user's question either: per your instruction,
ordinary user questions are answered from the model's own general
knowledge (see graph.py's chatbot_response_node), not by silently
triggering a fetch. If you later want a node that also searches this
ingested news corpus, that's a deliberate follow-up change, not something
this merge turns on by default.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

import config
from rag import add_doc

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Finance-news client - wraps Finnhub's `/company-news`
# endpoint.
# ===========================================================================

@dataclass
class Article:
    title: str
    description: str
    source: str
    url: str
    published_at: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.description or ''}".strip()


def fetch_articles(symbol: str, company_name: Optional[str] = None) -> List[Article]:
    """
    Returns recent finance-news articles mentioning the symbol/company.
    Returns an empty list (never raises) if FINNHUB_API_KEY isn't set or the
    request fails.
    """
    if not config.FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set - fetch_articles(%s) returning no results", symbol)
        return []

    symbol = symbol.strip().upper()
    if not symbol:
        logger.warning("fetch_articles called without a symbol")
        return []

    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=config.NEWS_LOOKBACK_DAYS)

    params = {
        "symbol": symbol,
        "from": since.isoformat(),
        "to": today.isoformat(),
        "token": config.FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(f"{config.FINNHUB_API_BASE_URL}/company-news", params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("fetch_articles(%s) failed: %s", symbol, e)
        return []

    if not isinstance(payload, list):
        logger.warning("fetch_articles(%s) returned an unexpected response shape", symbol)
        return []

    articles = []
    for item in payload[:config.NEWS_ARTICLES_PER_SYMBOL]:
        if not isinstance(item, dict):
            continue
        published_at = item.get("datetime")
        if isinstance(published_at, (int, float)):
            published_at = datetime.fromtimestamp(published_at, tz=timezone.utc).isoformat()
        articles.append(Article(
            title=item.get("headline") or "",
            description=item.get("summary") or "",
            source=item.get("source") or "unknown",
            url=item.get("url", ""),
            published_at=published_at or "",
        ))
    return articles


# ===========================================================================
# SECTION 2: RAG ingestion - articles -> markdown -> rag.add_doc()
# ===========================================================================

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "news"


def _articles_to_markdown(symbol: str, articles: List[Article]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"# Recent news - {symbol} ({today})", ""]
    if not articles:
        lines.append("No recent articles were found for this symbol at ingestion time.")
        return "\n".join(lines)

    for a in articles:
        lines.append(f"## {a.title or '(untitled)'}")
        lines.append(f"Source: {a.source} | Published: {a.published_at or 'unknown'}")
        if a.description:
            lines.append(a.description)
        if a.url:
            lines.append(f"Link: {a.url}")
        lines.append("")

    return "\n".join(lines)


def ingest_news_for_symbol(symbol: str) -> dict:
    """
    Fetch news for one symbol and write it into the RAG corpus via
    rag.add_doc() (unchanged). Returns a small status dict. Never sends
    email. Safe to call repeatedly - each run gets a timestamped doc name
    so it won't collide with (or silently overwrite) a previous ingest.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol must not be empty")

    articles = fetch_articles(symbol)
    markdown = _articles_to_markdown(symbol, articles)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    doc_name = f"news_{_slugify(symbol)}_{ts}"

    tmp_dir = tempfile.mkdtemp(prefix="news_ingest_")
    tmp_path = os.path.join(tmp_dir, f"{doc_name}.md")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    try:
        # country=None -> not tax-jurisdiction-specific, so it's retrieved
        # regardless of the user's country filter in TaxRAGRetriever.query()
        dest_path = add_doc(tmp_path, country=None, name=doc_name)
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return {
        "symbol": symbol,
        "articles_fetched": len(articles),
        "ingested_path": dest_path,
    }


def ingest_news_for_symbols(symbols: List[str]) -> dict:
    results = []
    for symbol in symbols:
        try:
            results.append(ingest_news_for_symbol(symbol))
        except Exception as e:
            logger.error("ingest_news_for_symbol(%s) failed: %s", symbol, e)
            results.append({"symbol": symbol.strip().upper(), "error": str(e)})
    return {"status": "ok", "results": results}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    symbols = sys.argv[1:]
    if not symbols:
        print("Usage: python news_agent.py TICKER [TICKER ...]")
        sys.exit(1)

    summary = ingest_news_for_symbols(symbols)
    for r in summary["results"]:
        if "error" in r:
            print(f"  {r['symbol']}: FAILED - {r['error']}")
        else:
            print(f"  {r['symbol']}: ingested {r['articles_fetched']} article(s) -> {r['ingested_path']}")
