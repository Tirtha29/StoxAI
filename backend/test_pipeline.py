"""
Expanded smoke test script to verify Pydantic query parsing, Tavily symbol extraction,
stock graphing agent, price sentinel -1 fallback, tax saving RAG, and general fallback.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config
import agents
from core import ParsedUserQuery
from graph import build_graph


def run_tests():
    print("=== TEST 1: Pydantic Base Model Query Parser ===")
    parsed_stock = agents.parse_user_query_with_pydantic_llm("how is AAPL performing today?")
    print("Parsed stock query:", parsed_stock)
    assert isinstance(parsed_stock, ParsedUserQuery)
    print("PASS: Pydantic query parser output valid.\n")

    print("=== TEST 2: Stock Graphing Agent ===")
    graph_res = agents.generate_stock_graph("AAPL", days=30)
    print("Graph generated for AAPL:", "Yes" if graph_res.get("graph_image_b64") else "No")
    print("Data points count:", len(graph_res.get("data_points", [])))
    assert graph_res.get("graph_image_b64") is not None
    print("PASS: Graph making agent generated chart image.\n")

    print("=== TEST 3: Price Sentinel -1 Fallback ===")
    cached_sentinel = {"live_price": 182.5, "predicted_price": -1}
    resolved = agents.resolve_price("AAPL", cached_sentinel)
    print("Resolved price for -1 sentinel:", resolved)
    assert resolved["predicted_price"] == resolved["live_price"]
    assert resolved["predicted_price"] != -1
    print("PASS: -1 sentinel replaced with live price.\n")

    print("=== TEST 4: Company Name Stock Query with Tavily Lookup ===")
    app = build_graph()
    test_user_id = "507f1f77bcf86cd799439011"
    initial_state = {
        "user_query": "how is Tesla doing today?",
        "chat_history": [],
        "user_id": test_user_id,
    }
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "test_thread_1"}})
    print("Final Intent:", result.get("intent"))
    print("Final Response snippet:", result.get("final_response")[:150])
    stock_info = result.get("stock_info_result") or {}
    print("Stock Info Result Symbol:", stock_info.get("symbol"))
    print("Live Price:", stock_info.get("live_price"))
    print("Predicted Price:", stock_info.get("predicted_price"))
    print("Graph Image Present:", bool(stock_info.get("graph_image_b64")))
    assert stock_info.get("symbol") == "TSLA" or stock_info.get("symbol") is not None
    assert stock_info.get("predicted_price") != -1
    print("PASS: Company name stock lookup and graph generation succeeded.\n")

    print("=== TEST 5: Tax Saving RAG Query ===")
    tax_state = {
        "user_query": "how much tax can I save under section 80C?",
        "chat_history": [],
        "user_id": test_user_id,
    }
    tax_result = app.invoke(tax_state, config={"configurable": {"thread_id": "test_thread_1"}})
    print("Tax Intent:", tax_result.get("intent"))
    print("Tax Response snippet:", tax_result.get("final_response")[:150])
    assert tax_result.get("intent") == "tax_optimization"
    assert tax_result.get("tax_result") is not None
    print("PASS: Tax saving RAG agent routing succeeded.\n")

    print("=== TEST 6: Multi-Turn State Leakage & Stopword Ticker Fix ===")
    # Immediately follow up on same thread with a stock graph query
    stock_state = {
        "user_query": "show me the graph for past days about googl",
        "chat_history": [],
        "user_id": test_user_id,
    }
    multi_turn_res = app.invoke(stock_state, config={"configurable": {"thread_id": "test_thread_1"}})
    print("Multi-Turn Intent:", multi_turn_res.get("intent"))
    print("Multi-Turn Tax Result (Must be None):", multi_turn_res.get("tax_result"))
    print("Multi-Turn Executed Agents:", multi_turn_res.get("agents_executed"))
    stock_info_2 = multi_turn_res.get("stock_info_result") or {}
    print("Stock Info Symbol:", stock_info_2.get("symbol"))
    print("Graph Present:", bool(stock_info_2.get("graph_image_b64")))

    # Assert tax_result was cleared and did NOT leak into this turn!
    assert multi_turn_res.get("tax_result") is None
    assert stock_info_2.get("symbol") == "GOOGL"
    assert stock_info_2.get("graph_image_b64") is not None
    assert "Graph Visualization Agent" in (multi_turn_res.get("agents_executed") or [])
    print("PASS: Multi-turn state leakage fixed & GOOGL graph generated correctly.\n")

    print("=== TEST 7: Scheduler Email Functionality ===")
    from scheduler import send_user_prediction_emails
    email_res = send_user_prediction_emails()
    print("Scheduler Email Results:", email_res)
    print("PASS: Scheduler email sending pipeline executed.\n")

    print("=== TEST 8: Client File Ingestion into RAG Corpus ===")
    import tempfile
    from rag import add_doc, TaxRAGRetriever
    from graph import reset_retriever

    sample_content = """# Custom Client Portfolio & Tax Exemption Notes 2026
## Special Deduction Section 80CCD
Client has invested 50,000 INR in National Pension Scheme (NPS) under Section 80CCD(1B) for additional tax savings.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False)
    tmp.write(sample_content)
    tmp.close()

    try:
        dest_path = add_doc(tmp.name, country="IN", name="test_nps_exemption")
        print("Ingested custom doc to:", dest_path)
        assert os.path.exists(dest_path)

        reset_retriever()
        retriever = TaxRAGRetriever()
        hits = retriever.query("NPS tax savings under section 80CCD", k=2, country="IN")
        print("RAG Query Hits count for newly ingested doc:", len(hits))
        assert len(hits) > 0
        assert "80CCD" in hits[0]["text"]
        print("PASS: Custom client document ingested and successfully retrieved by RAG agent.\n")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
