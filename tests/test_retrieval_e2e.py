import os
import math
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.core.config import get_settings
from src.core.lexical import ready as lexical_ready


client = TestClient(app)


def _skip_if_data_missing():
    s = get_settings()
    for p in [s.index_path, s.ids_path, s.parquet_path]:
        if not os.path.exists(p):
            pytest.skip(f"Missing artifact: {p}")


def test_unstructured_semantic_retrieval_basic():
    _skip_if_data_missing()
    body = {"query": "linen shirt", "limit": 12, "use_judge": False, "debug": True}
    r = client.post("/search", json=body)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("items"), list)
    assert len(data["items"]) > 0
    # ensure scores numeric
    for it in data["items"]:
        assert isinstance(it.get("score"), (int, float))
    # timings present
    tr = data.get("trace", {})
    assert "retrieval" in tr


def test_structured_budget_filter_enforced():
    _skip_if_data_missing()
    s = get_settings()
    q = "summer outfit under $80"
    r = client.post("/search", json={"query": q, "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    data = r.json()
    # Allow tolerance from settings
    tol = float(getattr(s, "price_tolerance", 0.15))
    # Try to extract budget from parsed facets in trace if present
    mx = None
    tr = data.get("trace", {})
    parsed = (tr.get("parsed") or {})
    bud = (parsed.get("budget_usd") or {})
    if isinstance(bud, dict) and isinstance(bud.get("max"), (int, float)):
        mx = float(bud["max"])
    # If budget extracted, validate items within tolerance
    if mx:
        for it in data["items"]:
            p = it.get("price")
            if p is not None and isinstance(p, (int, float)):
                assert p <= mx * (1.0 + tol) + 1e-6


def test_pagination_stability_with_result_id():
    _skip_if_data_missing()
    q = "cotton t-shirt"
    r1 = client.post("/search", json={"query": q, "page": 1, "limit": 12, "use_judge": False, "debug": True})
    assert r1.status_code == 200
    d1 = r1.json()
    rid = d1.get("result_id")
    assert rid is not None
    r2 = client.post("/search", json={"query": q, "page": 2, "limit": 12, "use_judge": False, "debug": True, "result_id": rid})
    assert r2.status_code == 200
    d2 = r2.json()
    ids1 = [it["id"] for it in d1.get("items", [])]
    ids2 = [it["id"] for it in d2.get("items", [])]
    # Expect disjoint pages (unless fewer than 13 items available)
    if len(ids1) == 12 and len(ids2) == 12:
        assert set(ids1).isdisjoint(set(ids2))


def test_mmr_diversity_no_duplicates():
    _skip_if_data_missing()
    q = "black dress"
    r = client.post("/search", json={"query": q, "limit": 20, "use_judge": False, "debug": True})
    assert r.status_code == 200
    d = r.json()
    ids = [it["id"] for it in d.get("items", [])]
    assert len(ids) == len(set(ids))


def test_hybrid_rrf_trace_when_ready():
    _skip_if_data_missing()
    # Only assert when lexical artifacts exist
    if not lexical_ready().get("ready"):
        pytest.skip("Lexical artifacts not ready")
    r = client.post("/search", json={"query": "tencel merino", "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    tr = r.json().get("trace", {})
    hyb = (tr.get("retrieval", {}) or {}).get("hybrid", {})
    assert hyb.get("enabled") is True
    assert hyb.get("fused_unique", 0) >= len(r.json().get("items", []))


