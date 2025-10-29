import os
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


def test_result_id_present_page1():
    _skip_if_data_missing()
    r = client.post("/search", json={"query": "simple shirt", "page": 1, "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    assert r.json().get("result_id")


def test_healthz_lexical_key_present():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    # lexical stanza must be a dict with 'ready'
    lx = data.get("lexical")
    assert isinstance(lx, dict)
    assert "ready" in lx


def test_debug_retrieve_respects_k():
    _skip_if_data_missing()
    r = client.post("/debug/retrieve", json={"query": "cotton", "k": 5, "use_hybrid": False})
    assert r.status_code == 200
    d = r.json()
    assert len(d.get("dense", [])) <= 5


def test_hybrid_disabled_when_not_ready():
    _skip_if_data_missing()
    if lexical_ready().get("ready"):
        pytest.skip("Lexical ready; this test checks the disabled case")
    r = client.post("/search", json={"query": "rare token xyzbrand", "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    tr = r.json().get("trace", {})
    hyb = (tr.get("retrieval", {}) or {}).get("hybrid", {})
    # hybrid entry exists but enabled is False when no artifacts
    assert isinstance(hyb, dict)
    assert hyb.get("enabled") in (False, None)


def test_parser_fallback_allows_response():
    _skip_if_data_missing()
    # Craft a prompt that may produce odd schema; route should still 200 via fallback
    r = client.post("/search", json={"query": "### $$$ !!!", "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d.get("items"), list)
    assert d.get("summary")


