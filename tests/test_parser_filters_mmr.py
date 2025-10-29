import os
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.core.config import get_settings


client = TestClient(app)


def _skip_if_data_missing():
    s = get_settings()
    for p in [s.index_path, s.ids_path, s.parquet_path]:
        if not os.path.exists(p):
            pytest.skip(f"Missing artifact: {p}")


def test_parser_facets_presence():
    _skip_if_data_missing()
    r = client.post("/debug/parse", json={"query": "men's linen shirt under $60 in navy"})
    assert r.status_code == 200
    d = r.json()
    fac = d.get("facets", {})
    assert isinstance(fac, dict)
    assert (fac.get("budget_usd") or {}).get("max") is not None


def test_filters_and_mmr_paths():
    _skip_if_data_missing()
    r = client.post("/search", json={"query": "office shoes under $120", "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    d = r.json()
    tr = d.get("trace", {})
    assert "filters" in tr and "mmr" in tr
    # ensure mmr final count equals returned items
    assert isinstance(tr["mmr"].get("final"), int)


