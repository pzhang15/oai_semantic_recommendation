import os
import time
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


def test_judge_cache_hit_on_repeat_query(monkeypatch: pytest.MonkeyPatch):
    _skip_if_data_missing()
    # Ensure cost guardrail does not force skip
    monkeypatch.setenv("JUDGE_COST_PER_REQ_USD_MAX", "10")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    body = {"query": "summer outfit", "limit": 12, "debug": True, "use_judge": True}
    r1 = client.post("/search", json=body)
    assert r1.status_code == 200, r1.text
    tr1 = r1.json().get("trace", {})
    jg1 = tr1.get("judge_gate", {})
    # First call may or may not hit cache; do not assert

    time.sleep(0.2)
    r2 = client.post("/search", json=body)
    assert r2.status_code == 200, r2.text
    tr2 = r2.json().get("trace", {})
    jg2 = tr2.get("judge_gate", {})
    gate_hit = (jg2.get("cache", {}) or {}).get("hit") is True
    per_item_hits = int((tr2.get("judge", {}) or {}).get("cached_hits", 0)) > 0
    assert gate_hit or per_item_hits


