import os
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.core.config import get_settings


def _skip_if_data_missing():
    s = get_settings()
    for p in [s.index_path, s.ids_path, s.parquet_path]:
        if not os.path.exists(p):
            pytest.skip(f"Missing artifact: {p}")


def test_judge_cost_guardrail(monkeypatch: pytest.MonkeyPatch):
    _skip_if_data_missing()
    # Force a very low per-request cost cap
    monkeypatch.setenv("JUDGE_COST_PER_REQ_USD_MAX", "0.000001")
    # Refresh settings cache
    get_settings.cache_clear()  # type: ignore[attr-defined]
    s = get_settings()
    assert float(s.judge_cost_per_req_usd_max) <= 0.000001

    client = TestClient(app)
    r = client.post("/search", json={"query": "broad clothing", "limit": 12, "debug": True, "use_judge": True})
    assert r.status_code == 200, r.text
    jg = r.json().get("trace", {}).get("judge_gate", {})
    assert jg, "missing judge_gate in trace"
    assert jg.get("mode") == "skip" and jg.get("reason") == "cost_guardrail"


