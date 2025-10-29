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


def test_judge_gate_full_or_cheap(monkeypatch: pytest.MonkeyPatch):
    _skip_if_data_missing()
    # Ensure cost guardrail does not force skip
    monkeypatch.setenv("JUDGE_COST_PER_REQ_USD_MAX", "10")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    body = {"query": "summer outfit", "limit": 12, "debug": True, "use_judge": True}
    r = client.post("/search", json=body)
    assert r.status_code == 200, r.text
    tr = r.json().get("trace", {})
    jg = tr.get("judge_gate", {})
    assert jg, "missing judge_gate in trace"
    assert jg.get("mode") in {"cheap", "full"}
    # cheap may escalate; both acceptable
    assert isinstance(jg.get("escalated"), (bool, type(None)))


