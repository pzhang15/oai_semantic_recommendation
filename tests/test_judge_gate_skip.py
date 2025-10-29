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


def test_judge_gate_skip_high_specificity(monkeypatch: pytest.MonkeyPatch):
    _skip_if_data_missing()
    if not lexical_ready().get("ready"):
        pytest.skip("Lexical artifacts not ready; jaccard signal required for skip test")

    # Soften skip thresholds to make decision deterministic in CI
    monkeypatch.setenv("JUDGE_COST_PER_REQ_USD_MAX", "10")
    monkeypatch.setenv("JUDGE_MARGIN_SKIP_MIN", "0.0")
    monkeypatch.setenv("JUDGE_ENTROPY_SKIP_MAX", "10.0")
    monkeypatch.setenv("JUDGE_JACCARD_SKIP_MIN", "0.0")
    monkeypatch.setenv("JUDGE_BUDGET_OK_SKIP_MIN", "0.0")
    monkeypatch.setenv("JUDGE_CATEGORY_OK_SKIP_MIN", "0.0")
    monkeypatch.setenv("JUDGE_DUP_SKIP_MAX", "1.0")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    body = {"query": "women linen blouse under $60", "limit": 12, "debug": True, "use_judge": True}
    r = client.post("/search", json=body)
    assert r.status_code == 200, r.text
    tr = r.json().get("trace", {})
    jg = tr.get("judge_gate", {})
    assert jg, "missing judge_gate in trace"
    assert jg.get("mode") == "skip"
    # judge time should be tiny when skipped
    ms = float((tr.get("timings", {}) or {}).get("judge_ms", 0.0))
    assert ms <= 50.0


