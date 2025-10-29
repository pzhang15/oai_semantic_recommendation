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


def test_search_with_deterministic_expansion(monkeypatch: pytest.MonkeyPatch):
    _skip_if_data_missing()
    if not lexical_ready().get("ready"):
        pytest.skip("Lexical not ready")
    # Enable expansion
    monkeypatch.setenv("EXPANSION_ENABLED", "true")
    # Keep hybrid enabled path
    get_settings.cache_clear()  # type: ignore[attr-defined]
    r = client.post("/search", json={"query": "docs boots", "limit": 12, "debug": True, "use_judge": False})
    assert r.status_code == 200
    tr = r.json().get("trace", {})
    hy = (tr.get("retrieval") or {}).get("hybrid") or {}
    exp = hy.get("expansion") or {}
    assert exp.get("enabled") is True
    # weights should include q0 and q1 entries
    w = exp.get("weights") or {}
    assert any(k.startswith("q1_") for k in w.keys())


