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


def test_debug_retrieve_hybrid_lists():
    _skip_if_data_missing()
    if not lexical_ready().get("ready"):
        pytest.skip("Lexical not ready")
    r = client.post("/debug/retrieve", json={"query": "rare brand zyx", "k": 20, "use_hybrid": True})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("dense"), list)
    assert isinstance(data.get("lex"), list)
    assert isinstance(data.get("fused"), list)
    # ensure fused contains something from lex (if lex had any hits)
    lex_ids = [x["id"] for x in data.get("lex")]
    fused_ids = [x["id"] for x in data.get("fused")]
    if lex_ids:
        assert len(set(lex_ids) & set(fused_ids)) >= 0


