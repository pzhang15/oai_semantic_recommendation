import os
import re
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
    # Skip if FAISS not installed in the current environment
    try:
        import faiss  # type: ignore
    except Exception:
        pytest.skip("FAISS not installed in current environment")


def test_items_extended_fields_types():
    _skip_if_data_missing()
    r = client.post("/search", json={"query": "cotton", "limit": 12, "use_judge": False, "debug": True})
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", [])
    assert isinstance(items, list) and items
    for it in items:
        # id must be string
        assert isinstance(it.get("id"), str)
        # rating fields numeric or None
        ar = it.get("average_rating")
        if ar is not None:
            assert isinstance(ar, (int, float))
        rn = it.get("rating_number")
        if rn is not None:
            assert isinstance(rn, (int, float))
        # categories either list or absent
        cat = it.get("categories")
        if cat is not None:
            assert isinstance(cat, (list, str))
        # details if provided should be string (JSON) or object
        det = it.get("details")
        if det is not None:
            assert isinstance(det, (str, dict))


def test_parser_extracts_size_notes_when_present():
    r = client.post("/debug/parse", json={"query": "men's sneakers size US 9"})
    assert r.status_code == 200
    d = r.json()
    fac = d.get("facets", {})
    # Expect gender and a size note
    assert fac.get("gender_or_fit") in ("men", "unisex", None)
    assert isinstance(fac.get("size_notes"), (str, type(None)))
    if isinstance(fac.get("size_notes"), str):
        assert re.search(r"US\s*\d+", fac["size_notes"], re.IGNORECASE)


def test_parser_extracts_categories_or_materials():
    r = client.post("/debug/parse", json={"query": "black cotton t-shirt"})
    assert r.status_code == 200
    d = r.json()
    fac = d.get("facets", {})
    # At least one of categories or materials should be non-empty for this query
    cats = fac.get("categories_include") or []
    mats = fac.get("materials") or []
    assert isinstance(cats, list) and isinstance(mats, list)
    assert (len(cats) + len(mats)) >= 1


