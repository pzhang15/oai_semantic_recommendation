from src.core.expand_det import generate_deterministic_expansions


def _norm(xs):
    return [" ".join(x.split()).lower() for x in xs]


def test_brand_alias_docs_boots():
    q = "docs boots"
    ex = generate_deterministic_expansions(q, max_items=2)
    exn = _norm(ex)
    assert any("dr martens boots" in s for s in exn)


def test_taxonomy_parent_chelsea_boots_not_leather():
    q = "chelsea boots not leather"
    ex = generate_deterministic_expansions(q, max_items=2)
    exn = _norm(ex)
    # parent lift retains negation
    assert any(s.startswith("boots ") and ("not leather" in s) for s in exn)


def test_attribute_normalization_rain_proof():
    q = "rain proof jacket"
    ex = generate_deterministic_expansions(q, max_items=2)
    exn = _norm(ex)
    assert any("waterproof jacket" in s for s in exn)


def test_price_number_preserved():
    q = "under $80 sneakers"
    ex = generate_deterministic_expansions(q, max_items=2)
    exn = _norm(ex)
    # ensure the number 80 is not altered by rewrites
    assert all("80" in s or s == q.lower() for s in exn)


