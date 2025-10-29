from src.core.fusion import weighted_rrf_fuse


def test_weighted_rrf_simple():
    base = ["a", "b", "c"]
    det = ["c", "d", "e"]
    rankings = {"q0": base, "q1": det}
    # Favor base more than det
    fused = weighted_rrf_fuse(rankings, {"q0": 1.0, "q1": 0.5}, k0=60, topn=5)
    # 'b' should rank above 'd' typically due to higher weight and earlier rank
    assert fused.index("b") < fused.index("d")


