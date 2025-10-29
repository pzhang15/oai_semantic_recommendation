from __future__ import annotations

from typing import Dict, List, Tuple


def rrf_fuse(rankings: Dict[str, List[str]], k0: int, topn: int) -> List[str]:
    scores: Dict[str, float] = {}
    best_rank: Dict[str, int] = {}
    for _, ids in rankings.items():
        for pos, pid in enumerate(ids, start=1):
            contrib = 1.0 / (k0 + pos)
            scores[pid] = scores.get(pid, 0.0) + contrib
            if (pid not in best_rank) or (pos < best_rank[pid]):
                best_rank[pid] = pos
    items: List[Tuple[str, float, int]] = [(pid, sc, best_rank.get(pid, 10**9)) for pid, sc in scores.items()]
    items.sort(key=lambda t: (-t[1], t[2], t[0]))
    return [pid for (pid, _, __) in items[:topn]]


def fuse_scored(dense: List[Dict[str, float]], lex: List[Dict[str, float]], k0: int, topn: int) -> List[str]:
    dense_ids = [d["id"] for d in dense]
    lex_ids = [l["id"] for l in lex]
    return rrf_fuse({"dense": dense_ids, "lex": lex_ids}, k0=k0, topn=topn)


