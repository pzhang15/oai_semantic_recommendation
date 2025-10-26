import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.parser import parse_query  # type: ignore


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}")
        raise SystemExit(1)


def approx_equal(a: float, b: float, eps: float = 1.0) -> bool:
    return abs(a - b) <= eps


def main() -> None:
    f1 = parse_query("$50-80 gym set, no logos")
    assert_true(f1.budget_usd.min is not None and f1.budget_usd.max is not None, "budget range parsed")
    assert_true(approx_equal(f1.budget_usd.min or 0, 50), "min ~= 50")
    assert_true(approx_equal(f1.budget_usd.max or 0, 80), "max ~= 80")
    assert_true(any("no logos" in x for x in (f1.hard_constraints or [])), "hard constraint no logos")
    assert_true((f1.occasion in {"gym"}), "occasion gym")

    f2 = parse_query("rainy day commute, waterproof boots")
    assert_true("waterproof" in (f2.must_have or []), "must_have waterproof")
    assert_true(any(c in (f2.categories_include or []) for c in ["boots", "sneakers", "jacket"]), "category present")

    f3 = parse_query("vegan leather jacket under 150")
    assert_true("vegan leather" in (f3.materials or []), "material vegan leather")
    assert_true("jacket" in (f3.categories_include or []) or f3.occasion in {"cold-weather", "rainy-day"}, "category or context present")
    assert_true((f3.budget_usd.max is not None and f3.budget_usd.max <= 150 + 1e-6), "budget max <= 150")

    print("[OK] parser_edgecases")


if __name__ == "__main__":
    main()


