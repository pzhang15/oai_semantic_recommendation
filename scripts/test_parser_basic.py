import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.parser import parse_query  # type: ignore


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}")
        raise SystemExit(1)


def main() -> None:
    f1 = parse_query("beach trip under $120")
    assert_true(f1.budget_usd.max is not None and f1.budget_usd.max <= 120 + 1e-6, "budget max <= 120")

    f2 = parse_query("avoid polyester")
    assert_true(any("polyester" in x for x in (f2.hard_constraints or [])), "hard_constraints include polyester")

    f3 = parse_query("linen shirt")
    assert_true("linen" in (f3.materials or []), "materials include linen")

    f4 = parse_query("men's winter outfit")
    assert_true((f4.gender_or_fit == "men"), "gender_or_fit men")
    assert_true((f4.season in {"winter", "cold-weather"}), "season winter/cold-weather")

    print("[OK] parser_basic")


if __name__ == "__main__":
    main()


