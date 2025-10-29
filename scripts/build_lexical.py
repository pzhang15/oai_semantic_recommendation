from pathlib import Path
from src.core.lexical_build import build_lexical


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out = build_lexical(repo_root)
    print(out)


if __name__ == "__main__":
    main()


