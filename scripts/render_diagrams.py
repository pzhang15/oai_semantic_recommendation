import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FILES = [
    (DOCS / "architecture.mmd", [DOCS / "architecture.png", DOCS / "architecture.pdf"]),
    (DOCS / "search-sequence.mmd", [DOCS / "search-sequence.png"]),
    (DOCS / "outfit-sequence.mmd", [DOCS / "outfit-sequence.png"]),
]


def main() -> None:
    missing = [str(src) for (src, _) in FILES if not src.exists()]
    if missing:
        print("[ERROR] Missing diagram sources:", ", ".join(missing))
        raise SystemExit(1)

    node = shutil.which("node")
    mmdc = shutil.which("mmdc")
    if not node or not mmdc:
        print("[WARN] Mermaid CLI not found. Install and re-run:")
        print("  - Install Node (LTS): https://nodejs.org/")
        print("  - Install Mermaid CLI: npm i -g @mermaid-js/mermaid-cli")
        raise SystemExit(1)

    exported = []
    for src, outs in FILES:
        for out in outs:
            cmd = [
                mmdc,
                "-i", str(src),
                "-o", str(out),
                "-t", "default",
            ]
            try:
                subprocess.run(cmd, check=True)
                exported.append(out)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] mmdc failed for {src}: {e}")
                raise SystemExit(1)
    print("[OK] Diagrams exported:")
    for p in exported:
        print(" -", p)


if __name__ == "__main__":
    main()
