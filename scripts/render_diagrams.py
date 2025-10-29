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
    (DOCS / "full-sequence.mmd", [DOCS / "full-sequence.png"]),
    (DOCS / "file-map.mmd", [DOCS / "file-map.png"]),
]

# PNG quality controls (env overrides)
PNG_SCALE = float(os.getenv("DIAG_SCALE", "2.5"))  # ~2–3 gives crisp PNGs
PNG_WIDTH = os.getenv("DIAG_WIDTH")  # optional absolute width in px (e.g., 2400)


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
            cmd = [mmdc, "-i", str(src), "-o", str(out), "-t", "default"]
            # Improve raster quality for PNGs
            if str(out).lower().endswith(".png"):
                cmd += ["-s", f"{PNG_SCALE}"]
                if PNG_WIDTH:
                    cmd += ["-w", str(PNG_WIDTH)]
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
