#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1; then
  docker compose up --build
else
  echo "Docker not found. Starting API locally..."
  python scripts/preflight.py || true
  python scripts/bootstrap_demo.py || true
  uvicorn src.app:app --host 0.0.0.0 --port 8000
fi


