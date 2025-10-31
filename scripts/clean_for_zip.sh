#!/usr/bin/env bash
set -euo pipefail

echo "[clean] Removing frontend/node_modules if present..."
rm -rf ./frontend/node_modules || true

echo "[clean] Removing Python __pycache__ directories..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true

echo "[clean] Removing .venv (virtualenv) if present..."
rm -rf ./.venv || true

echo "[clean] Removing .git directory (version control metadata)..."
rm -rf ./.git || true

echo "[clean] Done. Repository is ready to zip."


