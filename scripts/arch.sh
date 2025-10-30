#!/usr/bin/env bash
set -euo pipefail

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to render diagrams. Install Node.js LTS from https://nodejs.org/" >&2
  exit 1
fi

# Ensure mermaid-cli is available via devDependency
npm run arch || (npm install --no-audit --no-fund && npm run arch)

echo "[OK] Rendered docs/architecture/diagram.pdf and .png"


