Param()

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  Write-Error "npm is required to render diagrams. Install Node.js LTS from https://nodejs.org/";
  exit 1
}

try {
  npm run arch
} catch {
  npm install --no-audit --no-fund
  npm run arch
}

Write-Host "[OK] Rendered docs/architecture/diagram.pdf and .png"


