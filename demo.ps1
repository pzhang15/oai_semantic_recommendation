Param(
  [switch]$Headless
)

if (Get-Command docker -ErrorAction SilentlyContinue) {
  docker compose up --build
} else {
  Write-Host "Docker not found. Starting API locally..."
  try { python scripts/preflight.py } catch {}
  try { python scripts/bootstrap_demo.py } catch {}
  uvicorn src.app:app --host 0.0.0.0 --port 8000
}


