Param()

Write-Host "[clean] Removing frontend\node_modules if present..."
if (Test-Path "frontend\node_modules") {
  Remove-Item -Recurse -Force "frontend\node_modules"
}

Write-Host "[clean] Removing Python __pycache__ directories..."
Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
  try { Remove-Item -Recurse -Force $_.FullName } catch {}
}

Write-Host "[clean] Removing .venv (virtualenv) if present..."
if (Test-Path ".venv") {
  Remove-Item -Recurse -Force ".venv"
}

Write-Host "[clean] Removing .git directory (version control metadata)..."
if (Test-Path ".git") {
  Remove-Item -Recurse -Force ".git"
}

Write-Host "[clean] Done. Repository is ready to zip."


