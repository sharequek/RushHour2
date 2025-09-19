[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Join-Path $PSScriptRoot '..'
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
  Write-Error 'Virtualenv not found. Run scripts/setup.ps1 first.'
  exit 1
}

Write-Host "Dropping and recreating database schema with timing fields..." -ForegroundColor Yellow

# Run the database recreation
& $python -c "
import asyncio
import sys
sys.path.append('src')
from db import make_pool, drop_and_recreate_schema

async def main():
    pool = await make_pool()
    await drop_and_recreate_schema(pool)
    await pool.close()
    print('Database schema recreated successfully with timing fields')

asyncio.run(main())
"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Database schema recreated successfully!" -ForegroundColor Green
} else {
    Write-Error "Failed to recreate database schema"
    exit 1
}




