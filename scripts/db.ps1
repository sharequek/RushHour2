[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)] [switch] $Clear
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Join-Path $PSScriptRoot '..'
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
  Write-Error 'Virtualenv not found. Run scripts/setup.ps1 first.'
  exit 1
}

if ($Clear) {
  & $python -m src.db --clear-listings
} else {
  Write-Host 'Usage:'
  Write-Host '  ./scripts/db.ps1 -Clear'
}


