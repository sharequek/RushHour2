[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$venvPython = Join-Path $PSScriptRoot '..' | Join-Path -ChildPath '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
  Write-Host 'Creating virtual environment (.venv)...'
  python -m venv (Join-Path $PSScriptRoot '..' | Join-Path -ChildPath '.venv')
}

Write-Host 'Upgrading pip and installing requirements...'
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r (Join-Path $PSScriptRoot '..' | Join-Path -ChildPath 'requirements.txt')

Write-Host 'Installing Playwright browsers...'
& $venvPython -m playwright install

Write-Host 'Setup complete.'


