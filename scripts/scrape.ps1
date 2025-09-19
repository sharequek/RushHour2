[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)] [string] $StartUrl,
  [Parameter(Mandatory=$false)] [int] $MaxPages,
  [Parameter(Mandatory=$false)] [int] $MaxLinks,
  [Parameter(Mandatory=$false)] [int] $Concurrency,
  [Parameter(Mandatory=$false)] [string] $DetailUrl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Join-Path $PSScriptRoot '..'
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
  Write-Error 'Virtualenv not found. Run scripts/setup.ps1 first.'
  exit 1
}

${argsList} = @()
if ($PSBoundParameters.ContainsKey('StartUrl')) { ${argsList} += @('--start-url', $StartUrl) }
if ($PSBoundParameters.ContainsKey('Concurrency')) { ${argsList} += @('--concurrency', $Concurrency) }
if ($PSBoundParameters.ContainsKey('MaxPages')) { ${argsList} += @('--max-pages', $MaxPages) }
if ($PSBoundParameters.ContainsKey('MaxLinks')) { ${argsList} += @('--max-links', $MaxLinks) }
if ($PSBoundParameters.ContainsKey('DetailUrl')) { ${argsList} += @('--detail-url', $DetailUrl) }

& $python -m src.scrape @argsList
