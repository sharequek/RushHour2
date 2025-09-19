param(
  [switch]$Install,
  [switch]$SeparateWindows,
  [switch]$NoTunnel,
  [switch]$Stop
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host $msg -ForegroundColor Red }

# Stop functionality
if ($Stop) {
  Write-Info "Stopping RushHour2 web services..."
  
  # Stop all related jobs
  $jobsToStop = @("RH2_API", "RH2_WEB", "RH2_TUNNEL")
  $runningJobs = Get-Job -Name $jobsToStop -ErrorAction SilentlyContinue
  
  if ($runningJobs) {
    Write-Info "Found running jobs: $($runningJobs.Name -join ', ')"
    Stop-Job -Name $jobsToStop -ErrorAction SilentlyContinue
    Remove-Job -Name $jobsToStop -ErrorAction SilentlyContinue
    Write-Info "All services stopped and cleaned up."
  } else {
    Write-Info "No running RushHour2 jobs found."
  }
  
  # Also try to stop any cloudflared processes
  $cloudflaredProcesses = Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue
  if ($cloudflaredProcesses) {
    Write-Info "Stopping cloudflared processes..."
    Stop-Process -Name "cloudflared" -Force -ErrorAction SilentlyContinue
    Write-Info "Cloudflare Tunnel stopped."
  }
  
  return
}

Write-Info "RushHour2 web: starting API and Web UI"

# Repo root
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

# Resolve Python / uvicorn
$venvPy = Join-Path $root ".venv/Scripts/python.exe"
$venvUv = Join-Path $root ".venv/Scripts/uvicorn.exe"
if (Test-Path $venvPy) {
  $python = $venvPy
} else {
  $python = 'python'
}

if (Test-Path $venvUv) {
  $uvicornPath = $venvUv
} else {
  $uvicornPath = $null
}
$useModule = -not $uvicornPath

# Check if cloudflared is available
$cloudflaredAvailable = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflaredAvailable) {
  Write-Warn "cloudflared not found. Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
  Write-Warn "Tunnel functionality will be disabled."
  $NoTunnel = $true
}

# Optional installs
if ($Install) {
  Write-Info "Installing Python deps (requirements.txt)"
  & $python -m pip install -U pip > $null
  & $python -m pip install -r requirements.txt

  if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Err "npm not found. Install Node.js from https://nodejs.org/"
    exit 1
  }

  Push-Location (Join-Path $root 'web')
  try {
    if (Test-Path package-lock.json) {
      Write-Info "Installing web deps (npm ci)"
      npm ci
    } else {
      Write-Info "Installing web deps (npm install)"
      npm install
    }
  } finally {
    Pop-Location
  }
}

if ($SeparateWindows) {
  Write-Info "Launching API in a new window..."
  if ($useModule) {
    Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `"$root`"; & `"$python`" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"
  } else {
    Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `"$root`"; & `"$uvicornPath`" api.main:app --host 127.0.0.1 --port 8000 --reload"
  }

  Write-Info "Launching Web UI in a new window..."
  Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `"$root/web`"; npm run dev"

  if (-not $NoTunnel -and $cloudflaredAvailable) {
    Write-Info "Launching Cloudflare Tunnel in a new window..."
    Start-Process powershell -ArgumentList "-NoExit","-Command","cloudflared tunnel run rushhour"
  }

  Write-Host ""; Write-Info "API: http://127.0.0.1:8000  |  Web: http://localhost:5173"
  if (-not $NoTunnel -and $cloudflaredAvailable) {
    Write-Info "Tunnel: Running in separate window (check for public URL)"
  }
  return
}

# Run both as background jobs
Write-Info "Starting API on http://127.0.0.1:8000"
$apiJob = Start-Job -Name RH2_API -ScriptBlock {
  param($rootPath, $py, $uvPath, $useMod)
  Set-Location $rootPath
  if ($useMod) {
    & $py -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
  } else {
    & $uvPath api.main:app --host 127.0.0.1 --port 8000 --reload
  }
} -ArgumentList $root, $python, $uvicornPath, $useModule

Write-Info "Starting Web UI on http://localhost:5173"
$webJob = Start-Job -Name RH2_WEB -ScriptBlock {
  param($rootPath)
  Set-Location (Join-Path $rootPath 'web')
  npm run dev
} -ArgumentList $root

# Start Cloudflare Tunnel if available and not disabled
if (-not $NoTunnel -and $cloudflaredAvailable) {
  Write-Info "Starting Cloudflare Tunnel (rushhour)"
  $tunnelJob = Start-Job -Name RH2_TUNNEL -ScriptBlock {
    cloudflared tunnel run rushhour
  }
  Write-Info "Tunnel started. Check tunnel output for public URL"
} else {
  $tunnelJob = $null
}

Write-Host ""
Write-Info "Running. Open: http://localhost:5173"
if ($tunnelJob) {
  Write-Host "Use Get-Job to list jobs; Receive-Job -Name RH2_API,RH2_WEB,RH2_TUNNEL for logs"
  Write-Host "Stop with: .\web.ps1 -Stop"
} else {
  Write-Host "Use Get-Job to list jobs; Receive-Job -Name RH2_API,RH2_WEB for logs"
  Write-Host "Stop with: .\web.ps1 -Stop"
}

