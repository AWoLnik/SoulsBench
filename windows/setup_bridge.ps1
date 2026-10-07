<#
.SYNOPSIS
  One-time setup of the SoulsBench Windows bridge: CPython 3.12.15 venv + pinned SoulsGym + SoulsBench.

.DESCRIPTION
  Everything is kept under %LOCALAPPDATA%\soulsbench (no system Python, no PATH changes):
    bin\uv.exe        uv (downloads the managed CPython and installs packages)
    python\           uv-managed CPython 3.12.15 (64-bit)
    venv\             the bridge venv
    save_backups\     DS3 save backups made each time the bridge starts

  Run it from Windows PowerShell (not from inside WSL):
    powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\Sources\SoulsBench\windows\setup_bridge.ps1

  SoulsBench itself is installed editable from this repo, so edits made in WSL take effect the next time
  the bridge starts. Use -CopySource if you prefer a local copy (re-run setup to update it).
#>
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA "soulsbench"),
    [string]$PythonVersion = "3.12.15",
    [string]$UvVersion = "0.12.23",
    [switch]$Recreate,
    [switch]$CopySource
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath  # plain path, also for \\wsl.localhost UNC
$Bin = Join-Path $Prefix "bin"
$Venv = Join-Path $Prefix "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Uv = Join-Path $Bin "uv.exe"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

if (-not (Test-Path $Uv)) {
    Write-Host "Downloading uv $UvVersion ..."
    $zip = Join-Path $env:TEMP "uv-$UvVersion-soulsbench.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath $Bin -Force
    Remove-Item $zip
}

# Keep uv's managed Python and cache inside the prefix too.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Prefix "python"
$env:UV_CACHE_DIR = Join-Path $Prefix "cache"

if ($Recreate -and (Test-Path $Venv)) { Remove-Item -Recurse -Force $Venv }
if (-not (Test-Path $Py)) {
    Write-Host "Creating venv with CPython $PythonVersion ..."
    Invoke-Checked "uv venv" { & $Uv venv --python $PythonVersion $Venv }
}

Write-Host "Installing pinned bridge dependencies (SoulsGym @ 29c94d8) ..."
Invoke-Checked "dependency install" { & $Uv pip install --python $Py -r (Join-Path $Repo "windows\requirements-bridge.txt") }

if ($CopySource) {
    $Src = Join-Path $Prefix "src"
    if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
    New-Item -ItemType Directory -Force -Path $Src | Out-Null
    Copy-Item -Recurse (Join-Path $Repo "soulsbench") $Src
    Copy-Item (Join-Path $Repo "pyproject.toml"), (Join-Path $Repo "README.md") $Src
    Invoke-Checked "soulsbench install" { & $Uv pip install --python $Py --no-deps $Src }
} else {
    Invoke-Checked "soulsbench install" { & $Uv pip install --python $Py --no-deps -e $Repo }
}

Write-Host "`nBridge Python: $Py`n"
& $Py -m soulsbench.windows_bridge doctor
