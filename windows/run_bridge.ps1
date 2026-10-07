<#
.SYNOPSIS
  Start the SoulsBench bridge: launch/attach Dark Souls III and serve CombatState over HTTP.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\Sources\SoulsBench\windows\run_bridge.ps1
  ... --no-launch        attach to an already running game only
  Extra arguments are passed to `python -m soulsbench.windows_bridge serve` (see --help).
#>
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA "soulsbench"),
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Rest
)
$ErrorActionPreference = "Stop"
$Py = Join-Path $Prefix "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "No bridge venv at $Py. Run windows\setup_bridge.ps1 first." }
& $Py -m soulsbench.windows_bridge serve @Rest
exit $LASTEXITCODE
