<#
.SYNOPSIS
  One-time, elevated: let WSL2 reach the SoulsBench bridge through Windows Defender Firewall.

.DESCRIPTION
  Under WSL2's default NAT networking, WSL talks to Windows over the "vEthernet (WSL ...)" adapter, so
  the bridge's python.exe needs an inbound rule. The first time it listens, Windows usually answers
  with an automatic *Block* rule (which beats any Allow rule). This script:
    1. removes the Block rules for the bridge's python.exe only,
    2. adds one Allow rule: that python.exe, TCP <Port>, remote address LocalSubnet only.
  The bridge binds 127.0.0.1 and the WSL adapter only, so nothing on your LAN can reach it.

  Run from an elevated PowerShell (or let it re-launch itself elevated):
    powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu\home\<you>\Sources\SoulsBench\windows\allow_wsl_firewall.ps1
#>
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA "soulsbench"),
    [int]$Port = 47800
)
$ErrorActionPreference = "Stop"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Re-launching elevated (UAC prompt)..."
    # Elevated processes may not see \\wsl.localhost, so elevate a local copy of this script.
    $local = Join-Path $env:TEMP "soulsbench_allow_wsl_firewall.ps1"
    Copy-Item -LiteralPath $PSCommandPath -Destination $local -Force
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$local`" -Prefix `"$Prefix`" -Port $Port"
    $p = Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait -PassThru
    Remove-Item $local -ErrorAction SilentlyContinue
    exit $p.ExitCode
}

$VenvPy = Join-Path $Prefix "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "No bridge venv at $VenvPy. Run setup_bridge.ps1 first." }
# A uv venv's python.exe is a launcher; the process that listens is the base interpreter.
$Base = (& $VenvPy -c "import sys; print(sys._base_executable)").Trim()
$Name = "SoulsBench bridge (WSL to Windows)"

$blocks = Get-NetFirewallApplicationFilter | Where-Object { $_.Program -eq $Base } |
    Get-NetFirewallRule | Where-Object { $_.Action -eq "Block" -and $_.Direction -eq "Inbound" }
foreach ($r in $blocks) {
    Write-Host "Removing auto-created block rule: $($r.DisplayName) ($($r.Name))"
    Remove-NetFirewallRule -Name $r.Name
}
Get-NetFirewallRule -DisplayName $Name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $Name -Direction Inbound -Action Allow -Protocol TCP `
    -LocalPort $Port -Program $Base -RemoteAddress LocalSubnet -Profile Any | Out-Null
Write-Host "Allowed inbound TCP $Port for $Base from the local subnet (WSL)."
