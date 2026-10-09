#requires -Version 7.2
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-PoolAgent {
    param([string]$Executable, [string]$Config)
    & $Executable start --config $Config | Out-Host
    return $LASTEXITCODE
}

function Start-PoolAgent {
    param([string]$Root = $PSScriptRoot)
    if (-not $env:BUILDKITE_AGENT_TOKEN) {
        throw "Supply BUILDKITE_AGENT_TOKEN at runtime (or file:// followed by a protected token file path)."
    }
    foreach ($file in @("provisioning.json", "environment.json", "buildkite-agent.cfg", "bin\buildkite-agent.exe")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $file) -PathType Leaf)) {
            throw "Pool provisioning is incomplete; missing $file under $Root."
        }
    }
    $configuration = Get-Content -LiteralPath (Join-Path $Root "environment.json") -Raw |
        ConvertFrom-Json -AsHashtable
    $original = @{}
    $originalPath = $env:PATH
    try {
        foreach ($entry in $configuration.environment.GetEnumerator()) {
            $original[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
        $env:PATH = (@($configuration.paths) + @($originalPath)) -join [IO.Path]::PathSeparator
        return Invoke-PoolAgent (Join-Path $Root "bin\buildkite-agent.exe") (Join-Path $Root "buildkite-agent.cfg")
    } finally {
        $env:PATH = $originalPath
        foreach ($entry in $original.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    exit (Start-PoolAgent)
}
