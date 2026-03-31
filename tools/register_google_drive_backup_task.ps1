[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GoogleDriveRoot,

    [string]$TaskName = "WQToolGoogleDriveBackup",

    [int]$EveryMinutes = 30,

    [string]$ProjectRoot = "",

    [int]$KeepLatest = 48,

    [switch]$IncludeBackups
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Join-Path (Split-Path -Parent $PSCommandPath) ".."
}

if ($EveryMinutes -le 0) {
    throw "EveryMinutes must be > 0."
}

$powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$backupScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "backup_to_google_drive.ps1")).Path
$projectRootResolved = if (Test-Path -LiteralPath $ProjectRoot) {
    (Resolve-Path -LiteralPath $ProjectRoot).Path
} else {
    [System.IO.Path]::GetFullPath($ProjectRoot)
}

$argumentList = @(
    "-NoProfile"
    "-ExecutionPolicy", "Bypass"
    "-File", "`"$backupScript`""
    "-ProjectRoot", "`"$projectRootResolved`""
    "-GoogleDriveRoot", "`"$GoogleDriveRoot`""
    "-KeepLatest", $KeepLatest
)

if ($IncludeBackups) {
    $argumentList += "-IncludeBackups"
}

$taskCommand = "`"$powershellExe`" " + ($argumentList -join " ")

schtasks.exe /Create /F /SC MINUTE /MO $EveryMinutes /TN $TaskName /TR $taskCommand | Out-Null

Write-Host "Scheduled task created: $TaskName"
Write-Host "Runs every $EveryMinutes minute(s)."
Write-Host "Command: $taskCommand"
