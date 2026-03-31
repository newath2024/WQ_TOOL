[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GoogleDriveRoot,

    [string]$ProjectRoot = "",

    [int]$KeepLatest = 48,

    [switch]$IncludeBackups,

    [switch]$NoCompress
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Join-Path (Split-Path -Parent $PSCommandPath) ".."
}

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [switch]$AllowMissing
    )

    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if (Test-Path -LiteralPath $expanded) {
        return (Resolve-Path -LiteralPath $expanded).Path
    }
    if (-not $AllowMissing) {
        throw "Path not found: $Path"
    }
    if ([System.IO.Path]::IsPathRooted($expanded)) {
        return [System.IO.Path]::GetFullPath($expanded)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $expanded))
}

function Get-PythonExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return $python.Source
    }

    throw "Python executable not found. Expected .venv\Scripts\python.exe or python on PATH."
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,

        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $baseUri = New-Object System.Uri(($BasePath.TrimEnd("\") + "\"))
    $targetUri = New-Object System.Uri($TargetPath)
    return [System.Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($targetUri).ToString().Replace("/", "\")
    )
}

function Invoke-SqliteSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExecutable,

        [Parameter(Mandatory = $true)]
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    $backupScript = @'
import pathlib
import sqlite3
import sys

src = pathlib.Path(sys.argv[1]).resolve()
dst = pathlib.Path(sys.argv[2]).resolve()
dst.parent.mkdir(parents=True, exist_ok=True)

src_conn = sqlite3.connect(str(src), timeout=30)
dst_conn = sqlite3.connect(str(dst), timeout=30)

try:
    src_conn.execute("PRAGMA busy_timeout = 30000")
    dst_conn.execute("PRAGMA busy_timeout = 30000")
    src_conn.backup(dst_conn)
finally:
    dst_conn.close()
    src_conn.close()
'@

    $backupScript | & $PythonExecutable - $SourcePath $DestinationPath
    if ($LASTEXITCODE -ne 0) {
        throw "SQLite snapshot failed for $SourcePath"
    }
}

function Copy-DirectoryTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,

        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot,

        [string[]]$ExcludeRelativePaths = @()
    )

    if (-not (Test-Path -LiteralPath $SourceRoot)) {
        return
    }

    $sourceRootResolved = (Resolve-Path -LiteralPath $SourceRoot).Path
    $destinationRootResolved = Resolve-AbsolutePath -Path $DestinationRoot -AllowMissing
    New-Item -ItemType Directory -Path $destinationRootResolved -Force | Out-Null

    Get-ChildItem -LiteralPath $sourceRootResolved -Recurse -Force -File | ForEach-Object {
        $relativePath = Get-RelativePath -BasePath $sourceRootResolved -TargetPath $_.FullName
        if ($ExcludeRelativePaths -contains $relativePath) {
            return
        }

        $targetPath = Join-Path $destinationRootResolved $relativePath
        $targetParent = Split-Path -Parent $targetPath
        if (-not [string]::IsNullOrWhiteSpace($targetParent)) {
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $targetPath -Force
    }
}

function Get-GitMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        return @{
            branch = $null
            head = $null
            status = @()
        }
    }

    Push-Location $Root
    try {
        $branch = (& git branch --show-current 2>$null)
        $head = (& git rev-parse HEAD 2>$null)
        $status = @(& git status --short 2>$null)
        return @{
            branch = ($branch | Select-Object -First 1)
            head = ($head | Select-Object -First 1)
            status = $status
        }
    }
    finally {
        Pop-Location
    }
}

$projectRootResolved = Resolve-AbsolutePath -Path $ProjectRoot
$googleDriveRootResolved = Resolve-AbsolutePath -Path $GoogleDriveRoot -AllowMissing

New-Item -ItemType Directory -Path $googleDriveRootResolved -Force | Out-Null

$backupRoot = Join-Path $googleDriveRootResolved "WQ_TOOL_backups"
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

$pythonExecutable = Get-PythonExecutable -Root $projectRootResolved
$timestampUtc = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$snapshotName = "wq-tool-backup-$timestampUtc"
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("wq-tool-backup-" + [Guid]::NewGuid().ToString("N"))
$snapshotRoot = Join-Path $stagingRoot $snapshotName
$artifactPath = if ($NoCompress) { Join-Path $backupRoot $snapshotName } else { Join-Path $backupRoot "$snapshotName.zip" }

$includedDirectories = @("outputs", "progress_logs", "config")
if ($IncludeBackups) {
    $includedDirectories += "backups"
}

$excludedRelativePaths = @(
    "brain_api_session.json"
)

New-Item -ItemType Directory -Path $snapshotRoot -Force | Out-Null

try {
    $dbOutputRoot = Join-Path $snapshotRoot "db"
    $dbSnapshots = @()
    Get-ChildItem -LiteralPath $projectRootResolved -Filter *.sqlite3 -File | ForEach-Object {
        $destinationPath = Join-Path $dbOutputRoot $_.Name
        Invoke-SqliteSnapshot -PythonExecutable $pythonExecutable -SourcePath $_.FullName -DestinationPath $destinationPath
        $dbSnapshots += @{
            source = $_.FullName
            artifact_relative_path = "db/$($_.Name)"
            bytes = $_.Length
        }
    }

    foreach ($relativeDir in $includedDirectories) {
        $sourceDir = Join-Path $projectRootResolved $relativeDir
        $destinationDir = Join-Path $snapshotRoot $relativeDir
        if ($relativeDir -eq "outputs") {
            Copy-DirectoryTree -SourceRoot $sourceDir -DestinationRoot $destinationDir -ExcludeRelativePaths $excludedRelativePaths
            continue
        }
        Copy-DirectoryTree -SourceRoot $sourceDir -DestinationRoot $destinationDir
    }

    foreach ($relativeFile in @("README.md", "pyproject.toml", "main.py", ".gitignore")) {
        $sourceFile = Join-Path $projectRootResolved $relativeFile
        if (-not (Test-Path -LiteralPath $sourceFile)) {
            continue
        }
        Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $snapshotRoot $relativeFile) -Force
    }

    $gitMetadata = Get-GitMetadata -Root $projectRootResolved
    $manifest = @{
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        snapshot_name = $snapshotName
        project_root = $projectRootResolved
        backup_root = $backupRoot
        included_directories = $includedDirectories
        excluded_output_relative_paths = $excludedRelativePaths
        sqlite_snapshots = $dbSnapshots
        git = $gitMetadata
    }

    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $snapshotRoot "manifest.json") -Encoding UTF8

    if (Test-Path -LiteralPath $artifactPath) {
        Remove-Item -LiteralPath $artifactPath -Force
    }

    if ($NoCompress) {
        Copy-Item -LiteralPath $snapshotRoot -Destination $artifactPath -Recurse -Force
    }
    else {
        Compress-Archive -LiteralPath $snapshotRoot -DestinationPath $artifactPath -CompressionLevel Optimal -Force
    }

    if ($KeepLatest -gt 0) {
        $backupArtifacts = Get-ChildItem -LiteralPath $backupRoot -Force |
            Where-Object { $_.Name -like "wq-tool-backup-*" } |
            Sort-Object LastWriteTimeUtc -Descending

        $backupArtifacts | Select-Object -Skip $KeepLatest | ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
        }
    }

    Write-Host "Backup created: $artifactPath"
    Write-Host "SQLite snapshots: $($dbSnapshots.Count)"
    Write-Host "Included directories: $($includedDirectories -join ', ')"
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
