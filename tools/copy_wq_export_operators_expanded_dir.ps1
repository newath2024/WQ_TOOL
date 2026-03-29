$scriptPath = Join-Path $PSScriptRoot "wq_export_operators_expanded.js"
$prefix = "globalThis.WQ_EXPORT_OPTIONS = { saveToDirectory: true };" + [Environment]::NewLine
$payload = $prefix + (Get-Content $scriptPath -Raw)
Set-Clipboard -Value $payload
Write-Host "Copied expanded operators exporter with directory-save mode from $scriptPath to clipboard."
