$scriptPath = Join-Path $PSScriptRoot "wq_export_table.js"
$prefix = "globalThis.WQ_EXPORT_OPTIONS = { forceDom: true };" + [Environment]::NewLine
$payload = $prefix + (Get-Content $scriptPath -Raw)
Set-Clipboard -Value $payload
Write-Host "Copied DOM mode exporter from $scriptPath to clipboard."
