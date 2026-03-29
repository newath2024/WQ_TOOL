$scriptPath = Join-Path $PSScriptRoot "wq_export_table.js"
$prefix = "globalThis.WQ_EXPORT_OPTIONS = { saveToDirectory: true };" + [Environment]::NewLine
$payload = $prefix + (Get-Content $scriptPath -Raw)
Set-Clipboard -Value $payload
Write-Host "Copied directory-save exporter from $scriptPath to clipboard."
