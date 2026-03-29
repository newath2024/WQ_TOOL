$scriptPath = Join-Path $PSScriptRoot "wq_export_operators_expanded.js"
Set-Clipboard -Value (Get-Content $scriptPath -Raw)
Write-Host "Copied expanded operators exporter from $scriptPath to clipboard."
