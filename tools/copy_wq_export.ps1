$scriptPath = Join-Path $PSScriptRoot "wq_export_table.js"
Get-Content $scriptPath -Raw | Set-Clipboard
Write-Host "Copied $scriptPath to clipboard."
