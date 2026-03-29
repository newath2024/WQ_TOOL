# WQ Browser Export

`WQ Tool` does not ship with direct credentials-based scraping for WorldQuant BRAIN data pages.

The local BRAIN API documentation bundled in this repo documents `/authentication`, `/simulations`, and `/alphas`, but it does not document stable public endpoints for the Data Explorer tables. Because of that, the safest approach is:

1. Sign in to WorldQuant BRAIN in your normal browser session.
2. Open the exact `Datasets` or dataset `Fields` page you want.
3. Set `region`, `delay`, `universe`, search filters, and page size in the UI first.
4. Run the browser-side exporter script from this repo.

This approach uses your existing browser session and exports the table already rendered in the UI, page by page.

## Script location

- [tools/wq_export_table.js](/d:/WQ_TOOL/tools/wq_export_table.js)

## What it exports

The script walks all visible pages using the `Next` button and downloads:

- `*.csv`
- `*.json`

If your browser supports the File System Access API, the script can also write
directly into a folder you choose instead of always using the default
`Downloads` directory.

The export includes:

- all table rows across every page
- current page URL
- current query parameters
- page title
- page count
- row count

The current page number is also added as `__page`.

## How to run

1. Open a WQ Data Explorer table page.
2. Optionally set the UI page size to the maximum available.
3. Open `DevTools -> Console`.
4. Open [tools/wq_export_table.js](/d:/WQ_TOOL/tools/wq_export_table.js) locally.
5. Copy the file contents and paste them into the browser console.
6. Wait for the script to click through all pages and download the files.

## PowerShell shortcut

To copy the exporter script into your clipboard from the repo root:

```powershell
Get-Content .\tools\wq_export_table.js -Raw | Set-Clipboard
```

Then paste into the browser console.

To let Chrome ask for a destination folder and save files there directly:

```powershell
.\tools\copy_wq_export_dir.ps1
```

This copies a small prefix plus the exporter script so the browser runs with:

```javascript
globalThis.WQ_EXPORT_OPTIONS = { saveToDirectory: true };
```

When you paste and run it in DevTools Console, Chrome will open a folder picker.
Choose a local directory once and the exporter will write both the CSV and JSON
into that folder.

To force visible UI pagination by clicking `Next` in the browser:

```powershell
.\tools\copy_wq_export_dom.ps1
```

This copies a small prefix plus the exporter script so the browser runs with:

```javascript
globalThis.WQ_EXPORT_OPTIONS = { forceDom: true };
```

Use this mode when you specifically want to see the page move in the UI.

To combine visible UI pagination with direct folder saving:

```powershell
.\tools\copy_wq_export_dom_dir.ps1
```

## Suggested workflow

For dataset catalog pages:

- export one file per `category + region + delay + universe`

For dataset field pages:

- export one file per dataset id and settings combination

Suggested naming is automatic, but you can rename the downloads afterward and move them into a local folder such as:

```text
inputs/
  wq/
    datasets/
    fields/
```

If you use directory-save mode, choose one of those folders directly in the
picker and the exporter will write there immediately.

## Limits

- This exports table metadata, not necessarily raw point-in-time market values.
- If the WQ frontend changes its table or pagination DOM, the script may need a small selector update.
- Do not export or store cookies, auth headers, or credentials in this repo.
