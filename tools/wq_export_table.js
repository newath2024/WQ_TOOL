/*
 * Export the current WorldQuant BRAIN Data Explorer result set.
 *
 * Strategy:
 * 1. Prefer the same API endpoint the page is already calling.
 * 2. Page through offsets until exhausted.
 * 3. Retry with backoff on 429 rate-limit responses.
 * 4. Fall back to DOM scraping if API access is unavailable.
 *
 * Usage:
 * - Open a datasets page, dataset fields page, or learn/operators page in
 *   WorldQuant BRAIN.
 * - Set region/delay/universe/search filters first.
 * - Open DevTools -> Console.
 * - Paste this script and press Enter.
 */

(async () => {
  const exportOptions =
    globalThis.WQ_EXPORT_OPTIONS && typeof globalThis.WQ_EXPORT_OPTIONS === "object"
      ? globalThis.WQ_EXPORT_OPTIONS
      : {};
  const FORCE_DOM = Boolean(exportOptions.forceDom || globalThis.WQ_EXPORT_FORCE_DOM);
  const SAVE_TO_DIRECTORY = Boolean(
    exportOptions.saveToDirectory ||
      exportOptions.pickDirectory ||
      globalThis.WQ_EXPORT_SAVE_TO_DIRECTORY,
  );
  const API_ORIGIN = "https://api.worldquantbrain.com";
  const DEFAULT_LIMIT = 50;
  const INTER_PAGE_DELAY_MS = 1500;
  const RATE_LIMIT_BACKOFF_MS = 15000;
  const NEXT_PAGE_TIMEOUT_MS = 30000;
  const MAX_API_RETRIES = 6;
  const MAX_DOM_NEXT_RETRIES = 4;
  const DOM_SETTLE_MS = 1800;
  const OPERATOR_EXPAND_SETTLE_MS = 350;
  const MAX_OPERATOR_EXPAND_CLICKS = 200;
  const DEFAULT_DATASET_HEADERS = [
    "Dataset",
    "Fields",
    "Pyramid Theme Multiplier",
    "Coverage",
    "Date Coverage",
    "Value Score",
    "Alphas",
    "Resources",
  ];
  const DEFAULT_FIELD_HEADERS = [
    "Field",
    "Description",
    "Type",
    "Coverage",
    "Date Coverage",
    "Alphas",
  ];
  const DEFAULT_OPERATOR_HEADERS = ["Operator", "Scope", "Description"];

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const nowIso = () => new Date().toISOString();

  const normalizeText = (value) =>
    String(value || "")
      .replace(/\s+/g, " ")
      .trim();

  const cleanCellText = (cell) => {
    if (!cell) {
      return "";
    }
    const clone = cell.cloneNode(true);
    for (const button of clone.querySelectorAll("button, [role='button'], a")) {
      const text = normalizeText(button.textContent);
      if (/^show\s+(more|less)$/i.test(text)) {
        button.remove();
      }
    }
    return normalizeText(clone.textContent);
  };

  const isVisibleElement = (element) => {
    if (!element || !(element instanceof Element)) {
      return false;
    }
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0) {
      return false;
    }
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const getDirectCells = (row) => {
    if (!row) {
      return [];
    }
    const scoped = row.querySelectorAll(":scope > td, :scope > th");
    if (scoped.length) {
      return [...scoped];
    }
    return [...row.children].filter((child) => /^(TD|TH)$/i.test(child.tagName));
  };

  const slugify = (value) =>
    String(value || "wq_export")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 140);

  const csvEscape = (value) => {
    const text = String(value ?? "");
    if (/[",\n]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  };

  const flattenObject = (value, prefix = "", out = {}) => {
    if (Array.isArray(value)) {
      out[prefix] = JSON.stringify(value);
      return out;
    }
    if (value && typeof value === "object") {
      for (const [key, child] of Object.entries(value)) {
        const nextPrefix = prefix ? `${prefix}.${key}` : key;
        flattenObject(child, nextPrefix, out);
      }
      return out;
    }
    out[prefix] = value ?? "";
    return out;
  };

  const toCsv = (rows, headers) => {
    const lines = [headers.join(",")];
    for (const row of rows) {
      lines.push(headers.map((header) => csvEscape(row[header])).join(","));
    }
    return lines.join("\n");
  };

  const download = (filename, content, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  let directoryHandlePromise = null;

  const canUseDirectoryPicker = () =>
    window.isSecureContext && typeof window.showDirectoryPicker === "function";

  const getDirectoryHandle = async () => {
    if (directoryHandlePromise) {
      return directoryHandlePromise;
    }
    directoryHandlePromise = window.showDirectoryPicker();
    return directoryHandlePromise;
  };

  const writeFileToDirectory = async (directoryHandle, filename, content, mimeType) => {
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(new Blob([content], { type: mimeType }));
    await writable.close();
  };

  const saveArtifacts = async (baseName, csvContent, jsonContent) => {
    const csvFilename = `${baseName}.csv`;
    const jsonFilename = `${baseName}.json`;

    if (SAVE_TO_DIRECTORY) {
      if (!canUseDirectoryPicker()) {
        console.warn(
          "[WQ export] Browser does not support choosing a destination folder here. Falling back to downloads.",
        );
      } else {
        try {
          const directoryHandle = await getDirectoryHandle();
          await writeFileToDirectory(
            directoryHandle,
            csvFilename,
            csvContent,
            "text/csv;charset=utf-8",
          );
          await writeFileToDirectory(
            directoryHandle,
            jsonFilename,
            jsonContent,
            "application/json;charset=utf-8",
          );
          console.log(
            `[WQ export] Saved files directly to the selected folder: ${csvFilename}, ${jsonFilename}`,
          );
          return "directory";
        } catch (error) {
          console.warn(
            "[WQ export] Directory save failed. Falling back to browser downloads.",
            error,
          );
        }
      }
    }

    download(csvFilename, csvContent, "text/csv;charset=utf-8");
    download(jsonFilename, jsonContent, "application/json;charset=utf-8");
    return "download";
  };

  const buildBaseName = () => {
    const current = new URL(window.location.href);
    const title =
      normalizeText(document.querySelector("h1, h2")?.textContent) ||
      normalizeText(document.title) ||
      "wq_export";
    const querySuffix = [
      current.searchParams.get("region"),
      current.searchParams.get("delay"),
      current.searchParams.get("universe"),
      current.searchParams.get("category"),
    ]
      .filter(Boolean)
      .join("_");
    return slugify(`${title}_${querySuffix}`);
  };

  const inferPageTypeFromUrl = () => {
    const current = new URL(window.location.href);
    if (/^\/learn\/operators\/?$/.test(current.pathname)) {
      return "operators";
    }
    return /^\/data\/data-sets\/([^/?#]+)/.test(current.pathname) ? "dataset_fields" : "datasets";
  };

  const chooseLargestObjectArray = (payload) => {
    if (Array.isArray(payload) && payload.every((item) => item && typeof item === "object")) {
      return { items: payload, path: "root" };
    }

    const candidates = [];
    const visit = (value, path, depth) => {
      if (depth > 6 || value == null) {
        return;
      }
      if (Array.isArray(value)) {
        if (value.length && value.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
          candidates.push({ items: value, path });
        }
        return;
      }
      if (typeof value === "object") {
        for (const [key, child] of Object.entries(value)) {
          visit(child, path ? `${path}.${key}` : key, depth + 1);
        }
      }
    };

    visit(payload, "", 0);
    if (!candidates.length) {
      return { items: [], path: "" };
    }
    candidates.sort((left, right) => right.items.length - left.items.length);
    return candidates[0];
  };

  const extractApiRecords = (payload) => {
    if (payload && Array.isArray(payload.results)) {
      return {
        items: payload.results,
        path: "results",
        totalCount: typeof payload.count === "number" ? payload.count : payload.results.length,
      };
    }
    if (payload && payload.data && Array.isArray(payload.data.results)) {
      return {
        items: payload.data.results,
        path: "data.results",
        totalCount:
          typeof payload.data.count === "number" ? payload.data.count : payload.data.results.length,
      };
    }

    const fallback = chooseLargestObjectArray(payload);
    return {
      items: fallback.items,
      path: fallback.path,
      totalCount: extractCount(payload, fallback.items.length),
    };
  };

  const extractCount = (payload, fallback) => {
    const countKeys = new Set(["count", "total", "totalCount", "rowCount", "numFound"]);
    let best = null;

    const visit = (value, depth) => {
      if (depth > 6 || value == null) {
        return;
      }
      if (typeof value === "number" && Number.isFinite(value) && value >= fallback) {
        best = best == null ? value : Math.max(best, value);
        return;
      }
      if (Array.isArray(value)) {
        for (const child of value.slice(0, 5)) {
          visit(child, depth + 1);
        }
        return;
      }
      if (typeof value === "object") {
        for (const [key, child] of Object.entries(value)) {
          if (countKeys.has(key) && typeof child === "number" && Number.isFinite(child) && child >= fallback) {
            best = best == null ? child : Math.max(best, child);
          } else {
            visit(child, depth + 1);
          }
        }
      }
    };

    visit(payload, 0);
    return best;
  };

  const resolveApiRequestSpec = () => {
    const current = new URL(window.location.href);
    const query = new URLSearchParams(current.search);
    const fieldPageMatch = current.pathname.match(/^\/data\/data-sets\/([^/?#]+)/);
    const operatorPageMatch = current.pathname.match(/^\/learn\/operators\/?$/);

    if (operatorPageMatch) {
      return {
        endpoint: "",
        query,
        pageType: "operators",
        datasetId: "",
      };
    }

    if (fieldPageMatch) {
      query.set("dataset.id", fieldPageMatch[1]);
      return {
        endpoint: "/data-fields",
        query,
        pageType: "dataset_fields",
        datasetId: fieldPageMatch[1],
      };
    }

    return {
      endpoint: current.pathname.replace(/^\/data\//, "/"),
      query,
      pageType: "datasets",
      datasetId: "",
    };
  };

  const buildApiUrl = (requestSpec, offset) => {
    const apiUrl = new URL(requestSpec.endpoint, API_ORIGIN);
    const query = new URLSearchParams(requestSpec.query);
    const limit = Number(query.get("limit") || DEFAULT_LIMIT);
    query.set("limit", String(limit));
    query.set("offset", String(offset));
    apiUrl.search = query.toString();
    return apiUrl;
  };

  const fetchJsonWithRetry = async (url) => {
    let lastError = null;
    for (let attempt = 1; attempt <= MAX_API_RETRIES; attempt += 1) {
      let response;
      try {
        response = await fetch(url, {
          method: "GET",
          credentials: "include",
          headers: {
            Accept: "application/json, text/plain, */*",
          },
        });
      } catch (error) {
        lastError = error;
        console.warn(`[WQ export] Network error on attempt ${attempt}/${MAX_API_RETRIES}:`, error);
        await sleep(RATE_LIMIT_BACKOFF_MS);
        continue;
      }

      if (response.status === 429) {
        const retryAfter = Number(response.headers.get("Retry-After") || "0");
        const waitMs = retryAfter > 0 ? retryAfter * 1000 : RATE_LIMIT_BACKOFF_MS;
        console.warn(
          `[WQ export] API rate-limited (429) on attempt ${attempt}/${MAX_API_RETRIES}. Waiting ${Math.round(waitMs / 1000)}s...`,
        );
        await sleep(waitMs);
        continue;
      }

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`[WQ export] API request failed: ${response.status} ${response.statusText}\n${body}`);
      }

      return response.json();
    }

    throw lastError || new Error("[WQ export] Exhausted API retries.");
  };

  const exportViaApi = async (baseName) => {
    const current = new URL(window.location.href);
    const requestSpec = resolveApiRequestSpec();
    if (requestSpec.pageType === "operators") {
      throw new Error("[WQ export] Operators page does not have a verified API export path. Use DOM mode.");
    }
    const limit = Number(requestSpec.query.get("limit") || current.searchParams.get("limit") || DEFAULT_LIMIT);
    let offset = Number(requestSpec.query.get("offset") || current.searchParams.get("offset") || "0");
    let totalCount = null;
    let pageIndex = 1;
    const allItems = [];
    const seen = new Set();
    const pageMeta = [];

    console.log(
      `[WQ export] API mode enabled | pageType=${requestSpec.pageType} | endpoint=${requestSpec.endpoint} | limit=${limit}`,
    );

    while (true) {
      const url = buildApiUrl(requestSpec, offset);
      const payload = await fetchJsonWithRetry(url);
      const { items, path, totalCount: responseTotalCount } = extractApiRecords(payload);
      if (!items.length) {
        throw new Error("[WQ export] API response did not contain a usable array of records.");
      }

      const newItems = [];
      for (const item of items) {
        const key = JSON.stringify(item);
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        allItems.push(item);
        newItems.push(item);
      }

      totalCount = totalCount ?? responseTotalCount ?? extractCount(payload, allItems.length);
      pageMeta.push({
        page: pageIndex,
        offset,
        item_count: items.length,
        new_item_count: newItems.length,
        item_path: path,
        api_url: url.toString(),
      });

      console.log(
        `[WQ export] API page ${pageIndex}: ${items.length} rows (${newItems.length} new, total ${allItems.length})`,
      );

      const exhaustedByPageSize = items.length < limit;
      const exhaustedByTotal = totalCount != null && allItems.length >= totalCount;
      if (exhaustedByPageSize || exhaustedByTotal) {
        break;
      }

      offset += limit;
      pageIndex += 1;
      await sleep(INTER_PAGE_DELAY_MS);
    }

    const flatRows = allItems.map((item) => flattenObject(item));
    const headers = [...new Set(flatRows.flatMap((row) => Object.keys(row)))];
    const csvContent = toCsv(flatRows, headers);
    const jsonContent = JSON.stringify(
      {
        exported_at: nowIso(),
        source: "api",
        page_url: window.location.href,
        total_count: totalCount,
        row_count: allItems.length,
        page_count: pageMeta.length,
        pages: pageMeta,
        rows: allItems,
      },
      null,
      2,
    );

    const saveMode = await saveArtifacts(baseName, csvContent, jsonContent);

    console.log(
      `[WQ export] Completed via API. Saved ${allItems.length} rows across ${pageMeta.length} page(s) using ${saveMode}.`,
    );
    return true;
  };

  const getDomHeaderRow = () => {
    const pageType = inferPageTypeFromUrl();
    const candidates = [
      ...document.querySelectorAll(".rt-thead .rt-tr"),
      ...document.querySelectorAll(".rt-table .rt-tr"),
      ...document.querySelectorAll('[role="row"]'),
      ...document.querySelectorAll("thead tr"),
      ...document.querySelectorAll("tr"),
    ];
    return (
      candidates.find((row) => {
        const text = normalizeText(row.textContent);
        if (pageType === "operators") {
          return /\boperator\b/i.test(text) && /\bscope\b/i.test(text) && /\bdescription\b/i.test(text);
        }
        return /\b(field|dataset)\b/i.test(text) && /\b(description|coverage|alphas)\b/i.test(text);
      }) || null
    );
  };

  const getDomHeaders = (headerRow) => {
    const directCells = getDirectCells(headerRow);
    const cells = directCells.length
      ? directCells
      : [
          ...headerRow.querySelectorAll(".rt-th"),
          ...headerRow.querySelectorAll('[role="columnheader"]'),
          ...headerRow.querySelectorAll("th"),
          ...headerRow.children,
        ];
    return cells.map((cell, index) => normalizeText(cell.textContent) || `column_${index + 1}`);
  };

  const getOperatorsMainTable = () => {
    const tables = [...document.querySelectorAll("table")];
    return (
      tables.find((table) => {
        const headerRow = table.querySelector("thead tr");
        const headerText = normalizeText(headerRow?.textContent || "");
        return /\boperator\b/i.test(headerText) && /\bscope\b/i.test(headerText) && /\bdescription\b/i.test(headerText);
      }) || null
    );
  };

  const getOperatorsContentRoot = () => {
    const heading =
      [...document.querySelectorAll("h1, h2, h3, [role='heading']")]
        .find((node) => /^operators$/i.test(normalizeText(node.textContent))) || null;
    return (
      heading?.closest("main, section, article, div") ||
      document.querySelector("main") ||
      document.body
    );
  };

  const getOperatorDomRows = () => {
    const mainTable = getOperatorsMainTable();
    const bodyRows = [...(mainTable?.querySelector("tbody")?.children || [])].filter((row) => {
      const cells = getDirectCells(row);
      return cells.length >= 3;
    });
    if (bodyRows.length) {
      return bodyRows;
    }

    const roleRows = [...document.querySelectorAll('[role="row"]')].filter((row) => {
      const cells = row.querySelectorAll(":scope > [role='gridcell'], :scope > [role='cell']");
      return cells.length >= 3 && !/\boperator\b\s*\bscope\b\s*\bdescription\b/i.test(normalizeText(row.textContent));
    });
    return roleRows;
  };

  const getDomBodyRows = (headers) => {
    const pageType = inferPageTypeFromUrl();
    if (pageType === "operators") {
      return getOperatorDomRows();
    }

    const reactTableGroups = [...document.querySelectorAll(".rt-tbody .rt-tr-group")];
    if (reactTableGroups.length) {
      return reactTableGroups
        .map((group) => group.querySelector(".rt-tr"))
        .filter((row) => row && row.querySelectorAll(".rt-td").length >= Math.max(3, Math.min(headers.length, 5)));
    }

    const rowCandidates = [
      ...document.querySelectorAll(".rt-tbody .rt-tr"),
      ...document.querySelectorAll('[role="row"]'),
      ...document.querySelectorAll("tbody tr"),
      ...document.querySelectorAll("tr"),
    ];
    return rowCandidates.filter((row) => {
      const cells = [
        ...row.querySelectorAll(".rt-td"),
        ...row.querySelectorAll('[role="gridcell"],[role="cell"]'),
        ...row.querySelectorAll("td"),
      ];
      return cells.length >= Math.max(3, Math.min(headers.length, 5));
    });
  };

  const domRowFingerprint = (rows) => {
    const first = normalizeText(rows[0]?.textContent || "");
    const last = normalizeText(rows[rows.length - 1]?.textContent || "");
    return `${rows.length}|${first}|${last}`;
  };

  const waitForDomChange = async (beforeFingerprint) => {
    const startedAt = Date.now();
    while (Date.now() - startedAt < NEXT_PAGE_TIMEOUT_MS) {
      const fallbackHeaderRow = {
        querySelectorAll: () => [],
        children: [],
      };
      const headerRow = getDomHeaderRow() || fallbackHeaderRow;
      const headers = getDomHeaders(headerRow);
      const rows = getDomBodyRows(headers);
      if (rows.length && domRowFingerprint(rows) !== beforeFingerprint) {
        return rows;
      }
      await sleep(250);
    }
    throw new Error("[WQ export] Timed out waiting for DOM page change.");
  };

  const findNextButton = () => {
    const explicit = document.querySelector(".pagination-footer__button--next");
    if (explicit) {
      return explicit;
    }
    return (
      [...document.querySelectorAll("button, a")]
        .filter((element) => /\bnext\b/i.test(normalizeText(element.textContent)))
        .sort((left, right) => right.getBoundingClientRect().width - left.getBoundingClientRect().width)[0] || null
    );
  };

  const isDisabled = (element) =>
    !element ||
    Boolean(element.disabled) ||
    element.getAttribute("aria-disabled") === "true" ||
    element.classList.contains("disabled") ||
    /disabled/.test(String(element.className || ""));

  const clickElement = (element) => {
    element.scrollIntoView({ block: "center", inline: "center" });
    if (typeof element.focus === "function") {
      element.focus();
    }
    if (typeof element.click === "function") {
      element.click();
    }
    element.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    element.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    element.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  };

  const getOperatorShowMoreButtons = () => {
    const roots = [getOperatorsMainTable(), getOperatorsContentRoot(), document.body].filter(Boolean);
    const seen = new Set();
    const buttons = [];

    for (const root of roots) {
      for (const node of root.querySelectorAll("button, [role='button'], a, span, div")) {
        const text = normalizeText(node.textContent);
        if (!/show more/i.test(text)) {
          continue;
        }
        const clickable = node.closest("button, [role='button'], a") || node;
        if (!isVisibleElement(clickable) || seen.has(clickable)) {
          continue;
        }
        seen.add(clickable);
        buttons.push(clickable);
      }
      if (buttons.length) {
        break;
      }
    }

    return buttons;
  };

  const expandOperatorDetails = async () => {
    let expanded = 0;
    let pass = 0;
    while (expanded < MAX_OPERATOR_EXPAND_CLICKS) {
      const buttons = getOperatorShowMoreButtons();
      if (!buttons.length) {
        if (!expanded) {
          console.log("[WQ export] No visible Show more buttons found on operators page.");
        }
        break;
      }
      pass += 1;
      console.log(`[WQ export] Operator expand pass ${pass}: found ${buttons.length} Show more button(s).`);
      for (const button of buttons) {
        if (expanded >= MAX_OPERATOR_EXPAND_CLICKS) {
          break;
        }
        clickElement(button);
        expanded += 1;
        await sleep(OPERATOR_EXPAND_SETTLE_MS);
      }
    }
    if (expanded) {
      console.log(`[WQ export] Expanded ${expanded} operator detail panel(s).`);
      await sleep(DOM_SETTLE_MS);
    }
  };

  const exportViaDom = async (baseName) => {
    const headerRow = getDomHeaderRow();
    const pageType = inferPageTypeFromUrl();
    const headers = headerRow
      ? getDomHeaders(headerRow)
      : pageType === "dataset_fields"
        ? DEFAULT_FIELD_HEADERS
        : pageType === "operators"
          ? DEFAULT_OPERATOR_HEADERS
        : DEFAULT_DATASET_HEADERS;
    const exportHeaders = [...headers, "__page"];
    const collectedRows = [];
    const seen = new Set();
    let currentPage = 1;

    if (headerRow) {
      console.log("[WQ export] DOM fallback mode enabled with detected headers.");
    } else {
      console.warn(
        `[WQ export] DOM header row not found. Falling back to default ${pageType} headers: ${headers.join(" | ")}`,
      );
    }

    while (true) {
      if (pageType === "operators") {
        await expandOperatorDetails();
      }

      const rows = getDomBodyRows(headers);
      if (!rows.length) {
        throw new Error("[WQ export] No body rows found in DOM fallback mode.");
      }

      const pageRows = rows.map((row) => {
        const directCells = getDirectCells(row);
        const cells = directCells.length
          ? directCells
          : [
              ...row.querySelectorAll(".rt-td"),
              ...row.querySelectorAll('[role="gridcell"],[role="cell"]'),
              ...row.querySelectorAll("td"),
            ];
        const payload = Object.fromEntries(
          headers.map((header, index) => [header, cleanCellText(cells[index])]),
        );
        payload.__page = currentPage;
        return payload;
      });

      let newRows = 0;
      for (const row of pageRows) {
        const key = JSON.stringify(headers.map((header) => row[header]));
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        collectedRows.push(row);
        newRows += 1;
      }

      console.log(
        `[WQ export] DOM page ${currentPage}: ${pageRows.length} rows (${newRows} new, total ${collectedRows.length})`,
      );

      const nextButton = findNextButton();
      if (!nextButton || isDisabled(nextButton)) {
        break;
      }

      const beforeFingerprint = domRowFingerprint(rows);
      let moved = false;
      for (let attempt = 1; attempt <= MAX_DOM_NEXT_RETRIES; attempt += 1) {
        clickElement(nextButton);
        try {
          await waitForDomChange(beforeFingerprint);
          await sleep(DOM_SETTLE_MS);
          moved = true;
          break;
        } catch (error) {
          console.warn(
            `[WQ export] DOM next-page attempt ${attempt}/${MAX_DOM_NEXT_RETRIES} failed. Waiting ${Math.round(
              RATE_LIMIT_BACKOFF_MS / 1000,
            )}s...`,
          );
          await sleep(RATE_LIMIT_BACKOFF_MS);
        }
      }

      if (!moved) {
        throw new Error("[WQ export] Failed to paginate in DOM fallback mode.");
      }
      currentPage += 1;
    }

    const csvContent = toCsv(collectedRows, exportHeaders);
    const jsonContent = JSON.stringify(
      {
        exported_at: nowIso(),
        source: "dom",
        page_url: window.location.href,
        headers,
        row_count: collectedRows.length,
        page_count: currentPage,
        rows: collectedRows,
      },
      null,
      2,
    );

    const saveMode = await saveArtifacts(baseName, csvContent, jsonContent);

    console.log(
      `[WQ export] Completed via DOM fallback. Saved ${collectedRows.length} rows across ${currentPage} page(s) using ${saveMode}.`,
    );
    return true;
  };

  const baseName = buildBaseName();
  console.log(
    `[WQ export] Started ${nowIso()} | baseName="${baseName}" | forceDom=${FORCE_DOM ? "true" : "false"} | saveToDirectory=${SAVE_TO_DIRECTORY ? "true" : "false"}`,
  );

  if (FORCE_DOM) {
    await exportViaDom(baseName);
    return;
  }

  try {
    await exportViaApi(baseName);
  } catch (apiError) {
    console.warn("[WQ export] API mode failed. Falling back to DOM mode.", apiError);
    await exportViaDom(baseName);
  }
})();
