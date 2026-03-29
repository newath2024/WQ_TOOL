/*
 * Export the fully expanded WorldQuant BRAIN Operators page.
 *
 * Usage:
 * 1. Open https://platform.worldquantbrain.com/learn/operators
 * 2. Expand all visible "Show more" sections first.
 * 3. Open DevTools -> Console.
 * 4. Paste this file and press Enter.
 *
 * Output:
 * - worldquant_brain_operators_expanded.json
 *
 * The export is intentionally operator-focused and preserves:
 * - signature
 * - tier badge
 * - scope
 * - summary text
 * - full expanded detail text
 * - raw detail HTML
 * - image metadata found inside the expanded detail block
 */

(async () => {
  const exportOptions =
    globalThis.WQ_EXPORT_OPTIONS && typeof globalThis.WQ_EXPORT_OPTIONS === "object"
      ? globalThis.WQ_EXPORT_OPTIONS
      : {};
  const SAVE_TO_DIRECTORY = Boolean(
    exportOptions.saveToDirectory ||
      exportOptions.pickDirectory ||
      globalThis.WQ_EXPORT_SAVE_TO_DIRECTORY,
  );

  const normalizeText = (value) =>
    String(value || "")
      .replace(/\s+/g, " ")
      .trim();

  const slugify = (value) =>
    String(value || "worldquant_brain_operators_expanded")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 140);

  const canUseDirectoryPicker = () =>
    window.isSecureContext && typeof window.showDirectoryPicker === "function";

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

  let directoryHandlePromise = null;

  const getDirectoryHandle = async () => {
    if (directoryHandlePromise) {
      return directoryHandlePromise;
    }
    directoryHandlePromise = window.showDirectoryPicker();
    return directoryHandlePromise;
  };

  const saveTextFile = async (filename, content) => {
    if (SAVE_TO_DIRECTORY && canUseDirectoryPicker()) {
      const directoryHandle = await getDirectoryHandle();
      const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(new Blob([content], { type: "application/json;charset=utf-8" }));
      await writable.close();
      console.log(`[WQ operator export] Saved ${filename} to selected folder.`);
      return;
    }

    const blob = new Blob([content], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    console.log(`[WQ operator export] Downloaded ${filename}.`);
  };

  const getDirectCells = (row) => [...row.children].filter((child) => /^(TD|TH)$/i.test(child.tagName));

  const getRoleCells = (row) => [
    ...row.querySelectorAll(":scope > [role='gridcell'], :scope > [role='cell'], :scope > .rt-td"),
  ];

  const getRowCells = (row) => {
    const directCells = getDirectCells(row);
    if (directCells.length) {
      return directCells;
    }
    const roleCells = getRoleCells(row);
    if (roleCells.length) {
      return roleCells;
    }
    return [...row.children].filter((child) => child instanceof Element);
  };

  const getOperatorsTable = () => {
    const tables = [...document.querySelectorAll("table")];
    return (
      tables.find((table) => {
        const headerText = normalizeText(table.querySelector("thead")?.textContent || "");
        return /\boperator\b/i.test(headerText) && /\bscope\b/i.test(headerText) && /\bdescription\b/i.test(headerText);
      }) || null
    );
  };

  const getOperatorsContentRoot = () => {
    const heading =
      [...document.querySelectorAll("h1, h2, h3, [role='heading']")]
        .find((node) => /^operators$/i.test(normalizeText(node.textContent))) || null;
    return heading?.closest("main, section, article, div") || document.querySelector("main") || document.body;
  };

  const looksLikeOperatorSignature = (value) => /^[a-z][a-z0-9_]*\s*\(/i.test(normalizeText(value));

  const isOperatorRow = (row) => {
    if (!row || !isVisibleElement(row)) {
      return false;
    }
    const cells = getRowCells(row);
    if (cells.length < 3) {
      return false;
    }
    const rowText = normalizeText(row.textContent);
    if (/\boperator\b/i.test(rowText) && /\bscope\b/i.test(rowText) && /\bdescription\b/i.test(rowText)) {
      return false;
    }
    return looksLikeOperatorSignature(cells[0]?.textContent || "");
  };

  const getOperatorRows = () => {
    const table = getOperatorsTable();
    if (table) {
      const tableRows = [...(table.querySelector("tbody")?.children || [])].filter((row) => isOperatorRow(row));
      if (tableRows.length) {
        console.log(`[WQ operator export] Using HTML table mode with ${tableRows.length} operator rows.`);
        return tableRows;
      }
    }

    const root = getOperatorsContentRoot();
    const candidates = [
      ...root.querySelectorAll('[role="row"]'),
      ...root.querySelectorAll(".rt-tr"),
      ...root.querySelectorAll("tr"),
    ];
    const seen = new Set();
    const rows = [];
    for (const row of candidates) {
      if (seen.has(row) || !isOperatorRow(row)) {
        continue;
      }
      seen.add(row);
      rows.push(row);
    }
    console.log(`[WQ operator export] Using row-role fallback mode with ${rows.length} operator rows.`);
    return rows;
  };

  const stripToggleButtons = (root) => {
    const clone = root.cloneNode(true);
    for (const node of clone.querySelectorAll("button, [role='button'], a")) {
      if (/^show\s+(more|less)$/i.test(normalizeText(node.textContent))) {
        node.remove();
      }
    }
    return clone;
  };

  const getTierFromCell = (cell) => {
    const tierNode =
      [...cell.querySelectorAll("button, span, div, p")]
        .find((node) => /^(base|genius|expert|master|grandmaster)$/i.test(normalizeText(node.textContent))) || null;
    return tierNode ? normalizeText(tierNode.textContent).toLowerCase() : null;
  };

  const getSignatureFromCell = (cell) => {
    const clone = cell.cloneNode(true);
    for (const node of clone.querySelectorAll("button, span, div, p")) {
      if (/^(base|genius|expert|master|grandmaster)$/i.test(normalizeText(node.textContent))) {
        node.remove();
      }
    }
    return normalizeText(clone.textContent);
  };

  const getSummaryFromDescriptionCell = (cell) => {
    const clone = stripToggleButtons(cell);
    const lines = [...clone.childNodes]
      .map((node) => normalizeText(node.textContent || ""))
      .filter(Boolean);
    return lines[0] || normalizeText(clone.textContent);
  };

  const collectImages = (root) =>
    [...root.querySelectorAll("img")].map((image) => ({
      src: image.getAttribute("src") || "",
      alt: normalizeText(image.getAttribute("alt") || ""),
      width: image.naturalWidth || image.width || null,
      height: image.naturalHeight || image.height || null,
    }));

  const getDetailPayloadFromRoot = (root, summary = "") => {
    const clone = stripToggleButtons(root);
    const fullText = normalizeText(clone.textContent || "");
    const detailText =
      summary && fullText.startsWith(summary)
        ? normalizeText(fullText.slice(summary.length))
        : fullText;
    return {
      text: detailText,
      html: (clone.innerHTML || "").trim(),
      images: collectImages(root),
    };
  };

  const appendDetailPayload = (entry, payload) => {
    if (!payload) {
      return;
    }
    if (payload.text) {
      entry.detail_text = entry.detail_text ? `${entry.detail_text}\n\n${payload.text}` : payload.text;
    }
    if (payload.html) {
      entry.detail_html = entry.detail_html ? `${entry.detail_html}\n\n${payload.html}` : payload.html;
    }
    if (payload.images?.length) {
      entry.images.push(...payload.images);
    }
  };

  const getFollowingDetailSiblings = (row, operatorRowSet) => {
    const details = [];
    let sibling = row.nextElementSibling;
    while (sibling && !operatorRowSet.has(sibling)) {
      if (isVisibleElement(sibling)) {
        details.push(sibling);
      }
      sibling = sibling.nextElementSibling;
    }
    return details;
  };

  const collectEntries = () => {
    const rows = getOperatorRows();
    if (!rows.length) {
      throw new Error("[WQ operator export] Could not locate operator rows on the page.");
    }
    const operatorRowSet = new Set(rows);
    const entries = [];

    for (const row of rows) {
      const cells = getRowCells(row);
      const signature = getSignatureFromCell(cells[0]);
      if (!signature) {
        continue;
      }

      const summary = getSummaryFromDescriptionCell(cells[2]);
      const entry = {
        signature,
        name: signature.split("(")[0].trim(),
        tier: getTierFromCell(cells[0]),
        scope: normalizeText(cells[1].textContent),
        summary,
        detail_text: "",
        detail_html: "",
        images: [],
      };

      appendDetailPayload(entry, getDetailPayloadFromRoot(cells[2], summary));

      for (const sibling of getFollowingDetailSiblings(row, operatorRowSet)) {
        appendDetailPayload(entry, getDetailPayloadFromRoot(sibling, ""));
      }

      entry.images = entry.images.filter(
        (image, index, allImages) =>
          allImages.findIndex(
            (candidate) =>
              candidate.src === image.src &&
              candidate.alt === image.alt &&
              candidate.width === image.width &&
              candidate.height === image.height,
          ) === index,
      );
      entries.push(entry);
    }

    return entries;
  };

  const nowIso = new Date().toISOString();
  const filename = `${slugify("worldquant_brain_operators_expanded")}.json`;
  const operators = collectEntries();
  const payload = {
    exported_at: nowIso,
    page_url: window.location.href,
    title: normalizeText(document.title),
    operator_count: operators.length,
    operators,
  };

  await saveTextFile(filename, JSON.stringify(payload, null, 2));
  console.log(`[WQ operator export] Exported ${operators.length} expanded operators.`);
})();
