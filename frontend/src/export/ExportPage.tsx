import { useEffect, useState } from "react";
import type { ApiClient } from "../api/client";
import type { BundleImportResult, Document, ExportOptions } from "../api/types";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const PREVIEW_CHARS = 6000;

/** The file name the browser saves an export under (the server suggests the same one). */
export function exportFileName(o: ExportOptions, docName: string | null): string {
  const scope = docName ? docName.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-._]+|[-._]+$/g, "").slice(0, 60) || "project" : "all";
  return o.format === "pagexml" ? `htr-${scope}-pagexml.zip` : `htr-${scope}-${o.expanded ? "expanded" : "raw"}.txt`;
}

function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export interface ExportPageProps {
  api: ApiClient;
  /** Start the (streamed) bundle download; default: let the browser navigate to the download URL. */
  startDownload?: (url: string) => void;
  save?: (blob: Blob, name: string) => void;
}

export function ExportPage({ api, startDownload = (u) => window.location.assign(u), save = saveBlob }: ExportPageProps) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [docId, setDocId] = useState<string>("");
  const [format, setFormat] = useState<"txt" | "pagexml">("txt");
  const [expanded, setExpanded] = useState(true);
  const [mark, setMark] = useState(false);
  const [predicted, setPredicted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  const [bundleError, setBundleError] = useState<string | null>(null);
  const [bundleNote, setBundleNote] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState<BundleImportResult | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  useEffect(() => {
    api.listDocuments().then(setDocs).catch(() => undefined); // the scope list is optional: "All documents" still works
  }, [api]);

  const options: ExportOptions = {
    format,
    expanded: format === "txt" && expanded,
    ambiguous: mark ? "mark" : "raw",
    document_id: docId ? Number(docId) : undefined,
    include_predicted: predicted,
  };
  const docName = docs.find((d) => String(d.id) === docId)?.name ?? null;

  const download = async () => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const blob = await api.exportBlob(options);
      const name = exportFileName(options, docName);
      save(blob, name);
      setSaved(name);
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const showPreview = async () => {
    setBusy(true);
    setError(null);
    try {
      const text = await (await api.exportBlob({ ...options, format: "txt" })).text();
      setPreview(text.length > PREVIEW_CHARS ? text.slice(0, PREVIEW_CHARS) + "\n... (preview cut)" : text || "(nothing to export yet: no transcribed lines)");
    } catch (e) {
      setError(errText(e));
      setPreview(null);
    } finally {
      setBusy(false);
    }
  };

  const exportBundle = async () => {
    setBundleError(null);
    setBundleNote(null);
    try {
      const [q, r] = await Promise.all([api.listJobs({ status: "queued" }), api.listJobs({ status: "running" })]);
      const j = [...r, ...q][0];
      if (j) {
        setBundleError(
          `A ${j.type} job is ${j.status} (job ${j.id}). Wait for it to finish or cancel it, then export the bundle.`,
        );
        return;
      }
      startDownload(api.bundleExportUrl());
      setBundleNote("Your browser is saving the bundle. It asks where to put it, or uses its download folder; nothing is uploaded.");
    } catch (e) {
      setBundleError(errText(e));
    }
  };

  const importBundle = async () => {
    if (!file) return;
    setImporting(true);
    setImportError(null);
    setImported(null);
    try {
      setImported(await api.importBundle(file));
    } catch (e) {
      setImportError(errText(e));
    } finally {
      setImporting(false);
    }
  };

  return (
    <main className="training export-page">
      <header className="training-head">
        <h1>Export</h1>
        <span className="actions">
          <a className="button" href="#/">
            Library
          </a>
          <a className="button" href="#/abbreviations">
            Abbreviations
          </a>
        </span>
      </header>

      <section className="panel" aria-labelledby="text-heading">
        <h2 id="text-heading">Text and PAGE-XML</h2>
        <div className="export-form">
          <label>
            What to export
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              <option value="">All documents</option>
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
          <fieldset>
            <legend>Format</legend>
            <label className="check">
              <input type="radio" name="format" checked={format === "txt"} onChange={() => setFormat("txt")} /> Plain text (.txt, UTF-8)
            </label>
            <label className="check">
              <input type="radio" name="format" checked={format === "pagexml"} onChange={() => setFormat("pagexml")} /> PAGE-XML (zip, one file per page)
            </label>
          </fieldset>
          <label className="check">
            <input type="checkbox" checked={expanded} disabled={format !== "txt"} onChange={(e) => setExpanded(e.target.checked)} /> Expand
            abbreviations
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={mark}
              disabled={format !== "txt" || !expanded}
              onChange={(e) => setMark(e.target.checked)}
            />{" "}
            Mark ambiguous abbreviations as <code>[?]</code> (otherwise they stay as written)
          </label>
          <label className="check">
            <input type="checkbox" checked={predicted} onChange={(e) => setPredicted(e.target.checked)} /> Include predicted lines (not yet
            checked{format === "txt" ? "; each is prefixed [predicted]" : ""})
          </label>
        </div>
        <p className="muted small">
          Ignored and empty lines are never exported. Lines come in reading order, pages in import order.
        </p>
        <div className="actions">
          <button onClick={() => void download()} disabled={busy}>
            Download {format === "txt" ? "text" : "PAGE-XML zip"}
          </button>
          <button onClick={() => void showPreview()} disabled={busy}>
            Preview text
          </button>
        </div>
        {error && (
          <p role="alert" className="error banner">
            <span>{error}</span>
            <button onClick={() => setError(null)}>Dismiss</button>
          </p>
        )}
        {saved && (
          <p role="status" className="notice" data-testid="export-saved">
            Saved {saved}
          </p>
        )}
        {preview !== null && (
          <pre className="export-preview" aria-label="Export preview" tabIndex={0}>
            {preview}
          </pre>
        )}
      </section>

      <section className="panel" aria-labelledby="bundle-heading">
        <h2 id="bundle-heading">Project bundle</h2>
        <p role="note" className="warn-note">
          <strong>A bundle contains your scans and all transcriptions, so it contains personal data.</strong> It is written only where you
          choose, and nothing is uploaded anywhere. Keep it as carefully as the originals.
        </p>
        <p className="muted small">
          A bundle (<code>.htrbundle</code>) holds everything: scans, working copies, PAGE-XML, lines and flags, abbreviations, models and job
          history. Importing it into an empty project restores the work exactly where it stopped. It is refused if the project already has
          any documents, pages, lines, abbreviations or trained models (bundles are never merged).
        </p>
        <div className="actions">
          <button onClick={() => void exportBundle()}>Export project bundle</button>
        </div>
        {bundleNote && (
          <p role="status" className="notice" data-testid="bundle-note">
            {bundleNote}
          </p>
        )}
        {bundleError && (
          <p role="alert" className="error banner" data-testid="bundle-error">
            <span>{bundleError}</span>
            <button onClick={() => setBundleError(null)}>Dismiss</button>
          </p>
        )}

        <h3>Import a project bundle</h3>
        <div className="export-form">
          <label>
            Bundle file
            <input
              type="file"
              accept=".htrbundle,application/zip"
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                setImported(null);
                setImportError(null);
              }}
            />
          </label>
          <button onClick={() => void importBundle()} disabled={!file || importing}>
            Import project bundle
          </button>
        </div>
        {importing && (
          <p role="status" className="notice">
            Importing. Large bundles take a while: the bundle is verified completely before anything is changed. Keep this page open.
          </p>
        )}
        {importError && (
          <p role="alert" className="error banner" data-testid="import-error">
            <span>Import refused: {importError}</span>
            <button onClick={() => setImportError(null)}>Dismiss</button>
          </p>
        )}
        {imported && (
          <p role="status" className="notice" data-testid="import-ok">
            Imported {imported.counts.documents ?? 0} documents, {imported.counts.pages ?? 0} pages, {imported.counts.lines ?? 0} lines,{" "}
            {imported.counts.finetuned_models ?? 0} trained models and {imported.counts.abbreviations ?? 0} abbreviations
            {imported.migrated_from ? ` (database upgraded from schema ${imported.migrated_from})` : ""}. <a href="#/">Open the library</a>
          </p>
        )}
      </section>
    </main>
  );
}
