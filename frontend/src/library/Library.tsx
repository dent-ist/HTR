import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, type ApiClient } from "../api/client";
import type { Document, Job, Page } from "../api/types";
import { pageHash, TRAINING_HASH } from "../router";
import { jobIsActive, jobLabel, progressOf, statusLabel } from "./format";

const POLL_MS = 1000;
const ACCEPT = "image/png,image/jpeg,image/tiff,.png,.jpg,.jpeg,.tif,.tiff";

type UploadState = { key: string; name: string; kind: "uploading" | "ok" | "duplicate" | "error"; message: string };

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function Library({ api }: { api: ApiClient }) {
  const [docs, setDocs] = useState<Document[] | null>(null);
  const [pages, setPages] = useState<Page[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<number, Job>>({}); // by page id
  const [uploads, setUploads] = useState<Record<number, UploadState[]>>({}); // by document id
  const [segErrors, setSegErrors] = useState<Record<number, string>>({});
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [d, p] = await Promise.all([api.listDocuments(), api.listPages()]);
      setDocs(d);
      setPages(p);
      setError(null);
    } catch (e) {
      setError(errText(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
    // Resume tracking segmentation jobs that are already running (e.g. after a reload).
    Promise.all([api.listJobs({ status: "queued", type: "segment" }), api.listJobs({ status: "running", type: "segment" })])
      .then(([q, r]) => {
        const next: Record<number, Job> = {};
        for (const j of [...q, ...r]) {
          const pid = j.payload.page_id;
          if (typeof pid === "number") next[pid] = j;
        }
        setJobs((cur) => ({ ...next, ...cur }));
      })
      .catch(() => undefined);
  }, [api, refresh]);

  // Poll active jobs; refresh the lists when one finishes.
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const anyActive = Object.values(jobs).some(jobIsActive);
  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(async () => {
      let finished = false;
      for (const [pid, job] of Object.entries(jobsRef.current)) {
        if (!jobIsActive(job)) continue;
        try {
          const fresh = await api.getJob(job.id);
          setJobs((cur) => ({ ...cur, [Number(pid)]: fresh }));
          if (!jobIsActive(fresh)) finished = true;
        } catch (e) {
          setSegErrors((cur) => ({ ...cur, [Number(pid)]: errText(e) }));
        }
      }
      if (finished) void refresh();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [anyActive, api, refresh]);

  const segment = async (page: Page) => {
    setSegErrors((c) => {
      const { [page.id]: _x, ...rest } = c;
      void _x;
      return rest;
    });
    const start = async (force: boolean) => {
      const job = await api.segmentPage(page.id, force);
      setJobs((cur) => ({ ...cur, [page.id]: job }));
    };
    try {
      await start(false);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && !/segmentation job/i.test(e.message)) {
        if (window.confirm(`${e.message}\n\nRe-segment anyway? This replaces ALL lines on the page, including your transcriptions.`)) {
          try {
            await start(true);
          } catch (e2) {
            setSegErrors((c) => ({ ...c, [page.id]: errText(e2) }));
          }
        }
      } else {
        setSegErrors((c) => ({ ...c, [page.id]: errText(e) }));
      }
    }
  };

  const upload = async (doc: Document, files: File[]) => {
    for (const file of files) {
      const key = `${file.name}-${file.size}-${Math.random()}`;
      const set = (s: Omit<UploadState, "key" | "name">) =>
        setUploads((cur) => ({
          ...cur,
          [doc.id]: [...(cur[doc.id] ?? []).filter((u) => u.key !== key), { key, name: file.name, ...s }],
        }));
      set({ kind: "uploading", message: "Uploading..." });
      try {
        const res = await api.uploadPage(doc.id, file);
        if (res.duplicate) {
          set({ kind: "duplicate", message: `Already imported (identical file): page #${res.id} "${res.original_filename}". Nothing was added.` });
        } else {
          set({ kind: "ok", message: `Imported as page #${res.id}.${res.warnings.length ? ` Warnings: ${res.warnings.join("; ")}` : ""}` });
        }
      } catch (e) {
        set({ kind: "error", message: errText(e) });
      }
      await refresh();
    }
  };

  const createDoc = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await api.createDocument(name);
      setNewName("");
      await refresh();
    } catch (err) {
      setError(errText(err));
    } finally {
      setCreating(false);
    }
  };

  return (
    <main className="library">
      <header className="library-head">
        <h1>HTR library</h1>
        <a className="button" href={TRAINING_HASH}>
          Training
        </a>
        <form onSubmit={createDoc} className="new-doc">
          <label>
            New document name
            <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Notebook 1" maxLength={200} />
          </label>
          <button type="submit" disabled={creating || !newName.trim()}>
            Create document
          </button>
        </form>
      </header>

      {error && (
        <p role="alert" className="error banner">
          <span>{error}</span>
          <button onClick={() => void refresh()}>Retry</button>
        </p>
      )}
      {docs === null && !error && <p className="muted">Loading...</p>}
      {docs?.length === 0 && <p className="muted">No documents yet. Create one above, then add page images to it.</p>}

      {docs?.map((doc) => (
        <DocumentSection
          key={doc.id}
          doc={doc}
          pages={pages.filter((p) => p.document_id === doc.id).sort((a, b) => a.id - b.id)}
          jobs={jobs}
          segErrors={segErrors}
          uploads={uploads[doc.id] ?? []}
          onUpload={(files) => upload(doc, files)}
          onSegment={segment}
          onClearUploads={() => setUploads((c) => ({ ...c, [doc.id]: [] }))}
        />
      ))}
    </main>
  );
}

interface SectionProps {
  doc: Document;
  pages: Page[];
  jobs: Record<number, Job>;
  segErrors: Record<number, string>;
  uploads: UploadState[];
  onUpload(files: File[]): void;
  onSegment(page: Page): void;
  onClearUploads(): void;
}

function DocumentSection({ doc, pages, jobs, segErrors, uploads, onUpload, onSegment, onClearUploads }: SectionProps) {
  const [over, setOver] = useState(false);
  const pickRef = useRef<HTMLInputElement>(null);
  const take = (list: FileList | null) => {
    if (list && list.length) onUpload(Array.from(list));
  };
  return (
    <section className="doc" aria-labelledby={`doc-${doc.id}`}>
      <h2 id={`doc-${doc.id}`}>
        {doc.name} <span className="muted">({pages.length} {pages.length === 1 ? "page" : "pages"})</span>
      </h2>

      <div
        className={`dropzone ${over ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
      >
        <span>Drop PNG, JPEG or TIFF images here, or</span>
        <button type="button" onClick={() => pickRef.current?.click()}>
          Choose images
        </button>
        <input
          ref={pickRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          aria-label={`Add images to ${doc.name}`}
          data-testid={`file-input-${doc.id}`}
          onChange={(e) => {
            take(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {uploads.length > 0 && (
        <ul className="uploads" aria-label="Upload results">
          {uploads.map((u) => (
            <li key={u.key} className={u.kind}>
              <strong>{u.name}</strong> {u.message}
            </li>
          ))}
          <li>
            <button onClick={onClearUploads}>Clear</button>
          </li>
        </ul>
      )}

      {pages.length > 0 && (
        <table className="pages">
          <thead>
            <tr>
              <th scope="col">Page</th>
              <th scope="col">Status</th>
              <th scope="col">Progress</th>
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {pages.map((p) => (
              <PageRow key={p.id} page={p} job={jobs[p.id]} segError={segErrors[p.id]} onSegment={() => onSegment(p)} />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function PageRow({ page, job, segError, onSegment }: { page: Page; job: Job | undefined; segError: string | undefined; onSegment(): void }) {
  const active = job ? jobIsActive(job) : false;
  const { done, total } = progressOf(page);
  return (
    <tr data-testid={`page-row-${page.id}`}>
      <td>
        <a href={pageHash(page.id)}>{page.original_filename}</a>
        {page.warnings.length > 0 && (
          <div className="muted small" title={page.warnings.join("\n")}>
            {page.warnings.length} import warning{page.warnings.length > 1 ? "s" : ""}
          </div>
        )}
      </td>
      <td>
        <span className={`status ${page.status}`}>{statusLabel(page.status)}</span>
      </td>
      <td>
        {page.line_count === 0 ? (
          <span className="muted">no lines</span>
        ) : (
          <div className="progress-cell">
            <progress value={done} max={Math.max(total, 1)} aria-label={`${done} of ${total} active lines transcribed`} />
            <span>
              {done}/{total} lines
              {page.ignored_count > 0 ? `, ${page.ignored_count} ignored` : ""}
            </span>
          </div>
        )}
      </td>
      <td>
        <div className="actions">
          <button onClick={onSegment} disabled={active} aria-label={`${page.status === "unsegmented" ? "Segment" : "Re-segment"} ${page.original_filename}`}>
            {page.status === "unsegmented" ? "Segment" : "Re-segment"}
          </button>
          {page.line_count > 0 && (
            <a className="button" href={pageHash(page.id)}>
              Transcribe
            </a>
          )}
        </div>
        {job && (active || job.status === "failed") && (
          <div className={`job ${job.status}`} role="status">
            {jobLabel(job)}
            {active && <progress value={job.progress} max={1} aria-label="Segmentation progress" />}
            {job.status === "failed" && job.error && <div className="error">Segmentation failed: {job.error}</div>}
          </div>
        )}
        {segError && (
          <div className="error" role="alert">
            {segError}
          </div>
        )}
      </td>
    </tr>
  );
}
