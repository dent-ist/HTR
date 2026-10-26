import type { Document, Job, JobStatus, Line, LinePatch, Page, PageDetail, PageStatus, UploadResult } from "./types";

export const DEFAULT_API_BASE = "http://127.0.0.1:8000";

/** Base URL from VITE_API_BASE_URL (build/dev time), else the loopback default. */
export function apiBaseFromEnv(): string {
  const v = (import.meta.env?.VITE_API_BASE_URL as string | undefined) ?? "";
  return (v || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI errors are `{detail: string}` or `{detail: [{loc, msg, type}]}`. */
export function describeDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const e = d as { loc?: unknown[]; msg?: string };
        const where = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        return where ? `${where}: ${e.msg}` : String(e.msg);
      })
      .join("; ");
  }
  return "unknown error";
}

export type FetchLike = typeof fetch;

export class ApiClient {
  readonly base: string;
  private readonly fetchImpl: FetchLike;

  constructor(base: string = apiBaseFromEnv(), fetchImpl?: FetchLike) {
    this.base = base.replace(/\/+$/, "");
    this.fetchImpl = fetchImpl ?? ((...a) => fetch(...a));
  }

  /** Absolute URL for a server-relative path such as `line.crop_url` or `page.image_url`. */
  url(path: string): string {
    return /^https?:\/\//.test(path) ? path : this.base + path;
  }

  private async request<T>(method: string, path: string, init: { json?: unknown; form?: FormData } = {}): Promise<T> {
    const headers: Record<string, string> = {};
    let body: BodyInit | undefined;
    if (init.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(init.json);
    } else if (init.form) {
      body = init.form;
    }
    let res: Response;
    try {
      res = await this.fetchImpl(this.url(path), { method, headers, body });
    } catch (e) {
      throw new ApiError(0, `Cannot reach the API at ${this.base} (${e instanceof Error ? e.message : String(e)})`);
    }
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        msg = describeDetail(((await res.json()) as { detail?: unknown }).detail);
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(res.status, msg);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  listDocuments(): Promise<Document[]> {
    return this.request("GET", "/documents");
  }
  createDocument(name: string): Promise<Document> {
    return this.request("POST", "/documents", { json: { name } });
  }
  listPages(filter: { document_id?: number; status?: PageStatus } = {}): Promise<Page[]> {
    const q = new URLSearchParams();
    if (filter.document_id !== undefined) q.set("document_id", String(filter.document_id));
    if (filter.status) q.set("status", filter.status);
    const qs = q.toString();
    return this.request("GET", "/pages" + (qs ? `?${qs}` : ""));
  }
  getPage(id: number): Promise<PageDetail> {
    return this.request("GET", `/pages/${id}`);
  }
  uploadPage(documentId: number, file: File): Promise<UploadResult> {
    const form = new FormData();
    form.append("document_id", String(documentId));
    form.append("file", file, file.name);
    return this.request("POST", "/pages", { form });
  }
  segmentPage(id: number, force = false): Promise<Job> {
    return this.request("POST", `/pages/${id}/segment`, { json: { force } });
  }
  patchLine(id: number, patch: LinePatch): Promise<Line> {
    return this.request("PATCH", `/lines/${id}`, { json: patch });
  }
  getJob(id: number): Promise<Job> {
    return this.request("GET", `/jobs/${id}`);
  }
  listJobs(filter: { status?: JobStatus; type?: string; limit?: number } = {}): Promise<Job[]> {
    const q = new URLSearchParams();
    if (filter.status) q.set("status", filter.status);
    if (filter.type) q.set("type", filter.type);
    if (filter.limit) q.set("limit", String(filter.limit));
    const qs = q.toString();
    return this.request("GET", "/jobs" + (qs ? `?${qs}` : ""));
  }
}
