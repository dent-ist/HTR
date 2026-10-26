import { describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError, describeDetail } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function client(impl: (url: string, init: RequestInit) => Response | Promise<Response>) {
  const spy = vi.fn((url: RequestInfo | URL, init?: RequestInit) => Promise.resolve(impl(String(url), init ?? {})));
  return { api: new ApiClient("http://api.test/", spy as unknown as typeof fetch), spy };
}

describe("ApiClient", () => {
  it("builds urls and strips trailing slashes from the base", () => {
    const { api } = client(() => jsonResponse([]));
    expect(api.base).toBe("http://api.test");
    expect(api.url("/lines/3/crop.png")).toBe("http://api.test/lines/3/crop.png");
    expect(api.url("http://other/x")).toBe("http://other/x");
  });

  it("sends PATCH with a JSON body", async () => {
    const { api, spy } = client(() => jsonResponse({ id: 7, text: "hi" }));
    const line = await api.patchLine(7, { text: "hi", is_ground_truth: true });
    expect(line.id).toBe(7);
    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toBe("http://api.test/lines/7");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ text: "hi", is_ground_truth: true });
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("builds list queries", async () => {
    const { api, spy } = client(() => jsonResponse([]));
    await api.listPages({ document_id: 2, status: "done" });
    await api.listPages();
    await api.listJobs({ status: "running", type: "segment" });
    expect(spy.mock.calls.map((c) => String(c[0]))).toEqual([
      "http://api.test/pages?document_id=2&status=done",
      "http://api.test/pages",
      "http://api.test/jobs?status=running&type=segment",
    ]);
  });

  it("uploads multipart without a manual content type", async () => {
    const { api, spy } = client(() => jsonResponse({ id: 1, duplicate: true }, 200));
    const res = await api.uploadPage(4, new File(["x"], "a.png", { type: "image/png" }));
    expect(res.duplicate).toBe(true);
    const init = spy.mock.calls[0]![1]!;
    expect(init.body).toBeInstanceOf(FormData);
    const form = init.body as FormData;
    expect(form.get("document_id")).toBe("4");
    expect((form.get("file") as File).name).toBe("a.png");
    expect((init.headers as Record<string, string>)["Content-Type"]).toBeUndefined();
  });

  it("posts segment with force flag", async () => {
    const { api, spy } = client(() => jsonResponse({ id: 9 }, 202));
    await api.segmentPage(3, true);
    expect(JSON.parse(String(spy.mock.calls[0]![1]!.body))).toEqual({ force: true });
  });

  it("turns API errors into ApiError with the detail text", async () => {
    const { api } = client(() => jsonResponse({ detail: "page 1 already has a segmentation job" }, 409));
    await expect(api.segmentPage(1)).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      message: "page 1 already has a segmentation job",
    });
  });

  it("formats 422 validation lists", async () => {
    const { api } = client(() => jsonResponse({ detail: [{ loc: ["body", "text"], msg: "too long", type: "x" }] }, 422));
    const err = await api.patchLine(1, { text: "x" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).message).toBe("text: too long");
    expect(describeDetail(undefined)).toBe("unknown error");
  });

  it("reports an unreachable API as status 0", async () => {
    const api = new ApiClient("http://api.test", (() => Promise.reject(new TypeError("Failed to fetch"))) as typeof fetch);
    await expect(api.listDocuments()).rejects.toMatchObject({ status: 0 });
  });

  it("handles non-JSON error bodies", async () => {
    const { api } = client(() => new Response("boom", { status: 500 }));
    await expect(api.getJob(1)).rejects.toMatchObject({ status: 500, message: "HTTP 500" });
  });
});
