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

  it("calls the repair endpoints with the documented bodies", async () => {
    const { api, spy } = client(() => jsonResponse({}));
    await api.mergeLines([1, 2]);
    await api.splitLine(5, 120, 40);
    await api.addLine(3, [[1, 2], [3, 4]]);
    await api.redrawBaseline(8, [[1, 2], [3, 4]]);
    await api.ignoreRegion(3, { x: 1, y: 2, w: 3, h: 4 });
    await api.reorder(3, [2, 1]);
    await api.restore(3, [{ id: 1, baseline: [[0, 0], [1, 1]], polygon: [[0, 0], [1, 0], [1, 1]], text: null, text_source: null, confidence: null, is_ground_truth: false, ignored: false, page_id: 3, index: 0, crop_url: "" }], true);
    await api.deleteLine(9);
    const calls = spy.mock.calls.map(([url, init]) => [init?.method, String(url).replace("http://api.test", ""), init?.body ? JSON.parse(String(init.body)) : null]);
    expect(calls).toEqual([
      ["POST", "/lines/merge", { line_ids: [1, 2] }],
      ["POST", "/lines/5/split", { x: 120, y: 40 }],
      ["POST", "/pages/3/lines", { baseline: [[1, 2], [3, 4]] }],
      ["PATCH", "/lines/8", { baseline: [[1, 2], [3, 4]], regenerate_polygon: true }],
      ["POST", "/pages/3/ignore-region", { x: 1, y: 2, w: 3, h: 4 }],
      ["POST", "/pages/3/reorder", { line_ids: [2, 1] }],
      [
        "POST",
        "/pages/3/restore",
        { force: true, lines: [{ id: 1, baseline: [[0, 0], [1, 1]], polygon: [[0, 0], [1, 0], [1, 1]], text: "", text_source: null, confidence: null, predicted_by_model_id: null, predicted_text: null, is_ground_truth: false, ignored: false }] },
      ],
      ["DELETE", "/lines/9", null],
    ]);
  });

  it("gives a line's prediction state back on restore (undo)", async () => {
    const { api, spy } = client(() => jsonResponse({}));
    const line = { id: 1, baseline: [[0, 0], [1, 1]], polygon: [[0, 0], [1, 0], [1, 1]], text: "hello wxrld", text_source: "predicted", confidence: 0.9, predicted_by_model_id: 4, predicted_text: "hello wxrld", is_ground_truth: false, ignored: false, page_id: 3, index: 0, crop_url: "" };
    await api.restore(3, [line as never]);
    const sent = JSON.parse(String(spy.mock.calls[0]![1]!.body)).lines[0];
    expect(sent).toMatchObject({ predicted_by_model_id: 4, predicted_text: "hello wxrld", text_source: "predicted", confidence: 0.9 });
  });

  it("calls the health and suggestion endpoints", async () => {
    const { api, spy } = client(() => jsonResponse({}));
    await api.getHealth();
    await api.getHealthDetails();
    await api.listSuggestions();
    await api.listSuggestions("all");
    await api.generateSuggestions({ tokens: ["c/o"], max_tokens: 5 });
    await api.acceptSuggestion(3);
    await api.acceptSuggestion(4, { expansion: "as needed" });
    await api.rejectSuggestion(5);
    const calls = spy.mock.calls.map(([url, init]) => [init?.method, String(url).replace("http://api.test", ""), init?.body ? JSON.parse(String(init.body)) : null]);
    expect(calls).toEqual([
      ["GET", "/health", null],
      ["GET", "/health/details", null],
      ["GET", "/abbreviations/suggestions?status=pending", null],
      ["GET", "/abbreviations/suggestions?status=all", null],
      ["POST", "/abbreviations/suggestions/generate", { tokens: ["c/o"], max_tokens: 5 }],
      ["POST", "/abbreviations/suggestions/3/accept", {}],
      ["POST", "/abbreviations/suggestions/4/accept", { expansion: "as needed" }],
      ["POST", "/abbreviations/suggestions/5/reject", null],
    ]);
  });

  it("calls the training endpoints with the documented bodies", async () => {
    const { api, spy } = client(() => jsonResponse({ id: 1, warnings: [] }, 202));
    await api.startTraining({ epochs: 5, batch_size: 2 });
    await api.startTraining();
    await api.startPrediction({ model_id: 3, page_ids: [1, 2] });
    await api.cancelJob(7);
    await api.getTrainingSummary();
    await api.listModels();
    await api.activateModel(4);
    const calls = spy.mock.calls.map(([url, init]) => [init?.method, String(url).replace("http://api.test", ""), init?.body ? JSON.parse(String(init.body)) : null]);
    expect(calls).toEqual([
      ["POST", "/jobs/train", { epochs: 5, batch_size: 2 }],
      ["POST", "/jobs/train", {}],
      ["POST", "/jobs/predict", { model_id: 3, page_ids: [1, 2] }],
      ["POST", "/jobs/7/cancel", null],
      ["GET", "/training/summary", null],
      ["GET", "/models", null],
      ["POST", "/models/4/activate", null],
    ]);
  });

  it("surfaces a 422 from /jobs/train as an ApiError", async () => {
    const { api } = client(() => jsonResponse({ detail: "there are no usable ground-truth lines to train on" }, 422));
    await expect(api.startTraining()).rejects.toMatchObject({ status: 422, message: /no usable ground-truth/ });
  });
});
