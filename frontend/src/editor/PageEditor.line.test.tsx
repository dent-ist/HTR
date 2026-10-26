import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Line, PageDetail } from "../api/types";
import { PageEditor } from "./PageEditor";

const line = (id: number, text: string): Line => ({
  id,
  page_id: 1,
  index: id - 1,
  baseline: [
    [10, 20 * id],
    [90, 20 * id],
  ],
  polygon: [
    [10, 20 * id - 8],
    [90, 20 * id - 8],
    [90, 20 * id + 8],
    [10, 20 * id + 8],
  ],
  text,
  text_source: "manual",
  confidence: null,
  is_ground_truth: false,
  ignored: false,
  crop_url: `/lines/${id}/crop.png`,
});

const page = (): PageDetail =>
  ({
    id: 1,
    document_id: 1,
    original_filename: "p1.png",
    status: "segmented",
    width: 100,
    height: 200,
    warnings: [],
    image_url: "/pages/1/working",
    original_url: "/pages/1/original",
    lines: [line(1, "first"), line(2, "second"), line(3, "pt c/o")],
  }) as unknown as PageDetail;

function setup(lineId?: number) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "getPage").mockResolvedValue(page());
  vi.spyOn(api, "listPages").mockResolvedValue([]);
  return render(<PageEditor api={api} pageId={1} lineId={lineId} />);
}

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(() => vi.restoreAllMocks());

describe("PageEditor opened on a line (link from the Abbreviations page)", () => {
  it("selects the requested line", async () => {
    setup(3);
    expect(await screen.findByLabelText("Transcription of line 3")).toHaveValue("pt c/o");
  });

  it("ignores a line id that is not on the page and opens on the default line", async () => {
    setup(99);
    expect(await screen.findByLabelText(/Transcription of line 1/)).toHaveValue("first");
  });

  it("has a Text view button", async () => {
    setup();
    await screen.findByLabelText(/Transcription of line/);
    expect(screen.getByRole("button", { name: "Text view" })).toBeEnabled();
  });
});
