import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Line, PageDetail } from "../api/types";
import { PageEditor } from "./PageEditor";

function mkLine(id: number, over: Partial<Line> = {}): Line {
  return {
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
    text: null,
    text_source: null,
    confidence: null,
    is_ground_truth: false,
    ignored: false,
    crop_url: `/lines/${id}/crop.png`,
    ...over,
  };
}

function mkPage(lines: Line[]): PageDetail {
  return {
    id: 1,
    document_id: 1,
    original_filename: "p1.png",
    status: "segmented",
    width: 100,
    height: 200,
    warnings: [],
    imported_at: "",
    segmented_at: "",
    image_url: "/pages/1/working",
    original_url: "/pages/1/original",
    line_count: lines.length,
    ground_truth_count: 0,
    active_ground_truth_count: 0,
    ignored_count: 0,
    lines,
  };
}

function setup(lines: Line[], patch: ApiClient["patchLine"]) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "getPage").mockResolvedValue(mkPage(lines));
  vi.spyOn(api, "listPages").mockResolvedValue([]);
  const spy = vi.spyOn(api, "patchLine").mockImplementation(patch);
  render(<PageEditor api={api} pageId={1} />);
  return { api, spy };
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

describe("PageEditor keyboard flow", () => {
  it("Enter saves as ground truth and moves to the next active line, skipping ignored ones", async () => {
    const lines = [mkLine(1), mkLine(2, { ignored: true }), mkLine(3)];
    const { spy } = setup(lines, async (id, p) => ({ ...lines[id - 1]!, ...p, text_source: "manual" }) as Line);
    const user = userEvent.setup();
    const input = await screen.findByLabelText("Transcription of line 1");
    await user.type(input, "hello world{Enter}");
    expect(spy).toHaveBeenCalledWith(1, { text: "hello world", is_ground_truth: true });
    expect(await screen.findByLabelText("Transcription of line 3")).toHaveFocus();
    await waitFor(() => expect(screen.getByTestId("save-state")).toHaveTextContent("All changes saved"));
  });

  it("Tab and Shift+Tab move without saving and keep typed text", async () => {
    const { spy } = setup([mkLine(1), mkLine(2)], async () => mkLine(1));
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Transcription of line 1"), "draft");
    await user.tab();
    expect(await screen.findByLabelText("Transcription of line 2")).toBeInTheDocument();
    await user.tab({ shift: true });
    expect(await screen.findByLabelText("Transcription of line 1")).toHaveValue("draft");
    expect(spy).not.toHaveBeenCalled();
  });

  it("Esc reverts the unsaved edit", async () => {
    setup([mkLine(1, { text: "saved" })], async () => mkLine(1));
    const user = userEvent.setup();
    const input = await screen.findByLabelText("Transcription of line 1");
    await user.type(input, "XYZ");
    expect(input).toHaveValue("savedXYZ");
    await user.keyboard("{Escape}");
    expect(input).toHaveValue("saved");
  });

  it("Ctrl+I toggles ignore through the API", async () => {
    const { spy } = setup([mkLine(1)], async (_id, p) => ({ ...mkLine(1), ...p }) as Line);
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Control>}i{/Control}");
    expect(spy).toHaveBeenCalledWith(1, { ignored: true });
    await waitFor(() => expect(screen.getByLabelText("Transcription of line 1")).toHaveAttribute("readonly"));
  });

  it("a failed save keeps the typed text, shows an error, and can be retried", async () => {
    let fail = true;
    const lines = [mkLine(1), mkLine(2)];
    const { spy } = setup(lines, async (id, p) => {
      if (fail) throw new Error("HTTP 500");
      return { ...lines[id - 1]!, ...p, text_source: "manual" } as Line;
    });
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Transcription of line 1"), "precious text{Enter}");
    const alert = await screen.findByTestId("save-error");
    expect(alert).toHaveTextContent("HTTP 500");
    expect(alert).toHaveTextContent("Your text is kept");
    expect(screen.getByTestId("save-state")).toHaveTextContent("Save failed");
    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry save" }));
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy).toHaveBeenLastCalledWith(1, { text: "precious text", is_ground_truth: true });
    await waitFor(() => expect(screen.queryByTestId("save-error")).toBeNull());
    await waitFor(() => expect(screen.getByTestId("save-state")).toHaveTextContent("All changes saved"));
  });

  it("opens the help overlay with Ctrl+/ and closes it with Esc", async () => {
    setup([mkLine(1)], async () => mkLine(1));
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Control>}/{/Control}");
    expect(await screen.findByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
