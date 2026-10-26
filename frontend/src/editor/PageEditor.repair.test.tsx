import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";
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

function setup(lines: Line[]) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "getPage").mockResolvedValue(mkPage(lines));
  vi.spyOn(api, "listPages").mockResolvedValue([]);
  render(<PageEditor api={api} pageId={1} />);
  return api;
}

const merged = mkLine(7, { baseline: [[10, 20], [90, 40]] });

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

describe("PageEditor repair operations", () => {
  it("Alt+J merges the selected line with the next, and Ctrl+Z restores the snapshot", async () => {
    const original = [mkLine(1), mkLine(2), mkLine(3)];
    const api = setup(original);
    const mergedPage = mkPage([{ ...merged, index: 0 }, { ...mkLine(3), index: 1 }]);
    const mergeSpy = vi.spyOn(api, "mergeLines").mockResolvedValue({ line: mergedPage.lines[0]!, removed_ids: [1, 2], page: mergedPage });
    const restoreSpy = vi.spyOn(api, "restore").mockResolvedValue(mkPage(original));
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Alt>}j{/Alt}");
    await waitFor(() => expect(mergeSpy).toHaveBeenCalledWith([1, 2]));
    expect(await screen.findByText(/Line 1 of 2/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Undo: Merged lines/ })).toBeEnabled();
    await user.keyboard("{Control>}z{/Control}");
    await waitFor(() => expect(restoreSpy).toHaveBeenCalledTimes(1));
    expect(restoreSpy.mock.calls[0]![1].map((l) => l.id)).toEqual([1, 2, 3]);
    expect(await screen.findByText(/Line 1 of 3/)).toBeInTheDocument();
  });

  it("asks before merging transcribed lines and does nothing when declined", async () => {
    const api = setup([mkLine(1), mkLine(2, { text: "hello", text_source: "manual", is_ground_truth: true })]);
    const mergeSpy = vi.spyOn(api, "mergeLines");
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Alt>}j{/Alt}");
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/resets its ground-truth/));
    expect(mergeSpy).not.toHaveBeenCalled();
  });

  it("refuses to merge a line with unsaved typed text instead of losing it", async () => {
    const api = setup([mkLine(1), mkLine(2)]);
    const mergeSpy = vi.spyOn(api, "mergeLines");
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Transcription of line 1"), "unsaved");
    await user.keyboard("{Alt>}j{/Alt}");
    expect(await screen.findByTestId("repair-message")).toHaveTextContent(/unsaved text/i);
    expect(mergeSpy).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Transcription of line 1")).toHaveValue("unsaved");
  });

  it("undo that would drop typed text asks for confirmation; declining changes nothing", async () => {
    const original = [mkLine(1), mkLine(2)];
    const api = setup(original);
    const after = mkPage([{ ...merged, index: 0 }]);
    vi.spyOn(api, "mergeLines").mockResolvedValue({ line: after.lines[0]!, removed_ids: [1, 2], page: after });
    const restoreSpy = vi
      .spyOn(api, "restore")
      .mockRejectedValue(new ApiError(409, 'restoring would discard text of 1 line(s): "typed after"'));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Alt>}j{/Alt}");
    await screen.findByText(/Line 1 of 1/);
    await user.keyboard("{Alt>}z{/Alt}");
    await waitFor(() => expect(confirm).toHaveBeenCalledWith(expect.stringContaining("typed after")));
    expect(restoreSpy).toHaveBeenCalledTimes(1); // not forced
    expect(await screen.findByText(/Line 1 of 1/)).toBeInTheDocument(); // unchanged
    expect(screen.getByRole("button", { name: /Undo: Merged lines/ })).toBeEnabled(); // still undoable

    confirm.mockReturnValue(true);
    restoreSpy.mockReset();
    restoreSpy.mockResolvedValueOnce(mkPage(original));
    restoreSpy.mockRejectedValueOnce(new ApiError(409, "restoring would discard text")); // never used
    await user.keyboard("{Alt>}z{/Alt}");
    await waitFor(() => expect(restoreSpy).toHaveBeenCalled());
  });

  it("modes: Alt+M shows the indicator, clicking list lines picks them, Enter merges, Esc cancels", async () => {
    const api = setup([mkLine(1), mkLine(2), mkLine(3)]);
    const mergeSpy = vi.spyOn(api, "mergeLines").mockResolvedValue({
      line: mkLine(9),
      removed_ids: [1, 3],
      page: mkPage([mkLine(9), mkLine(2)]),
    });
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    const banner = screen.getByTestId("mode-banner");
    expect(banner).toHaveAttribute("data-mode", "select");
    await user.keyboard("{Alt>}m{/Alt}");
    expect(banner).toHaveAttribute("data-mode", "merge");
    await user.keyboard("{Escape}");
    expect(banner).toHaveAttribute("data-mode", "select");
    await user.keyboard("{Alt>}m{/Alt}");
    await user.click(screen.getByRole("button", { name: /^3/ })); // line list entry 3
    await user.keyboard("{Enter}");
    await waitFor(() => expect(mergeSpy).toHaveBeenCalledWith([1, 3]));
    await waitFor(() => expect(banner).toHaveAttribute("data-mode", "select"));
  });

  it("shows API errors without losing state", async () => {
    const api = setup([mkLine(1), mkLine(2)]);
    vi.spyOn(api, "mergeLines").mockRejectedValue(new ApiError(422, "all lines must be on the same page"));
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Alt>}j{/Alt}");
    expect(await screen.findByTestId("repair-message")).toHaveTextContent("all lines must be on the same page");
    expect(screen.getByText(/Line 1 of 2/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Undo$/ })).toBeDisabled();
  });
});
