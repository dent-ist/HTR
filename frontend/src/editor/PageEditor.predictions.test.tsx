import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Line, PageDetail } from "../api/types";
import { navigate } from "../router";
import { PageEditor } from "./PageEditor";

const mk = (id: number, over: Partial<Line> = {}): Line => ({
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
});

function setup(lines: Line[]) {
  const api = new ApiClient("http://api.test");
  const page = {
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
  } as PageDetail;
  vi.spyOn(api, "getPage").mockResolvedValue(page);
  vi.spyOn(api, "listPages").mockResolvedValue([]);
  const patch = vi
    .spyOn(api, "patchLine")
    .mockImplementation(async (id, p) => ({ ...lines[id - 1]!, ...p, text_source: "manual", confidence: null }) as Line);
  render(<PageEditor api={api} pageId={1} />);
  return { api, patch };
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

const predicted = [
  mk(1, { text: "hello wxrld", text_source: "predicted", confidence: 0.62 }),
  mk(2, { text: "second line", text_source: "predicted", confidence: 0.95 }),
  mk(3, { text: "middling", text_source: "predicted", confidence: 0.8 }),
];

describe("predictions in the editor", () => {
  it("prefills the selected prediction, labels it, and shows confidence as number + word", async () => {
    setup(predicted);
    const input = await screen.findByLabelText("Transcription of line 1");
    expect(input).toHaveValue("hello wxrld");
    expect(screen.getByText(/Predicted text, not yet checked/)).toBeInTheDocument();
    expect(input).toHaveAccessibleDescription(/Enter accepts it as ground truth/);
    expect(within(screen.getByTestId("predicted-badge")).getByText(/62% low/)).toBeInTheDocument();
    const list = screen.getByRole("navigation", { name: "Lines" });
    expect(within(list).getAllByText(/62% low/).length).toBeGreaterThan(0);
    expect(within(list).getByText(/95% high/)).toBeInTheDocument();
    expect(within(list).getByText(/80% medium/)).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Confidence legend" })).toHaveTextContent("under 70%");
  });

  it("Enter accepts the prediction in one keystroke: ground truth with the predicted text", async () => {
    const { patch } = setup(predicted);
    const user = userEvent.setup();
    await screen.findByLabelText("Transcription of line 1");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(patch).toHaveBeenCalledWith(1, { text: "hello wxrld", is_ground_truth: true }));
    expect(await screen.findByLabelText("Transcription of line 2")).toHaveValue("second line");
    expect(screen.getByTestId("save-state")).toHaveTextContent("All changes saved");
  });

  it("editing then Enter saves the correction; Esc restores the prediction", async () => {
    const { patch } = setup(predicted);
    const user = userEvent.setup();
    const input = await screen.findByLabelText("Transcription of line 1");
    await user.type(input, "!");
    expect(screen.getByTestId("save-state")).toHaveTextContent("Unsaved changes");
    await user.keyboard("{Escape}");
    expect(input).toHaveValue("hello wxrld");
    expect(screen.getByTestId("save-state")).toHaveTextContent("All changes saved");
    await user.clear(input);
    await user.type(input, "hello world{Enter}");
    await waitFor(() => expect(patch).toHaveBeenCalledWith(1, { text: "hello world", is_ground_truth: true }));
  });

  it("a prefilled prediction does not trigger the leave-page guard until it is edited", async () => {
    setup(predicted);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    await screen.findByLabelText("Transcription of line 1");
    navigate("#/");
    expect(confirm).not.toHaveBeenCalled();
    await userEvent.setup().type(screen.getByLabelText("Transcription of line 1"), "x");
    navigate("#/training");
    expect(confirm).toHaveBeenCalled();
  });

  it("links to the review page", async () => {
    setup(predicted);
    expect(await screen.findByRole("button", { name: "Review" })).toBeInTheDocument();
  });

  it("shows no prediction label for ordinary lines", async () => {
    setup([mk(1, { text: "typed", text_source: "manual", is_ground_truth: true })]);
    await screen.findByLabelText("Transcription of line 1");
    expect(screen.queryByTestId("predicted-badge")).toBeNull();
    expect(screen.queryByText(/Predicted text, not yet checked/)).toBeNull();
  });
});
