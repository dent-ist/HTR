import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Line, ReviewItem, ReviewNext } from "../api/types";
import { Review } from "./Review";

const item = (id: number, text: string, conf: number, page = 5): ReviewItem => ({
  line: {
    id,
    page_id: page,
    index: 0,
    baseline: [],
    polygon: [],
    text,
    text_source: "predicted",
    confidence: conf,
    is_ground_truth: false,
    ignored: false,
    crop_url: `/lines/${id}/crop.png`,
  },
  page_id: page,
  crop_url: `/lines/${id}/crop.png`,
  predicted_text: text,
  confidence: conf,
  model_id: 2,
});

const queue = [item(11, "hello wxrld", 0.41), item(12, "second", 0.6), item(13, "third", 0.8)];

/** A fake server: next = first queue item not excluded; patching removes it from the queue. */
function setup(items = queue) {
  const api = new ApiClient("http://api.test");
  const gone = new Set<number>();
  vi.spyOn(api, "getTrainingSummary").mockRejectedValue(new Error("none"));
  const next = vi.spyOn(api, "reviewNext").mockImplementation(async (exclude = []) => {
    const left = items.filter((i) => !gone.has(i.line.id) && !exclude.includes(i.line.id));
    return { items: left.slice(0, 1), remaining: left.length } satisfies ReviewNext;
  });
  const patch = vi.spyOn(api, "patchLine").mockImplementation(async (id) => {
    gone.add(id);
    return {} as Line;
  });
  render(<Review api={api} />);
  return { api, next, patch };
}

describe("Review page", () => {
  it("shows the least certain line first: crop, confidence, prefilled prediction, queue size", async () => {
    setup();
    const input = await screen.findByLabelText("Text of the line (prefilled with the prediction)");
    expect(input).toHaveValue("hello wxrld");
    expect(input).toHaveFocus();
    expect(screen.getByText(/41% low/)).toBeInTheDocument();
    expect(screen.getByTestId("review-crop")).toHaveAttribute("src", "http://api.test/lines/11/crop.png");
    expect(screen.getByTestId("review-progress")).toHaveTextContent("3 lines left, least certain first.");
    expect(screen.getByTestId("review-hint")).toHaveTextContent("Enter accept + next");
    expect(screen.getByTestId("open-in-editor")).toHaveAttribute("href", "#/pages/5");
  });

  it("Enter accepts an unchanged prediction and loads the next line", async () => {
    const { patch } = setup();
    const user = userEvent.setup();
    await screen.findByDisplayValue("hello wxrld");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(patch).toHaveBeenCalledWith(11, { text: "hello wxrld", is_ground_truth: true }));
    expect(await screen.findByDisplayValue("second")).toHaveFocus();
    expect(screen.getByTestId("review-progress")).toHaveTextContent("2 lines left");
    expect(screen.getByTestId("review-progress")).toHaveTextContent("1 reviewed (1 accepted, 0 corrected, 0 ignored)");
  });

  it("an edited prediction is saved as a correction", async () => {
    const { patch } = setup();
    const user = userEvent.setup();
    const input = await screen.findByDisplayValue("hello wxrld");
    await user.clear(input);
    await user.type(input, "hello world");
    expect(screen.getByTestId("review-hint")).toHaveTextContent("Enter save correction");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(patch).toHaveBeenCalledWith(11, { text: "hello world", is_ground_truth: true }));
    await screen.findByDisplayValue("second");
    expect(screen.getByTestId("review-progress")).toHaveTextContent("(0 accepted, 1 corrected, 0 ignored)");
  });

  it("Esc restores the prediction", async () => {
    setup();
    const user = userEvent.setup();
    const input = await screen.findByDisplayValue("hello wxrld");
    await user.type(input, "zzz");
    await user.keyboard("{Escape}");
    expect(input).toHaveValue("hello wxrld");
  });

  it("Tab skips without saving and the skipped line stays out until reset", async () => {
    const { patch, next } = setup();
    const user = userEvent.setup();
    await screen.findByDisplayValue("hello wxrld");
    await user.keyboard("{Tab}");
    await screen.findByDisplayValue("second");
    expect(patch).not.toHaveBeenCalled();
    expect(next).toHaveBeenLastCalledWith([11], 1);
    await user.keyboard("{Enter}{Enter}"); // second, then third
    expect(await screen.findByTestId("review-empty")).toHaveTextContent("Nothing left except 1 skipped line");
    await user.click(screen.getByRole("button", { name: "Reset skipped" }));
    expect(await screen.findByDisplayValue("hello wxrld")).toBeInTheDocument();
  });

  it("Ctrl+I ignores the line and records it in the session", async () => {
    const { patch } = setup();
    const user = userEvent.setup();
    await screen.findByDisplayValue("hello wxrld");
    await user.keyboard("{Control>}i{/Control}");
    await waitFor(() => expect(patch).toHaveBeenCalledWith(11, { ignored: true }));
    await screen.findByDisplayValue("second");
    expect(screen.getByTestId("review-progress")).toHaveTextContent("(0 accepted, 0 corrected, 1 ignored)");
  });

  it("refuses to save an empty line", async () => {
    const { patch } = setup();
    const user = userEvent.setup();
    const input = await screen.findByDisplayValue("hello wxrld");
    await user.clear(input);
    await user.keyboard("{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot save an empty line");
    expect(patch).not.toHaveBeenCalled();
  });

  it("a three-line session ends with the empty-queue message", async () => {
    setup();
    const user = userEvent.setup();
    await screen.findByDisplayValue("hello wxrld");
    await user.keyboard("{Enter}");
    await screen.findByDisplayValue("second");
    await user.keyboard("{Enter}");
    await screen.findByDisplayValue("third");
    await user.keyboard("{Enter}");
    expect(await screen.findByTestId("review-empty")).toHaveTextContent("No predictions are waiting for review.");
    expect(screen.getByTestId("review-progress")).toHaveTextContent("3 reviewed (3 accepted");
  });

  it("shows a failed save and keeps the line", async () => {
    const { api } = setup();
    vi.spyOn(api, "patchLine").mockRejectedValue(new Error("save failed: disk"));
    const user = userEvent.setup();
    await screen.findByDisplayValue("hello wxrld");
    await user.keyboard("{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("save failed: disk");
    expect(screen.getByDisplayValue("hello wxrld")).toBeInTheDocument();
  });

  it("shows the retrain notice when suggested", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "reviewNext").mockResolvedValue({ items: [], remaining: 0 });
    vi.spyOn(api, "getTrainingSummary").mockResolvedValue({
      retrain_suggested: true,
      retrain_reason: "52 new ground-truth lines since the last round (about 50 suggested)",
    } as never);
    render(<Review api={api} />);
    expect(await screen.findByTestId("retrain-notice")).toHaveTextContent("52 new ground-truth lines");
  });
});
