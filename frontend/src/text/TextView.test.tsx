import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Page, PageDetail, PageText } from "../api/types";
import { TextView } from "./TextView";

const text: PageText = {
  page_id: 2,
  document_id: 1,
  expanded: true,
  ambiguous: "mark",
  include_predicted: true,
  text: "",
  lines: [
    {
      line_id: 21,
      index: 0,
      raw: "pt c/o SOB",
      expanded: "patient complains of SOB[?]",
      text_source: "manual",
      is_ground_truth: true,
      confidence: null,
      spans: [
        { start: 0, end: 2, written: "pt", expansion: "patient", status: "expanded", candidates: [], entry_ids: [1] },
        { start: 3, end: 6, written: "c/o", expansion: "complains of", status: "expanded", candidates: [], entry_ids: [2] },
        { start: 7, end: 10, written: "SOB", expansion: null, status: "ambiguous", candidates: ["shortness of breath", "son of a bitch"], entry_ids: [3, 4] },
      ],
    },
    { line_id: 22, index: 2, raw: "seen", expanded: "seen", text_source: "predicted", is_ground_truth: false, confidence: 0.4, spans: [] },
  ],
};

const detail = { id: 2, document_id: 1, original_filename: "b.png" } as unknown as PageDetail;
const sibling = (id: number) => ({ id }) as Page;

function setup(t = text) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "getPage").mockResolvedValue(detail);
  vi.spyOn(api, "listPages").mockResolvedValue([sibling(1), sibling(2), sibling(3)]);
  const pt = vi.spyOn(api, "pageText").mockResolvedValue(t);
  render(<TextView api={api} pageId={2} />);
  return { api, pt };
}

afterEach(() => vi.restoreAllMocks());

describe("Text view", () => {
  it("shows raw and expanded text side by side with highlighted abbreviations", async () => {
    setup();
    const row = await screen.findByTestId("text-line-21");
    const cells = within(row).getAllByRole("cell");
    expect(cells[0]).toHaveTextContent("pt c/o SOB");
    expect(cells[1]).toHaveTextContent("patient complains of SOB[?]");
    const marks = within(cells[0]!).getAllByText(/./, { selector: "mark" });
    expect(marks.map((m) => m.textContent)).toEqual(["pt", "c/o", "SOB"]);
    expect(marks[2]).toHaveClass("ambiguous");
    expect(marks[2]).toHaveAttribute("title", "Ambiguous: shortness of breath / son of a bitch");
    expect(marks[0]).toHaveAttribute("title", "= patient");
    expect(screen.getByRole("columnheader", { name: "As written" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Expanded" })).toBeInTheDocument();
    expect(within(screen.getByTestId("text-line-22")).getByText("predicted")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("1 line has an ambiguous abbreviation");
    expect(within(row).getByRole("link", { name: "1" })).toHaveAttribute("href", "#/pages/2?line=21");
  });

  it("links to the editor and the neighbouring pages", async () => {
    setup();
    await screen.findByTestId("text-line-21");
    expect(screen.getByRole("link", { name: "Back to editor" })).toHaveAttribute("href", "#/pages/2");
    expect(screen.getByRole("link", { name: /Prev page/ })).toHaveAttribute("href", "#/pages/1/text");
    expect(screen.getByRole("link", { name: /Next page/ })).toHaveAttribute("href", "#/pages/3/text");
    expect(screen.getByRole("heading", { name: "Text: b.png" })).toBeInTheDocument();
  });

  it("the options reload the text", async () => {
    const { pt } = setup();
    await screen.findByTestId("text-line-21");
    expect(pt).toHaveBeenLastCalledWith(2, { ambiguous: "mark", include_predicted: true });
    const user = userEvent.setup();
    await user.click(screen.getByLabelText(/Mark ambiguous/));
    await user.click(screen.getByLabelText("Show predicted lines"));
    await vi.waitFor(() => expect(pt).toHaveBeenLastCalledWith(2, { ambiguous: "raw", include_predicted: false }));
  });

  it("explains an empty page and shows errors", async () => {
    setup({ ...text, lines: [] });
    expect(await screen.findByText(/no transcribed text yet/)).toBeInTheDocument();
    vi.restoreAllMocks();
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "getPage").mockRejectedValue(new Error("page 2 not found"));
    vi.spyOn(api, "pageText").mockRejectedValue(new Error("page 2 not found"));
    render(<TextView api={api} pageId={2} />);
    expect((await screen.findAllByRole("alert"))[0]).toHaveTextContent("page 2 not found");
  });
});
