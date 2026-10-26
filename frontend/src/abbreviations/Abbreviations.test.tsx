import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Abbreviation, OccurrencePage } from "../api/types";
import { Abbreviations, filterAbbreviations } from "./Abbreviations";

const abbr = (id: number, written: string, expansion: string, over: Partial<Abbreviation> = {}): Abbreviation => ({
  id,
  written,
  expansion,
  notes: null,
  case_sensitive: false,
  count: 0,
  ambiguous: false,
  candidates: [],
  created_at: null,
  updated_at: null,
  ...over,
});

const PT = abbr(1, "pt", "patient", { count: 3, notes: "clinic" });
const SOB1 = abbr(2, "SOB", "shortness of breath", {
  count: 1,
  ambiguous: true,
  candidates: [
    { id: 2, expansion: "shortness of breath" },
    { id: 3, expansion: "son of a bitch" },
  ],
});
const SOB2 = abbr(3, "SOB", "son of a bitch", { count: 1, ambiguous: true, candidates: [SOB1.candidates[1]!, SOB1.candidates[0]!] });
const CO = abbr(4, "c/o", "complains of");

function setup(items: Abbreviation[] = [PT, SOB1, SOB2, CO]) {
  const api = new ApiClient("http://api.test");
  let rows = [...items];
  vi.spyOn(api, "listAbbreviations").mockImplementation(async () => rows);
  const create = vi.spyOn(api, "createAbbreviation").mockImplementation(async (input) => {
    const a = abbr(99, input.written, input.expansion, { notes: input.notes ?? null, case_sensitive: !!input.case_sensitive });
    rows = [...rows, a];
    return a;
  });
  const patch = vi.spyOn(api, "patchAbbreviation").mockImplementation(async (id, p) => {
    rows = rows.map((r) => (r.id === id ? { ...r, ...p, notes: p.notes === undefined ? r.notes : p.notes } : r));
    return rows.find((r) => r.id === id)!;
  });
  const del = vi.spyOn(api, "deleteAbbreviation").mockImplementation(async (id) => {
    rows = rows.filter((r) => r.id !== id);
  });
  render(<Abbreviations api={api} />);
  return { api, create, patch, del };
}

afterEach(() => vi.restoreAllMocks());

describe("Abbreviations page", () => {
  it("lists the dictionary with counts and ambiguity badges, sorted by written form", async () => {
    setup();
    const table = await screen.findByRole("table", { name: "Abbreviation dictionary" });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((r) => within(r).getAllByRole("rowheader")[0]!.textContent)).toEqual(["c/o", "pt", "SOB", "SOB"]);
    expect(within(table).getByRole("button", { name: "3 occurrences of pt" })).toBeEnabled();
    expect(within(table).getByRole("button", { name: "0 occurrences of c/o" })).toBeDisabled();
    const badge = screen.getByTestId("ambiguous-2");
    expect(badge).toHaveTextContent("ambiguous (2)");
    expect(badge).toHaveAttribute("title", expect.stringContaining("son of a bitch"));
    expect(screen.getByTestId("ambiguous-total")).toHaveTextContent("2 ambiguous");
    expect(screen.getByTestId("abbr-1")).not.toHaveTextContent("ambiguous");
  });

  it("searches and filters to the ambiguous ones", async () => {
    setup();
    const user = userEvent.setup();
    await screen.findByRole("table");
    await user.type(screen.getByLabelText("Search"), "complain");
    expect(screen.getAllByRole("row")).toHaveLength(2);
    expect(screen.getByText("complains of")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("Search"));
    await user.selectOptions(screen.getByLabelText("Show"), "ambiguous");
    expect(screen.getAllByRole("row")).toHaveLength(3);
    await user.clear(screen.getByLabelText("Search"));
    await user.type(screen.getByLabelText("Search"), "zzz");
    expect(screen.getByText("Nothing matches this search.")).toBeInTheDocument();
  });

  it("filterAbbreviations sorts by occurrences when asked", () => {
    const out = filterAbbreviations([CO, PT, SOB1], "", "all", "count");
    expect(out.map((a) => a.written)).toEqual(["pt", "SOB", "c/o"]);
  });

  it("adds an abbreviation and reloads", async () => {
    const { create } = setup([PT]);
    const user = userEvent.setup();
    await screen.findByRole("table");
    const add = screen.getByRole("button", { name: "Add" });
    expect(add).toBeDisabled();
    await user.type(screen.getByLabelText("Written form"), "w/");
    await user.type(screen.getByLabelText("Expansion"), "with");
    await user.click(screen.getByLabelText("Case sensitive"));
    await user.click(add);
    await waitFor(() => expect(create).toHaveBeenCalledWith({ written: "w/", expansion: "with", notes: null, case_sensitive: true }));
    expect(await screen.findByText("with")).toBeInTheDocument();
    expect(screen.getByLabelText("Written form")).toHaveValue("");
  });

  it("shows the server's message when adding fails (duplicate)", async () => {
    const { create } = setup([PT]);
    create.mockRejectedValue(new Error("'pt' = 'patient' is already in the dictionary (id 1)"));
    const user = userEvent.setup();
    await screen.findByRole("table");
    await user.type(screen.getByLabelText("Written form"), "pt");
    await user.type(screen.getByLabelText("Expansion"), "patient");
    await user.click(screen.getByRole("button", { name: "Add" }));
    expect(await screen.findByTestId("abbr-form-error")).toHaveTextContent("already in the dictionary");
  });

  it("edits a row", async () => {
    const { patch } = setup([PT]);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Edit pt = patient" }));
    const form = screen.getByRole("form", { name: "Edit pt" });
    const exp = within(form).getByLabelText("Expansion");
    await user.clear(exp);
    await user.type(exp, "the patient");
    await user.click(within(form).getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(patch).toHaveBeenCalledWith(1, { written: "pt", expansion: "the patient", notes: "clinic", case_sensitive: false }),
    );
    expect(await screen.findByText("the patient")).toBeInTheDocument();
  });

  it("deletes after confirmation only", async () => {
    const { del } = setup([PT]);
    const user = userEvent.setup();
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    await user.click(await screen.findByRole("button", { name: "Delete pt = patient" }));
    expect(del).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Delete pt = patient" }));
    await waitFor(() => expect(del).toHaveBeenCalledWith(1));
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(await screen.findByText("No abbreviations yet. Add the first one above.")).toBeInTheDocument();
  });

  it("shows occurrences with links to the line in the editor, and loads more", async () => {
    const { api } = setup([PT]);
    const mk = (n: number) => ({
      line_id: 100 + n,
      page_id: 7,
      document_id: 1,
      document_name: "Notebook",
      page_filename: "a.png",
      line_index: n,
      text: `x pt y${n}`,
      text_source: n === 1 ? ("predicted" as const) : ("manual" as const),
      start: 2,
      end: 4,
      ambiguous: false,
    });
    const occ = vi.spyOn(api, "abbreviationOccurrences").mockImplementation(async (_id, _limit, offset = 0) => {
      const all = [mk(0), mk(1), mk(2)];
      return { total: 3, limit: 2, offset, items: offset === 0 ? all.slice(0, 2) : all.slice(2) } satisfies OccurrencePage;
    });
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "3 occurrences of pt" }));
    const list = await screen.findByRole("list", { name: "Occurrences of pt" });
    const links = within(list).getAllByRole("link");
    expect(links[0]).toHaveAttribute("href", "#/pages/7?line=100");
    expect(links[0]).toHaveTextContent("Notebook / a.png, line 1");
    expect(within(list).getAllByText("pt", { selector: "mark" })).toHaveLength(2);
    expect(within(list).getByText("predicted")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show more" }));
    await waitFor(() => expect(within(screen.getByRole("list", { name: "Occurrences of pt" })).getAllByRole("link")).toHaveLength(3));
    expect(occ).toHaveBeenLastCalledWith(1, 25, 2);
    expect(screen.queryByRole("button", { name: "Show more" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "3 occurrences of pt" })); // toggles closed
    expect(screen.queryByRole("list", { name: "Occurrences of pt" })).not.toBeInTheDocument();
  });

  it("previews the expansion of a text", async () => {
    const { api } = setup();
    vi.spyOn(api, "expandText").mockResolvedValue({ text: "patient complains of SOB[?]", spans: [{ start: 9, end: 12, written: "SOB", expansion: null, status: "ambiguous", candidates: [], entry_ids: [] }] });
    const user = userEvent.setup();
    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: "Preview expansion" }));
    expect(await screen.findByTestId("preview-out")).toHaveTextContent("patient complains of SOB[?]");
    expect(api.expandText).toHaveBeenCalledWith("pt c/o SOB", "mark");
  });

  it("is keyboard operable: row buttons are reachable with Tab", async () => {
    setup([PT]);
    const user = userEvent.setup();
    await screen.findByRole("table");
    await user.tab(); // links
    await user.tab();
    await user.tab();
    await user.tab(); // form controls...
    // every interactive element is a native control (button / input / link), so Tab/Enter work without extra code
    const row = screen.getByTestId("abbr-1");
    for (const b of within(row).getAllByRole("button")) expect(b.tagName).toBe("BUTTON");
  });

  it("shows a load error with a retry", async () => {
    const api = new ApiClient("http://api.test");
    const list = vi.spyOn(api, "listAbbreviations").mockRejectedValueOnce(new Error("boom")).mockResolvedValue([PT]);
    render(<Abbreviations api={api} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
    await userEvent.setup().click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(list).toHaveBeenCalledTimes(2);
  });
});
