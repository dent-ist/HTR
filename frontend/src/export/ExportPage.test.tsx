import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";
import type { BundleImportResult, Document, Job } from "../api/types";
import { ExportPage, exportFileName } from "./ExportPage";

const docs: Document[] = [
  { id: 1, name: "Field notes", imported_at: "", page_count: 2 },
  { id: 2, name: "Other", imported_at: "", page_count: 1 },
];

const job = (over: Partial<Job> = {}): Job => ({
  id: 5,
  type: "train",
  status: "running",
  payload: {},
  progress: 0.3,
  error: null,
  result: null,
  created_at: "",
  started_at: null,
  finished_at: null,
  cancel_requested: false,
  ...over,
});

function setup() {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "listDocuments").mockResolvedValue(docs);
  vi.spyOn(api, "listJobs").mockResolvedValue([]);
  const exportBlob = vi.spyOn(api, "exportBlob").mockResolvedValue(new Blob(["=== Field notes / page 1 (a.png) ===\npatient complains of\n"]));
  const save = vi.fn();
  const startDownload = vi.fn();
  render(<ExportPage api={api} save={save} startDownload={startDownload} />);
  return { api, exportBlob, save, startDownload };
}

afterEach(() => vi.restoreAllMocks());

describe("Export page", () => {
  it("downloads expanded text for a document with the chosen options", async () => {
    const { exportBlob, save } = setup();
    const user = userEvent.setup();
    await screen.findByRole("option", { name: "Field notes" });
    await user.selectOptions(screen.getByLabelText("What to export"), "1");
    await user.click(screen.getByLabelText(/Mark ambiguous abbreviations/));
    await user.click(screen.getByLabelText(/Include predicted lines/));
    await user.click(screen.getByRole("button", { name: "Download text" }));
    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(exportBlob).toHaveBeenCalledWith({ format: "txt", expanded: true, ambiguous: "mark", document_id: 1, include_predicted: true });
    expect(save.mock.calls[0]![1]).toBe("htr-Field-notes-expanded.txt");
    expect(await screen.findByTestId("export-saved")).toHaveTextContent("htr-Field-notes-expanded.txt");
  });

  it("raw text, all documents; the ambiguity option only applies to expanded text", async () => {
    const { exportBlob } = setup();
    const user = userEvent.setup();
    const expand = screen.getByLabelText("Expand abbreviations");
    const mark = screen.getByLabelText(/Mark ambiguous abbreviations/);
    expect(expand).toBeChecked();
    expect(mark).toBeEnabled();
    await user.click(expand);
    expect(mark).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Download text" }));
    await waitFor(() => expect(exportBlob).toHaveBeenCalled());
    expect(exportBlob).toHaveBeenCalledWith({ format: "txt", expanded: false, ambiguous: "raw", document_id: undefined, include_predicted: false });
  });

  it("PAGE-XML export is not expanded and downloads a zip", async () => {
    const { exportBlob, save } = setup();
    const user = userEvent.setup();
    await user.click(screen.getByRole("radio", { name: /PAGE-XML/ }));
    expect(screen.getByLabelText("Expand abbreviations")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Download PAGE-XML zip" }));
    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(exportBlob).toHaveBeenCalledWith(expect.objectContaining({ format: "pagexml", expanded: false }));
    expect(save.mock.calls[0]![1]).toBe("htr-all-pagexml.zip");
  });

  it("previews the text and shows errors inline", async () => {
    const { exportBlob } = setup();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Preview text" }));
    expect(await screen.findByLabelText("Export preview")).toHaveTextContent("patient complains of");
    exportBlob.mockRejectedValueOnce(new ApiError(404, "document 9 not found"));
    await user.click(screen.getByRole("button", { name: "Download text" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("document 9 not found");
  });

  it("warns that bundles hold personal data and are saved only where the user chooses", () => {
    setup();
    expect(screen.getByRole("note")).toHaveTextContent(/personal data/);
    expect(screen.getByRole("note")).toHaveTextContent(/only where you\s+choose/);
    expect(screen.getByRole("note")).toHaveTextContent(/nothing is uploaded/);
    expect(screen.getByText(/refused if the project already has/)).toBeInTheDocument();
  });

  it("starts the streamed bundle download when no job is active", async () => {
    const { startDownload } = setup();
    await userEvent.setup().click(screen.getByRole("button", { name: "Export project bundle" }));
    await waitFor(() => expect(startDownload).toHaveBeenCalledWith("http://api.test/bundle/export"));
    expect(await screen.findByTestId("bundle-note")).toHaveTextContent(/nothing is uploaded/);
  });

  it("does not start the bundle download while a job is running", async () => {
    const { api, startDownload } = setup();
    vi.spyOn(api, "listJobs").mockImplementation(async (f) => (f?.status === "running" ? [job()] : []));
    await userEvent.setup().click(screen.getByRole("button", { name: "Export project bundle" }));
    expect(await screen.findByTestId("bundle-error")).toHaveTextContent("A train job is running (job 5)");
    expect(startDownload).not.toHaveBeenCalled();
  });

  it("imports a bundle and summarizes it", async () => {
    const { api } = setup();
    const result: BundleImportResult = {
      imported: true,
      counts: { documents: 2, pages: 10, lines: 80, finetuned_models: 1, abbreviations: 4 },
      schema_version: 4,
      migrated_from: 3,
      bundle_created_at: "",
      bundle_app_version: "0.3.0",
      files: 30,
      missing_files: [],
    };
    const imp = vi.spyOn(api, "importBundle").mockResolvedValue(result);
    const user = userEvent.setup();
    const button = screen.getByRole("button", { name: "Import project bundle" });
    expect(button).toBeDisabled();
    const file = new File(["zip"], "project.htrbundle");
    await user.upload(screen.getByLabelText("Bundle file"), file);
    await user.click(button);
    await waitFor(() => expect(imp).toHaveBeenCalledWith(file));
    const ok = await screen.findByTestId("import-ok");
    expect(ok).toHaveTextContent("Imported 2 documents, 10 pages, 80 lines, 1 trained models and 4 abbreviations");
    expect(ok).toHaveTextContent("upgraded from schema 3");
  });

  it("shows the refusal message for a non-empty project", async () => {
    const { api } = setup();
    vi.spyOn(api, "importBundle").mockRejectedValue(
      new ApiError(409, "the project is not empty (3 documents, 10 pages); a bundle can only be imported into an empty project (v1 never merges). Use a fresh data directory"),
    );
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText("Bundle file"), new File(["zip"], "p.htrbundle"));
    await user.click(screen.getByRole("button", { name: "Import project bundle" }));
    const err = await screen.findByTestId("import-error");
    expect(err).toHaveTextContent("Import refused: the project is not empty (3 documents, 10 pages)");
    expect(err).toHaveTextContent("v1 never merges");
    expect(screen.queryByTestId("import-ok")).not.toBeInTheDocument();
  });

  it("file names", () => {
    const base = { format: "txt" as const, expanded: false, ambiguous: "raw" as const, include_predicted: false };
    expect(exportFileName(base, null)).toBe("htr-all-raw.txt");
    expect(exportFileName({ ...base, expanded: true }, "Field notes!")).toBe("htr-Field-notes-expanded.txt");
    expect(exportFileName({ ...base, format: "pagexml" }, "x y")).toBe("htr-x-y-pagexml.zip");
  });
});
