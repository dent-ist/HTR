import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Document, Job, Page, UploadResult } from "../api/types";
import { Library } from "./Library";

const doc: Document = { id: 1, name: "Notebook", imported_at: "", page_count: 1 };
const page: Page = {
  id: 7,
  document_id: 1,
  original_filename: "scan1.png",
  status: "unsegmented",
  width: 100,
  height: 100,
  warnings: [],
  imported_at: "",
  segmented_at: null,
  image_url: "/pages/7/working",
  original_url: "/pages/7/original",
  line_count: 0,
  ground_truth_count: 0,
  ignored_count: 0,
};
const job = (over: Partial<Job>): Job => ({
  id: 3,
  type: "segment",
  status: "queued",
  payload: { page_id: 7 },
  progress: 0,
  error: null,
  result: null,
  created_at: "",
  started_at: null,
  finished_at: null,
  ...over,
});

function setup(pages: Page[] = [page]) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "listDocuments").mockResolvedValue([doc]);
  vi.spyOn(api, "listPages").mockResolvedValue(pages);
  vi.spyOn(api, "listJobs").mockResolvedValue([]);
  return api;
}

describe("Library", () => {
  it("lists documents, pages, status and progress", async () => {
    const api = setup([
      { ...page, status: "partially_transcribed", line_count: 10, ground_truth_count: 4, ignored_count: 2 },
      { ...page, id: 8, original_filename: "scan2.png", status: "done", line_count: 3, ground_truth_count: 3 },
    ]);
    render(<Library api={api} />);
    expect(await screen.findByText("Notebook")).toBeInTheDocument();
    const row = screen.getByTestId("page-row-7");
    expect(within(row).getByText("Partially transcribed")).toBeInTheDocument();
    expect(within(row).getByText(/4\/8 lines, 2 ignored/)).toBeInTheDocument();
    expect(within(screen.getByTestId("page-row-8")).getByText("Done")).toBeInTheDocument();
  });

  it("uploads files and flags duplicates clearly", async () => {
    const api = setup();
    const dup: UploadResult = { ...page, duplicate: true };
    vi.spyOn(api, "uploadPage").mockResolvedValue(dup);
    render(<Library api={api} />);
    const input = await screen.findByTestId("file-input-1");
    await userEvent.setup().upload(input, new File(["x"], "again.png", { type: "image/png" }));
    expect(await screen.findByText(/Already imported \(identical file\)/)).toBeInTheDocument();
  });

  it("segment enqueues a job, polls it and shows failure details", async () => {
    const api = setup();
    vi.spyOn(api, "segmentPage").mockResolvedValue(job({}));
    vi.spyOn(api, "getJob").mockResolvedValue(job({ status: "failed", error: "model exploded" }));
    render(<Library api={api} />);
    await userEvent.setup().click(await screen.findByRole("button", { name: /Segment scan1\.png/ }));
    expect(await screen.findByText(/Queued for segmentation/)).toBeInTheDocument();
    expect(await screen.findByText(/Segmentation failed: model exploded/, undefined, { timeout: 3000 })).toBeInTheDocument();
  });

  it("creates a document", async () => {
    const api = setup();
    const create = vi.spyOn(api, "createDocument").mockResolvedValue({ ...doc, id: 2, name: "New" });
    render(<Library api={api} />);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("New document name"), "New");
    await user.click(screen.getByRole("button", { name: "Create document" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("New"));
  });

  it("shows an error when the API is unreachable", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "listDocuments").mockRejectedValue(new Error("Cannot reach the API"));
    vi.spyOn(api, "listPages").mockResolvedValue([]);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);
    render(<Library api={api} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the API");
  });
});
