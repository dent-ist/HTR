import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Job, Suggestion, SuggestionList } from "../api/types";
import { Suggestions } from "./Suggestions";

const sugg = (id: number, written: string, expansion: string, over: Partial<Suggestion> = {}): Suggestion => ({
  id,
  written,
  expansion,
  rationale: "fits the sentence",
  line_id: 7,
  context: "pt c/o SOB since mon",
  model: "llama3.2:3b",
  status: "pending",
  abbreviation_id: null,
  created_at: "2026-10-26T10:00:00Z",
  resolved_at: null,
  ...over,
});

const job = (over: Partial<Job> = {}): Job => ({
  id: 5,
  type: "suggest",
  status: "queued",
  payload: {},
  progress: 0,
  error: null,
  result: null,
  created_at: "t",
  started_at: null,
  finished_at: null,
  ...over,
});

const ON = { api: "ollama", url: "http://127.0.0.1:11434", model: "llama3.2:3b" };

function setup(list: SuggestionList) {
  const api = new ApiClient("http://api.test");
  let current = list;
  vi.spyOn(api, "listSuggestions").mockImplementation(async () => current);
  const set = (l: SuggestionList) => {
    current = l;
  };
  const onAccepted = vi.fn();
  render(<Suggestions api={api} onAccepted={onAccepted} pollMs={20} />);
  return { api, set, onAccepted };
}

afterEach(() => vi.restoreAllMocks());

describe("Suggestions panel", () => {
  it("explains that the feature is off and offers no way to send anything", async () => {
    setup({ enabled: false, backend: null, items: [] });
    expect(await screen.findByTestId("llm-off")).toHaveTextContent("HTR_LLM_ENABLED=1");
    expect(screen.queryByRole("button", { name: "Suggest abbreviations" })).not.toBeInTheDocument();
  });

  it("states what is and is not done with the text", async () => {
    setup({ enabled: false, backend: null, items: [] });
    const region = await screen.findByRole("region", { name: /Suggestions from a local language model/ });
    expect(region).toHaveTextContent("never change your transcriptions");
    expect(region).toHaveTextContent("only when you accept it");
  });

  it("lists suggestions with rationale and the line they came from; accept adds to the dictionary", async () => {
    const { api, set, onAccepted } = setup({ enabled: true, backend: ON, items: [sugg(1, "c/o", "complains of")] });
    const row = await screen.findByTestId("suggestion-1");
    expect(row).toHaveTextContent("c/o");
    expect(row).toHaveTextContent("complains of");
    expect(row).toHaveTextContent("fits the sentence");
    expect(row).toHaveTextContent("pt c/o SOB since mon");
    const accept = vi.spyOn(api, "acceptSuggestion").mockResolvedValue({
      suggestion: sugg(1, "c/o", "complains of", { status: "accepted" }),
      abbreviation: {} as never,
    });
    set({ enabled: true, backend: ON, items: [] });
    await userEvent.click(within(row).getByRole("button", { name: "Accept c/o = complains of" }));
    expect(accept).toHaveBeenCalledWith(1);
    await waitFor(() => expect(screen.queryByTestId("suggestion-1")).not.toBeInTheDocument());
    expect(onAccepted).toHaveBeenCalledTimes(1);
    expect(screen.getByText("No suggestions waiting for a decision.")).toBeInTheDocument();
  });

  it("edit then accept sends the edited values", async () => {
    const { api } = setup({ enabled: true, backend: ON, items: [sugg(2, "prn", "pro re nata")] });
    await screen.findByTestId("suggestion-2");
    const accept = vi.spyOn(api, "acceptSuggestion").mockResolvedValue({ suggestion: sugg(2, "prn", "as needed"), abbreviation: {} as never });
    await userEvent.click(screen.getByRole("button", { name: "Edit suggestion prn = pro re nata" }));
    const form = screen.getByRole("form", { name: "Edit suggestion prn" });
    const expansion = within(form).getByLabelText("Expansion");
    await userEvent.clear(expansion);
    await userEvent.type(expansion, "as needed");
    await userEvent.click(within(form).getByRole("button", { name: "Accept as edited" }));
    expect(accept).toHaveBeenCalledWith(2, { written: "prn", expansion: "as needed" });
  });

  it("reject does not touch the dictionary", async () => {
    const { api, onAccepted } = setup({ enabled: true, backend: ON, items: [sugg(3, "BP", "blood pressure")] });
    const reject = vi.spyOn(api, "rejectSuggestion").mockResolvedValue(sugg(3, "BP", "blood pressure", { status: "rejected" }));
    await userEvent.click(await screen.findByRole("button", { name: "Reject BP = blood pressure" }));
    expect(reject).toHaveBeenCalledWith(3);
    expect(onAccepted).not.toHaveBeenCalled();
  });

  it("shows a refused decision and keeps the row", async () => {
    const { api } = setup({ enabled: true, backend: ON, items: [sugg(4, "x/y", "x over y")] });
    vi.spyOn(api, "acceptSuggestion").mockRejectedValue(new Error("suggestion 4 was already accepted"));
    await userEvent.click(await screen.findByRole("button", { name: "Accept x/y = x over y" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already accepted");
    expect(screen.getByTestId("suggestion-4")).toBeInTheDocument();
  });

  it("runs a job, shows progress, then the finished summary", async () => {
    const { api, set } = setup({ enabled: true, backend: ON, items: [] });
    const gen = vi.spyOn(api, "generateSuggestions").mockResolvedValue(job());
    const poll = vi.spyOn(api, "getJob");
    poll.mockResolvedValueOnce(job({ status: "running", progress: 0.5, result: { tokens: 4, asked: 2, suggestions_added: 3 } }));
    poll.mockResolvedValue(job({ status: "succeeded", progress: 1, result: { tokens: 4, asked: 4, suggestions_added: 6 } }));
    await userEvent.click(await screen.findByRole("button", { name: "Suggest abbreviations" }));
    expect(gen).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Asking the model\.\.\. 2 of 4/)).toBeInTheDocument();
    set({ enabled: true, backend: ON, items: [sugg(9, "f/u", "follow up")] });
    expect(await screen.findByText(/asked about 4 abbreviation\(s\), 6 new suggestion\(s\)/)).toBeInTheDocument();
    expect(await screen.findByTestId("suggestion-9")).toBeInTheDocument();
  });

  it("can cancel a running job", async () => {
    const { api } = setup({ enabled: true, backend: ON, items: [] });
    vi.spyOn(api, "generateSuggestions").mockResolvedValue(job({ status: "running" }));
    let cancelRequested = false; // the server reports the request from the moment it was made
    vi.spyOn(api, "getJob").mockImplementation(async () => job({ status: "running", cancel_requested: cancelRequested }));
    const cancel = vi.spyOn(api, "cancelJob").mockImplementation(async () => {
      cancelRequested = true;
      return job({ status: "running", cancel_requested: true });
    });
    await userEvent.click(await screen.findByRole("button", { name: "Suggest abbreviations" }));
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(cancel).toHaveBeenCalledWith(5);
    expect(await screen.findByRole("button", { name: "Stopping..." })).toBeDisabled();
  });

  it("shows why a job failed, with the hint", async () => {
    const { api } = setup({ enabled: true, backend: ON, items: [] });
    vi.spyOn(api, "generateSuggestions").mockResolvedValue(job({ status: "running" }));
    vi.spyOn(api, "getJob").mockResolvedValue(
      job({
        status: "failed",
        error: "Cannot reach the local LLM server at 127.0.0.1:11434.",
        error_code: "llm_unreachable",
        error_hint: "Start it (for Ollama: `ollama serve`).",
      }),
    );
    await userEvent.click(await screen.findByRole("button", { name: "Suggest abbreviations" }));
    const status = await screen.findByText(/Suggestions failed/);
    expect(status).toHaveTextContent("Cannot reach the local LLM server");
    expect(status).toHaveTextContent("ollama serve");
  });

  it("shows a start error such as 'already running' and a load error with Retry", async () => {
    const { api } = setup({ enabled: true, backend: ON, items: [] });
    vi.spyOn(api, "generateSuggestions").mockRejectedValue(new Error("a suggestion job is already running (job 3)"));
    await userEvent.click(await screen.findByRole("button", { name: "Suggest abbreviations" }));
    expect(await screen.findByTestId("suggest-error")).toHaveTextContent("already running");
  });
});
