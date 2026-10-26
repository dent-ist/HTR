import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiClient } from "../api/client";
import type { Job, Model, ReviewStats, TrainingSummary } from "../api/types";
import { Training } from "./Training";

const summary = (over: Partial<TrainingSummary> = {}): TrainingSummary => ({
  ground_truth_lines: 60,
  held_out_lines: 7,
  unassigned_lines: 0,
  projected_held_out_lines: 7,
  train_pool_lines: 53,
  new_since_last_round: 55,
  last_round: { model_id: 2, trained_at: "", train_lines: 40 },
  recommended_first_round: 100,
  recommended_per_retrain: 50,
  recommended: 50,
  predictable_lines: 100,
  active_model: { id: 2, name: "fine-tune 1", kind: "finetuned", cer: 0.1 },
  warnings: [],
  retrain_suggested: true,
  retrain_reason: "55 new ground-truth lines since the last round (about 50 suggested)",
  ...over,
});

const job = (over: Partial<Job>): Job => ({
  id: 5,
  type: "train",
  status: "queued",
  payload: { then_predict: true },
  progress: 0,
  error: null,
  result: null,
  created_at: "",
  started_at: null,
  finished_at: null,
  cancel_requested: false,
  ...over,
});

const stats: ReviewStats = {
  rounds: [
    { round: 1, model_id: 1, model_name: "base", kind: "base", trained_at: null, cer: null, wer: null, reviewed: 10, accepted: 2, corrected: 8, ignored: 0, accepted_rate: 0.2, mean_edits_per_line: 4, mean_edits_per_char: 0.2 },
    { round: 2, model_id: 2, model_name: "fine-tune 1", kind: "finetuned", trained_at: "", cer: 0.1, wer: 0.3, reviewed: 10, accepted: 6, corrected: 4, ignored: 0, accepted_rate: 0.6, mean_edits_per_line: 1, mean_edits_per_char: 0.05 },
  ],
  total_reviewed: 20,
  accepted_rate: 0.4,
};

function setup(s = summary()) {
  const api = new ApiClient("http://api.test");
  vi.spyOn(api, "getTrainingSummary").mockResolvedValue(s);
  vi.spyOn(api, "listModels").mockResolvedValue([{ id: 2, name: "fine-tune 1", kind: "finetuned", is_active: true, cer: 0.1, base_cer: 0.3, trained_at: "2026-01-01T00:00:00Z" } as Model]);
  vi.spyOn(api, "listJobs").mockResolvedValue([]);
  vi.spyOn(api, "reviewStats").mockResolvedValue(stats);
  return api;
}

describe("Training page: rounds and review effort", () => {
  it("shows the retrain-suggested banner as advice, never blocking", async () => {
    render(<Training api={setup()} />);
    expect(await screen.findByTestId("retrain-banner")).toHaveTextContent("55 new ground-truth lines");
    expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled();
  });

  it("has no banner when not suggested", async () => {
    render(<Training api={setup(summary({ retrain_suggested: false }))} />);
    await screen.findByTestId("gt-count");
    expect(screen.queryByTestId("retrain-banner")).toBeNull();
  });

  it("shows the review-effort charts and accepted-unchanged rates", async () => {
    render(<Training api={setup()} />);
    expect(await screen.findByTestId("effort-edits")).toBeInTheDocument();
    expect(screen.getByTestId("effort-accepted")).toHaveAccessibleDescription("Round 1: 20%; Round 2: 60%");
    expect(screen.getByTestId("effort-table")).toHaveTextContent("fine-tune 1");
    expect(screen.getByTestId("cer-chart")).toBeInTheDocument();
  });

  it("Run round starts a round, then follows the train job into its predict job", async () => {
    const api = setup();
    const start = vi.spyOn(api, "startRound").mockResolvedValue({ ...job({}), warnings: [], round: true });
    const get = vi.spyOn(api, "getJob").mockImplementation(async (id) =>
      id === 5
        ? job({ status: "succeeded", progress: 1, result: { phase: "done", cer: 0.05, base_cer: 0.1, held_out_lines: 7, epochs_run: 3, seconds: 9, activated: true, next_job_id: 6 } })
        : job({ id: 6, type: "predict", payload: {}, status: "running", progress: 0.4, result: { phase: "predicting", predicted: 40, lines: 100 } }),
    );
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    await userEvent.setup().click(screen.getByRole("button", { name: "Run round (train + predict)" }));
    await waitFor(() => expect(start).toHaveBeenCalledWith({ epochs: 15, batch_size: 4 }));
    expect(await screen.findByText("Reading lines... 40 of 100", undefined, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByTestId("round-status")).toHaveTextContent("Step 2 of 2: predicting");
    expect(get).toHaveBeenCalledWith(6);
    expect(screen.getByTestId("train-result")).toHaveTextContent("CER 10.0% before, 5.0% after");
  });

  it("a failed training step ends the round with a clear status", async () => {
    const api = setup();
    vi.spyOn(api, "startRound").mockResolvedValue({ ...job({}), warnings: [], round: true });
    vi.spyOn(api, "getJob").mockResolvedValue(job({ status: "failed", error: "the GPU ran out of memory at batch size 4. Retry with a smaller batch size" }));
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    await userEvent.setup().click(screen.getByRole("button", { name: "Run round (train + predict)" }));
    expect(await screen.findByText(/the training step failed, so no prediction was started/, undefined, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText(/smaller batch size/)).toBeInTheDocument();
  });

  it("a round can be cancelled while training", async () => {
    const api = setup();
    vi.spyOn(api, "listJobs").mockImplementation(async (f) => (f?.type === "train" ? [job({ status: "running", result: { phase: "training", epoch: 1, epochs: 5 } })] : []));
    const cancel = vi.spyOn(api, "cancelJob").mockResolvedValue(job({ status: "running", cancel_requested: true }));
    render(<Training api={api} />);
    expect(await screen.findByText("Step 1 of 2: training")).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "Cancel" }));
    expect(cancel).toHaveBeenCalledWith(5);
  });
});
