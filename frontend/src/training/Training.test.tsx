import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";
import type { Job, Model, StartedJob, TrainingSummary } from "../api/types";
import { Training } from "./Training";
import { CerChart } from "./CerChart";
import { chartRounds, pct, yAxis } from "./format";

const summary = (over: Partial<TrainingSummary> = {}): TrainingSummary => ({
  ground_truth_lines: 24,
  held_out_lines: 3,
  unassigned_lines: 0,
  projected_held_out_lines: 3,
  train_pool_lines: 21,
  new_since_last_round: 24,
  last_round: null,
  recommended_first_round: 100,
  recommended_per_retrain: 50,
  recommended: 100,
  predictable_lines: 180,
  retrain_suggested: false,
  retrain_reason: "",
  active_model: { id: 1, name: "revcity-english-handwriting", kind: "base", cer: null },
  warnings: ["Only 24 ground-truth lines; about 100 are recommended for the first round."],
  ...over,
});

const model = (over: Partial<Model> = {}): Model => ({
  id: 1,
  name: "revcity-english-handwriting",
  kind: "base",
  engine: "kraken",
  path: "models/base/x.mlmodel",
  base_model: null,
  parent_model_id: null,
  trained_at: null,
  train_lines: null,
  eval_lines: null,
  cer: null,
  wer: null,
  base_cer: null,
  base_wer: null,
  hyperparameters: {},
  split_info: {},
  metrics: {},
  job_id: null,
  license: "Apache-2.0",
  is_active: true,
  ...over,
});

const tuned = (id: number, cer: number, baseCer: number, over: Partial<Model> = {}): Model =>
  model({
    id,
    name: `fine-tune ${id - 1}`,
    kind: "finetuned",
    base_model: "revcity-english-handwriting",
    trained_at: `2026-10-2${id}T10:00:00Z`,
    train_lines: 100 * (id - 1),
    eval_lines: 14,
    cer,
    wer: cer * 2,
    base_cer: baseCer,
    is_active: false,
    ...over,
  });

const job = (over: Partial<Job> = {}): Job => ({
  id: 5,
  type: "train",
  status: "queued",
  payload: {},
  progress: 0,
  error: null,
  result: null,
  created_at: "",
  started_at: null,
  finished_at: null,
  cancel_requested: false,
  ...over,
});

function setup(opts: { summary?: TrainingSummary; models?: Model[] } = {}) {
  const api = new ApiClient("http://api.test");
  const s = vi.spyOn(api, "getTrainingSummary").mockResolvedValue(opts.summary ?? summary());
  const m = vi.spyOn(api, "listModels").mockResolvedValue(opts.models ?? [model()]);
  vi.spyOn(api, "listJobs").mockResolvedValue([]);
  return { api, summaryMock: s, modelsMock: m };
}

describe("Training page", () => {
  it("shows the ground-truth count against the recommended minimum with a non-blocking warning", async () => {
    const { api } = setup();
    render(<Training api={api} />);
    expect(await screen.findByTestId("gt-count")).toHaveTextContent("24 ground-truth lines, about 100 recommended for the first round");
    const meter = screen.getByRole("progressbar", { name: "24 of 100 recommended ground-truth lines" });
    expect(meter).toHaveAttribute("max", "100");
    expect(meter).toHaveAttribute("value", "24");
    expect(screen.getByText(/Only 24 ground-truth lines; about 100 are recommended/)).toBeInTheDocument();
    // the warning does not block anything
    expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Predict remaining lines" })).toBeEnabled();
  });

  it("uses the retrain minimum after a first round", async () => {
    const { api } = setup({
      summary: summary({ recommended: 50, last_round: { model_id: 2, trained_at: "", train_lines: 20 }, new_since_last_round: 6 }),
    });
    render(<Training api={api} />);
    expect(await screen.findByTestId("gt-count")).toHaveTextContent(
      "24 ground-truth lines; 6 new since the last round, about 50 recommended before retraining",
    );
    const meter = screen.getByRole("progressbar", { name: "6 of 50 recommended new ground-truth lines" });
    expect(meter).toHaveAttribute("value", "6");
    expect(screen.getByText(/New since the last training round: 6/)).toBeInTheDocument();
  });

  it("starts training with the form values and shows the warnings returned by the API", async () => {
    const { api } = setup();
    const start = vi
      .spyOn(api, "startTraining")
      .mockResolvedValue({ ...job(), warnings: ["Only 24 ground-truth lines; run anyway."] } satisfies StartedJob);
    render(<Training api={api} />);
    const user = userEvent.setup();
    await screen.findByTestId("gt-count");
    await user.clear(screen.getByLabelText("Epochs"));
    await user.type(screen.getByLabelText("Epochs"), "5");
    await user.clear(screen.getByLabelText("Batch size"));
    await user.type(screen.getByLabelText("Batch size"), "2");
    await user.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(start).toHaveBeenCalledWith({ epochs: 5, batch_size: 2 }));
    expect(await screen.findByText("Waiting for its turn...")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Warnings for this run" })).toHaveTextContent("run anyway");
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled(); // one run at a time
  });

  it("rejects invalid numbers before calling the API", async () => {
    const { api } = setup();
    const start = vi.spyOn(api, "startTraining");
    render(<Training api={api} />);
    const user = userEvent.setup();
    await screen.findByTestId("gt-count");
    await user.clear(screen.getByLabelText("Epochs"));
    await user.type(screen.getByLabelText("Epochs"), "0");
    expect(screen.getByLabelText("Epochs")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByText(/Epochs must be 1-500/)).toBeInTheDocument();
    expect(start).not.toHaveBeenCalled();
  });

  it("shows live progress, loss and validation accuracy of a running job, and cancels it", async () => {
    const { api } = setup();
    vi.spyOn(api, "listJobs").mockImplementation((f) =>
      Promise.resolve(
        f?.type === "train"
          ? [
              job({
                status: "running",
                progress: 0.4,
                result: {
                  phase: "training",
                  epoch: 3,
                  epochs: 10,
                  loss: 123.4,
                  history: [{ epoch: 2, epochs: 10, train_loss: 150, val_accuracy: 0.9312, seconds: 12 }],
                },
              }),
            ]
          : [],
      ),
    );
    const cancel = vi.spyOn(api, "cancelJob").mockResolvedValue(job({ status: "running", cancel_requested: true }));
    render(<Training api={api} />);
    expect(await screen.findByText("Epoch 3 of 10")).toBeInTheDocument();
    expect(screen.getByTestId("train-metrics")).toHaveTextContent("Loss 123.4; validation accuracy 93.1%");
    expect(screen.getByRole("progressbar", { name: "Training progress" })).toHaveAttribute("value", "0.4");
    await userEvent.setup().click(screen.getByRole("button", { name: "Cancel" }));
    expect(cancel).toHaveBeenCalledWith(5);
    expect(await screen.findByRole("button", { name: "Stopping..." })).toBeDisabled();
  });

  it("polls a job to the end, then reloads the models and reports before/after CER", async () => {
    const { api, modelsMock } = setup();
    vi.spyOn(api, "startTraining").mockResolvedValue({ ...job(), warnings: [] });
    vi.spyOn(api, "getJob").mockResolvedValue(
      job({
        status: "succeeded",
        progress: 1,
        result: { phase: "done", cer: 0.04, base_cer: 0.29, held_out_lines: 17, epochs_run: 15, seconds: 190, activated: true, model_id: 2 },
      }),
    );
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    await userEvent.setup().click(screen.getByRole("button", { name: "Start training" }));
    const result = await screen.findByTestId("train-result", undefined, { timeout: 4000 });
    expect(result).toHaveTextContent("CER 29.0% before, 4.0% after");
    expect(result).toHaveTextContent("17 held-out lines");
    expect(result).toHaveTextContent("The new model is now active.");
    await waitFor(() => expect(modelsMock.mock.calls.length).toBeGreaterThan(1)); // refreshed after the job ended
  });

  it("shows a failed job's error, such as out of memory, as an alert", async () => {
    const { api } = setup();
    vi.spyOn(api, "startTraining").mockResolvedValue({ ...job(), warnings: [] });
    vi.spyOn(api, "getJob").mockResolvedValue(
      job({ status: "failed", error: "the GPU ran out of memory at batch size 8. Retry with a smaller batch size (for example 4)" }),
    );
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    await userEvent.setup().click(screen.getByRole("button", { name: "Start training" }));
    const status = await screen.findByTestId("train-status", undefined, { timeout: 4000 });
    await waitFor(() => expect(within(status).getByRole("alert")).toHaveTextContent("smaller batch size"));
  });

  it("shows API refusals (no ground truth) and disables training without ground truth", async () => {
    const { api } = setup({ summary: summary({ ground_truth_lines: 0, warnings: [] }) });
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();

    const { api: api2 } = setup();
    vi.spyOn(api2, "startTraining").mockRejectedValue(new ApiError(422, "there are no usable ground-truth lines to train on"));
    render(<Training api={api2} />);
    const buttons = await screen.findAllByRole("button", { name: "Start training" });
    await userEvent.setup().click(buttons[buttons.length - 1]!);
    expect(await screen.findByText(/no usable ground-truth lines/)).toBeInTheDocument();
  });

  it("asks for the base model when no model is installed", async () => {
    const { api } = setup({ summary: summary({ active_model: null }), models: [] });
    render(<Training api={api} />);
    expect(await screen.findByText(/make base-model/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Predict remaining lines" })).toBeDisabled();
  });

  it("lists models with CER, WER, the active marker, and activates another one", async () => {
    const models = [tuned(3, 0.06, 0.12, { is_active: true }), tuned(2, 0.12, 0.29), model()].map((m) => (m.id === 1 ? { ...m, is_active: false } : m));
    const { api, summaryMock } = setup({ models });
    const activate = vi.spyOn(api, "activateModel").mockResolvedValue(models[1]!);
    render(<Training api={api} />);
    const row3 = await screen.findByTestId("model-row-3");
    expect(within(row3).getByText("Active")).toBeInTheDocument();
    expect(row3).toHaveTextContent("6.0% (was 12.0%)");
    expect(row3).toHaveTextContent("12.0%"); // WER = 2 x CER in the fixture
    expect(row3).toHaveTextContent("14");
    expect(row3).toHaveTextContent("revcity-english-handwriting");
    const row1 = screen.getByTestId("model-row-1");
    expect(row1).toHaveTextContent("(pretrained)");
    await userEvent.setup().click(within(screen.getByTestId("model-row-2")).getByRole("button", { name: "Activate fine-tune 1" }));
    expect(activate).toHaveBeenCalledWith(2);
    await waitFor(() => expect(summaryMock.mock.calls.length).toBeGreaterThan(1));
  });

  it("predicts with progress and cancel, and keeps the predictions made so far", async () => {
    const { api } = setup();
    vi.spyOn(api, "startPrediction").mockResolvedValue({
      ...job({ id: 8, type: "predict" }),
      warnings: ["This is the pretrained base model, not yet adapted to your handwriting."],
    });
    vi.spyOn(api, "getJob").mockResolvedValue(
      job({ id: 8, type: "predict", status: "running", progress: 0.5, result: { phase: "predicting", predicted: 90, lines: 180, model: "fine-tune 1" } }),
    );
    const cancel = vi.spyOn(api, "cancelJob").mockResolvedValue(job({ id: 8, type: "predict", status: "running", cancel_requested: true }));
    render(<Training api={api} />);
    const user = userEvent.setup();
    await screen.findByTestId("gt-count");
    await user.click(screen.getByRole("button", { name: "Predict remaining lines" }));
    expect(await screen.findByRole("list", { name: "Warnings for this prediction" })).toHaveTextContent("pretrained base model");
    expect(await screen.findByText("Reading lines... 90 of 180", undefined, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Prediction progress" })).toHaveAttribute("value", "0.5");
    await user.click(within(screen.getByTestId("predict-status")).getByRole("button", { name: "Cancel" }));
    expect(cancel).toHaveBeenCalledWith(8);
  });

  it("reports a finished prediction", async () => {
    const { api } = setup();
    vi.spyOn(api, "startPrediction").mockResolvedValue({ ...job({ id: 8, type: "predict" }), warnings: [] });
    vi.spyOn(api, "getJob").mockResolvedValue(
      job({ id: 8, type: "predict", status: "succeeded", progress: 1, result: { phase: "done", predicted: 180, lines: 180, mean_confidence: 0.882 } }),
    );
    render(<Training api={api} />);
    await screen.findByTestId("gt-count");
    await userEvent.setup().click(screen.getByRole("button", { name: "Predict remaining lines" }));
    expect(await screen.findByText("Predicted 180 lines.", undefined, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByTestId("predict-status")).toHaveTextContent("Average confidence 88%");
  });

  it("shows a load error with a retry", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "getTrainingSummary").mockRejectedValue(new Error("Cannot reach the API"));
    vi.spyOn(api, "listModels").mockResolvedValue([]);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);
    render(<Training api={api} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the API");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("links back to the library", async () => {
    const { api } = setup();
    render(<Training api={api} />);
    expect(await screen.findByRole("link", { name: "Back to library" })).toHaveAttribute("href", "#/");
  });
});

describe("CER chart", () => {
  it("draws one point per round for both series, with an accessible description", () => {
    render(<CerChart models={[model(), tuned(3, 0.06, 0.12), tuned(2, 0.12, 0.29)]} />);
    const svg = screen.getByRole("img", { name: "Character error rate by training round" });
    expect(svg).toHaveAccessibleDescription(/round 1: 12\.0% \(start model 29\.0%\); round 2: 6\.0% \(start model 12\.0%\)/);
    expect(screen.getAllByTestId("cer-after")).toHaveLength(2);
    expect(screen.getAllByTestId("cer-before")).toHaveLength(2);
    expect(screen.getByText("Round 1")).toBeInTheDocument();
    expect(screen.getByText("Round 2")).toBeInTheDocument();
    expect(svg.querySelectorAll("polyline")).toHaveLength(2);
  });

  it("copes with a single round and with no measured CER", () => {
    const { rerender } = render(<CerChart models={[tuned(2, 0.2, 0.3)]} />);
    expect(screen.getAllByTestId("cer-after")).toHaveLength(1);
    rerender(<CerChart models={[model(), tuned(2, 0.2, 0.3, { cer: null, base_cer: null })]} />);
    expect(screen.queryByTestId("cer-chart")).toBeNull();
    expect(screen.getByText(/No accuracy numbers yet/)).toBeInTheDocument();
  });
});

describe("training format helpers", () => {
  it("formats percentages", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(0.5, 0)).toBe("50%");
    expect(pct(null)).toBe("-");
    expect(pct(undefined)).toBe("-");
  });

  it("orders chart rounds by training time and drops base / unmeasured models", () => {
    const rounds = chartRounds([tuned(3, 0.1, 0.2), model(), tuned(2, 0.2, 0.3), tuned(4, 0.1, 0.2, { cer: null })]);
    expect(rounds.map((m) => m.id)).toEqual([2, 3]);
  });

  it("picks a y axis that contains the data", () => {
    for (const v of [0.01, 0.07, 0.29, 0.5, 1.3]) {
      const { max, ticks } = yAxis(v);
      expect(max).toBeGreaterThanOrEqual(v);
      expect(ticks[0]).toBe(0);
      expect(ticks[ticks.length - 1]).toBeCloseTo(max);
      expect(ticks.length).toBeLessThanOrEqual(7);
    }
  });
});
