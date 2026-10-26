import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { Job, Model, PredictResult, ReviewStats, TrainResult, TrainingSummary } from "../api/types";
import { CerChart } from "./CerChart";
import { EffortCharts } from "./EffortCharts";
import { dateLabel, isActive, modelLabel, pct } from "./format";

const POLL_MS = 1000;
const DEFAULT_EPOCHS = "15";
const DEFAULT_BATCH = "4";

const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

function intInRange(text: string, lo: number, hi: number): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isInteger(n) && n >= lo && n <= hi ? n : null;
}

const PHASES: Record<string, string> = {
  exporting: "Preparing the training data...",
  "evaluating the start model": "Measuring the start model on the held-out lines...",
  "evaluating the new model": "Measuring the new model on the held-out lines...",
};

export function Training({ api }: { api: ApiClient }) {
  const [summary, setSummary] = useState<TrainingSummary | null>(null);
  const [models, setModels] = useState<Model[] | null>(null);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [trainJob, setTrainJob] = useState<Job | null>(null);
  const [predictJob, setPredictJob] = useState<Job | null>(null);
  const [trainNotes, setTrainNotes] = useState<string[]>([]);
  const [predictNotes, setPredictNotes] = useState<string[]>([]);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [batch, setBatch] = useState(DEFAULT_BATCH);

  const refresh = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([api.getTrainingSummary(), api.listModels()]);
      setSummary(s);
      setModels(m);
      setLoadError(null);
      api
        .reviewStats()
        .then(setStats)
        .catch(() => setStats(null)); // the effort charts are secondary: never an error banner
    } catch (e) {
      setLoadError(errText(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
    // Pick up a job that is already running (page reload, or started from elsewhere).
    for (const [type, set] of [
      ["train", setTrainJob],
      ["predict", setPredictJob],
    ] as const) {
      api
        .listJobs({ type, limit: 1 })
        .then(([latest]) => {
          if (latest && isActive(latest)) set(latest);
        })
        .catch(() => undefined);
    }
  }, [api, refresh]);

  // Poll active jobs; reload counts and models when one ends.
  const jobsRef = useRef({ trainJob, predictJob });
  jobsRef.current = { trainJob, predictJob };
  const anyActive = isActive(trainJob) || isActive(predictJob);
  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(async () => {
      let ended = false;
      for (const [job, set, setErr] of [
        [jobsRef.current.trainJob, setTrainJob, setTrainError],
        [jobsRef.current.predictJob, setPredictJob, setPredictError],
      ] as const) {
        if (!job || !isActive(job)) continue;
        try {
          const fresh = await api.getJob(job.id);
          set(fresh);
          if (!isActive(fresh)) ended = true;
          // a round: the finished train job names the predict job queued behind it
          const next = (fresh.result as TrainResult | null)?.next_job_id;
          if (fresh.type === "train" && fresh.status === "succeeded" && next !== undefined) {
            setPredictJob(await api.getJob(next));
            ended = true;
          }
        } catch (e) {
          setErr(errText(e));
        }
      }
      if (ended) void refresh();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [anyActive, api, refresh]);

  const epochsValue = intInRange(epochs, 1, 500);
  const batchValue = intInRange(batch, 1, 256);
  const noModel = summary !== null && summary.active_model === null;
  const canTrain = summary !== null && summary.ground_truth_lines > 0 && !noModel;
  // the number the recommended minimum applies to: all lines for the first round, new lines for a retrain
  const measured = summary ? (summary.last_round ? summary.new_since_last_round : summary.ground_truth_lines) : 0;

  const startTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    if (epochsValue === null || batchValue === null) return;
    setTrainError(null);
    setTrainNotes([]);
    try {
      const job = await api.startTraining({ epochs: epochsValue, batch_size: batchValue });
      setTrainJob(job);
      setTrainNotes(job.warnings);
    } catch (err) {
      setTrainError(errText(err));
    }
  };

  const startRound = async () => {
    if (epochsValue === null || batchValue === null) return;
    setTrainError(null);
    setTrainNotes([]);
    setPredictJob(null);
    try {
      const job = await api.startRound({ epochs: epochsValue, batch_size: batchValue });
      setTrainJob(job);
      setTrainNotes(job.warnings);
    } catch (err) {
      setTrainError(errText(err));
    }
  };

  const startPrediction = async () => {
    setPredictError(null);
    setPredictNotes([]);
    try {
      const job = await api.startPrediction({});
      setPredictJob(job);
      setPredictNotes(job.warnings);
    } catch (err) {
      setPredictError(errText(err));
    }
  };

  const cancel = async (job: Job, set: (j: Job) => void, setErr: (m: string | null) => void) => {
    try {
      set(await api.cancelJob(job.id));
    } catch (err) {
      setErr(errText(err));
    }
  };

  const activate = async (m: Model) => {
    setModelError(null);
    try {
      await api.activateModel(m.id);
      await refresh();
    } catch (err) {
      setModelError(errText(err));
    }
  };

  const trainRunning = isActive(trainJob);
  const predictRunning = isActive(predictJob);

  return (
    <main className="training">
      <header className="training-head">
        <h1>Training</h1>
        <a className="button" href="#/">
          Back to library
        </a>
      </header>

      {loadError && (
        <p role="alert" className="error banner">
          <span>{loadError}</span>
          <button onClick={() => void refresh()}>Retry</button>
        </p>
      )}
      {summary === null && !loadError && <p className="muted">Loading...</p>}

      {summary?.retrain_suggested && (
        <p className="notice retrain" role="status" data-testid="retrain-banner">
          <strong>Retraining suggested:</strong> {summary.retrain_reason}. This is advice only; you can train whenever you like.
        </p>
      )}
      {summary && (
        <section aria-labelledby="gt-heading" className="panel">
          <h2 id="gt-heading">Ground truth</h2>
          <div className="progress-cell">
            <progress
              value={Math.min(measured, summary.recommended)}
              max={summary.recommended}
              aria-label={`${measured} of ${summary.recommended} recommended ${summary.last_round ? "new " : ""}ground-truth lines`}
            />
            <span data-testid="gt-count">
              {summary.ground_truth_lines} ground-truth {summary.ground_truth_lines === 1 ? "line" : "lines"}
              {summary.last_round
                ? `; ${summary.new_since_last_round} new since the last round, about ${summary.recommended} recommended before retraining`
                : `, about ${summary.recommended} recommended for the first round`}
            </span>
          </div>
          <ul className="facts small muted">
            <li>
              Held out to measure accuracy: {summary.held_out_lines}
              {summary.unassigned_lines > 0 ? ` (plus a share of ${summary.unassigned_lines} lines not yet assigned)` : ""}. Held-out lines are chosen
              once and never trained on.
            </li>
            <li>
              New since the last training round: {summary.new_since_last_round}. Lines still to read: {summary.predictable_lines}.
            </li>
            <li>
              Active model:{" "}
              {summary.active_model ? (
                <strong>{summary.active_model.name ?? `model ${summary.active_model.id}`}</strong>
              ) : (
                <strong>none</strong>
              )}
              .
            </li>
          </ul>
          {noModel && (
            <p role="alert" className="error">
              No recognition model is installed. Run <code>make base-model</code> once (it downloads the pretrained model), then reload.
            </p>
          )}
          {summary.warnings.length > 0 && (
            <ul className="warnings" aria-label="Warnings (training is still allowed)">
              {summary.warnings.map((w) => (
                <li key={w}>
                  <strong>Note:</strong> {w}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section aria-labelledby="train-heading" className="panel">
        <h2 id="train-heading">Train</h2>
        <form onSubmit={startTraining} className="train-form">
          <label>
            Epochs
            <input
              type="number"
              min={1}
              max={500}
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
              aria-invalid={epochsValue === null}
              disabled={trainRunning}
            />
          </label>
          <label>
            Batch size
            <input
              type="number"
              min={1}
              max={256}
              value={batch}
              onChange={(e) => setBatch(e.target.value)}
              aria-invalid={batchValue === null}
              disabled={trainRunning}
            />
          </label>
          <button type="submit" disabled={!canTrain || trainRunning || predictRunning || epochsValue === null || batchValue === null}>
            Start training
          </button>
          <button
            type="button"
            onClick={() => void startRound()}
            disabled={!canTrain || trainRunning || predictRunning || epochsValue === null || batchValue === null}
            title="Train, then predict every line that is not ground truth with the new model"
          >
            Run round (train + predict)
          </button>
        </form>
        <p className="muted small">
          Training continues from the active model with your ground-truth lines. If the GPU runs out of memory, lower the batch size. Training stops early when
          it stops improving.
        </p>
        {(epochsValue === null || batchValue === null) && (
          <p className="error small" role="alert">
            Epochs must be 1-500 and batch size 1-256 (whole numbers).
          </p>
        )}
        {trainError && (
          <p role="alert" className="error">
            {trainError}
          </p>
        )}
        {trainNotes.length > 0 && (
          <ul className="warnings" aria-label="Warnings for this run">
            {trainNotes.map((w) => (
              <li key={w}>
                <strong>Note:</strong> {w}
              </li>
            ))}
          </ul>
        )}
        {trainJob?.payload.then_predict === true && <RoundStatus train={trainJob} predict={predictJob} />}
        {trainJob && <TrainStatus job={trainJob} onCancel={() => void cancel(trainJob, setTrainJob, setTrainError)} />}
      </section>

      <section aria-labelledby="predict-heading" className="panel">
        <h2 id="predict-heading">Predict</h2>
        <div className="actions">
          <button onClick={() => void startPrediction()} disabled={noModel || predictRunning || trainRunning || summary === null}>
            Predict remaining lines
          </button>
          <span className="muted small">
            Reads {summary ? summary.predictable_lines : "the"} lines that are not ground truth, ignored or typed by you, with the active model. Your own text is
            never overwritten.
          </span>
        </div>
        {predictError && (
          <p role="alert" className="error">
            {predictError}
          </p>
        )}
        {predictNotes.length > 0 && (
          <ul className="warnings" aria-label="Warnings for this prediction">
            {predictNotes.map((w) => (
              <li key={w}>
                <strong>Note:</strong> {w}
              </li>
            ))}
          </ul>
        )}
        {predictJob && <PredictStatus job={predictJob} onCancel={() => void cancel(predictJob, setPredictJob, setPredictError)} />}
      </section>

      <section aria-labelledby="chart-heading" className="panel">
        <h2 id="chart-heading">Accuracy by round</h2>
        {models && <CerChart models={models} />}
      </section>

      <section aria-labelledby="effort-heading" className="panel">
        <h2 id="effort-heading">Review effort by round</h2>
        {stats && <EffortCharts stats={stats} />}
        {!stats && <p className="muted">No review data yet.</p>}
      </section>

      <section aria-labelledby="models-heading" className="panel">
        <h2 id="models-heading">Models</h2>
        {modelError && (
          <p role="alert" className="error">
            {modelError}
          </p>
        )}
        {models && models.length === 0 && <p className="muted">No models yet.</p>}
        {models && models.length > 0 && (
          <table className="models">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Trained</th>
                <th scope="col">Started from</th>
                <th scope="col">Train lines</th>
                <th scope="col">Held-out lines</th>
                <th scope="col" title="Character error rate on the held-out lines; lower is better">
                  CER
                </th>
                <th scope="col" title="Word error rate on the held-out lines; lower is better">
                  WER
                </th>
                <th scope="col">Active</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} data-testid={`model-row-${m.id}`} className={m.is_active ? "active-model" : undefined}>
                  <th scope="row">
                    {modelLabel(m)}
                    {m.kind === "base" && <span className="muted small"> (pretrained)</span>}
                  </th>
                  <td>{m.kind === "base" ? "-" : dateLabel(m.trained_at)}</td>
                  <td>{m.base_model ?? "-"}</td>
                  <td>{m.train_lines ?? "-"}</td>
                  <td>{m.eval_lines ?? "-"}</td>
                  <td>
                    {pct(m.cer)}
                    {m.base_cer !== null && m.cer !== null && <span className="muted small"> (was {pct(m.base_cer)})</span>}
                  </td>
                  <td>{pct(m.wer)}</td>
                  <td>
                    {m.is_active ? (
                      <span className="badge">Active</span>
                    ) : (
                      <button onClick={() => void activate(m)} disabled={trainRunning || predictRunning} aria-label={`Activate ${modelLabel(m)}`}>
                        Activate
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted small">
          CER and WER are measured on the same fixed held-out lines for every round. On the very first lines they are noisy; they say nothing about pages
          written differently from the ones you transcribed.
        </p>
      </section>
    </main>
  );
}

function RoundStatus({ train, predict }: { train: Job; predict: Job | null }) {
  let text: string;
  if (isActive(train)) text = "Step 1 of 2: training";
  else if (train.status !== "succeeded") text = `Stopped: the training step ${train.status === "cancelled" ? "was cancelled" : "failed"}, so no prediction was started.`;
  else if (predict === null) text = "Training finished; the new model was not activated, so no prediction was started.";
  else if (isActive(predict)) text = "Step 2 of 2: predicting every line that is not ground truth";
  else if (predict.status === "succeeded") text = "Round finished: trained and predicted.";
  else text = `Training finished, but the prediction step ${predict.status === "cancelled" ? "was cancelled" : "failed"}.`;
  return (
    <p className="round-status" role="status" data-testid="round-status">
      <strong>Round:</strong> {text}
    </p>
  );
}

function TrainStatus({ job, onCancel }: { job: Job; onCancel(): void }) {
  const r = (job.result ?? {}) as TrainResult;
  const active = isActive(job);
  let line: string;
  if (job.status === "queued") line = "Waiting for its turn...";
  else if (job.status === "running" && r.phase === "training")
    line = `Epoch ${r.epoch ?? 1} of ${r.epochs ?? "?"}`;
  else if (job.status === "running") line = PHASES[r.phase ?? "exporting"] ?? "Working...";
  else if (job.status === "succeeded") line = "Training finished.";
  else if (job.status === "cancelled") line = "Training was cancelled. No model was saved.";
  else line = "Training failed.";
  const last = r.history?.[r.history.length - 1];
  return (
    <div className={`job ${job.status}`} role="status" data-testid="train-status">
      <div>
        <strong>{line}</strong> <span className="muted small">(job #{job.id})</span>
      </div>
      {active && <progress value={job.progress} max={1} aria-label="Training progress" />}
      {active && r.phase === "training" && (
        <div className="small" data-testid="train-metrics">
          Loss {r.loss !== null && r.loss !== undefined ? r.loss.toFixed(1) : "-"}; validation accuracy{" "}
          {last?.val_accuracy !== null && last?.val_accuracy !== undefined ? pct(last.val_accuracy) : "after the first epoch"}.
        </div>
      )}
      {active && (
        <button onClick={onCancel} disabled={job.cancel_requested}>
          {job.cancel_requested ? "Stopping..." : "Cancel"}
        </button>
      )}
      {job.status === "succeeded" && (
        <div data-testid="train-result">
          {r.cer !== null && r.cer !== undefined ? (
            <>
              Accuracy on {r.held_out_lines} held-out lines: CER <strong>{pct(r.base_cer)}</strong> before, <strong>{pct(r.cer)}</strong> after
              ({r.epochs_run ?? "?"} epochs, {Math.round(r.seconds ?? 0)} s).
            </>
          ) : (
            <>No lines were held out in this run, so accuracy was not measured.</>
          )}{" "}
          {r.activated === false ? "The new model is not better than its start model and was not activated." : "The new model is now active."}
          {r.warnings?.map((w) => (
            <div key={w} className="small muted">
              {w}
            </div>
          ))}
        </div>
      )}
      {job.status === "failed" && (
        <div className="error" role="alert">
          {job.error}
        </div>
      )}
    </div>
  );
}

function PredictStatus({ job, onCancel }: { job: Job; onCancel(): void }) {
  const r = (job.result ?? {}) as PredictResult;
  const active = isActive(job);
  let line: string;
  if (job.status === "queued") line = "Waiting for its turn...";
  else if (job.status === "running") line = `Reading lines... ${r.predicted ?? 0} of ${r.lines ?? "?"}`;
  else if (job.status === "succeeded") line = `Predicted ${r.predicted ?? 0} lines.`;
  else if (job.status === "cancelled") line = `Cancelled after ${r.predicted ?? 0} lines; those predictions were kept.`;
  else line = "Prediction failed.";
  return (
    <div className={`job ${job.status}`} role="status" data-testid="predict-status">
      <div>
        <strong>{line}</strong> <span className="muted small">(job #{job.id}{r.model ? `, ${r.model}` : ""})</span>
      </div>
      {active && <progress value={job.progress} max={1} aria-label="Prediction progress" />}
      {active && (
        <button onClick={onCancel} disabled={job.cancel_requested}>
          {job.cancel_requested ? "Stopping..." : "Cancel"}
        </button>
      )}
      {job.status === "succeeded" && r.mean_confidence !== null && r.mean_confidence !== undefined && (
        <div className="small muted">
          Average confidence {pct(r.mean_confidence, 0)}. Confidence ranks lines for review (lowest first); it is not a probability of being right.
        </div>
      )}
      {job.status === "failed" && (
        <div className="error" role="alert">
          {job.error}
        </div>
      )}
    </div>
  );
}
