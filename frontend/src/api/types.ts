// Types mirror docs/api.md. Coordinates are working-copy pixels.
export type Point = [number, number];

export type PageStatus = "unsegmented" | "segmented" | "partially_transcribed" | "done";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Document {
  id: number;
  name: string;
  imported_at: string;
  page_count: number;
}

export interface Line {
  id: number;
  page_id: number;
  index: number;
  baseline: Point[];
  polygon: Point[];
  text: string | null;
  text_source: "manual" | "predicted" | null;
  confidence: number | null;
  is_ground_truth: boolean;
  ignored: boolean;
  crop_url: string;
}

export interface Page {
  id: number;
  document_id: number;
  original_filename: string;
  status: PageStatus;
  width: number;
  height: number;
  warnings: string[];
  imported_at: string;
  segmented_at: string | null;
  image_url: string;
  original_url: string;
  line_count: number;
  ground_truth_count: number;
  /** Ground-truth lines that are not ignored (use for progress). */
  active_ground_truth_count: number;
  ignored_count: number;
}

export interface PageDetail extends Page {
  lines: Line[];
}

export interface UploadResult extends Page {
  duplicate: boolean;
}

export interface Job {
  id: number;
  type: string;
  status: JobStatus;
  payload: Record<string, unknown>;
  progress: number;
  error: string | null;
  cancel_requested?: boolean;
  result: unknown;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

/** Fields accepted by PATCH /lines/{id}. */
export interface LinePatch {
  text?: string;
  ignored?: boolean;
  is_ground_truth?: boolean;
  index?: number;
  polygon?: Point[];
  baseline?: Point[];
  /** With a baseline: let the server rebuild the polygon from it (redraw baseline). */
  regenerate_polygon?: boolean;
}

export interface MergeResult {
  line: Line;
  removed_ids: number[];
  page: PageDetail;
}

export interface SplitResult {
  lines: [Line, Line];
  /** Text of the line before it was cut (both halves start empty). */
  old_text: string;
  page: PageDetail;
}

export interface IgnoreRegionResult {
  affected: number[];
  page: PageDetail;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ---- training, prediction, models (M4) ----

export interface Model {
  id: number;
  name: string | null;
  /** "base" is a pretrained model downloaded with `make base-model`; "finetuned" comes from a train job. */
  kind: "base" | "finetuned";
  engine: string;
  path: string;
  base_model: string | null;
  parent_model_id: number | null;
  trained_at: string | null;
  train_lines: number | null;
  eval_lines: number | null;
  /** Fractions (0.12 = 12%), measured on the held-out lines; null when nothing was held out. */
  cer: number | null;
  wer: number | null;
  /** The start model's CER/WER on the same held-out lines. */
  base_cer: number | null;
  base_wer: number | null;
  hyperparameters: Record<string, unknown>;
  split_info: Record<string, unknown>;
  metrics: Record<string, unknown>;
  job_id: number | null;
  license: string | null;
  is_active: boolean;
}

export interface TrainingSummary {
  ground_truth_lines: number;
  held_out_lines: number;
  unassigned_lines: number;
  projected_held_out_lines: number;
  train_pool_lines: number;
  new_since_last_round: number;
  last_round: { model_id: number; trained_at: string | null; train_lines: number | null } | null;
  recommended_first_round: number;
  recommended_per_retrain: number;
  /** The recommended minimum that applies now (first round, or new lines per retrain). */
  recommended: number;
  predictable_lines: number;
  active_model: { id: number; name: string | null; kind: string; cer: number | null } | null;
  warnings: string[];
}

export interface TrainOptions {
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  base_model_id?: number;
  device?: "auto" | "cpu" | "cuda:0";
}

export interface PredictOptions {
  model_id?: number;
  page_ids?: number[];
}

/** A job just enqueued by POST /jobs/train or /jobs/predict, with non-blocking warnings. */
export interface StartedJob extends Job {
  warnings: string[];
}

export interface EpochRecord {
  epoch: number;
  epochs: number;
  train_loss: number | null;
  val_accuracy: number | null;
  seconds: number;
}

/** `result` of a train job, live while running and final when done (all fields optional: it grows as the job runs). */
export interface TrainResult {
  phase?: string;
  start_model?: string;
  train_lines?: number;
  validation_lines?: number;
  held_out_lines?: number;
  epoch?: number;
  epochs?: number;
  loss?: number | null;
  val_accuracy?: number | null;
  history?: EpochRecord[];
  model_id?: number;
  activated?: boolean;
  epochs_run?: number;
  best_epoch?: number;
  cer?: number | null;
  wer?: number | null;
  base_cer?: number | null;
  device?: string;
  peak_vram_mb?: number | null;
  seconds?: number;
  warnings?: string[];
}

/** `result` of a predict job. */
export interface PredictResult {
  phase?: string;
  model?: string;
  lines?: number;
  predicted?: number;
  unreadable?: number;
  changed_meanwhile?: number;
  mean_confidence?: number | null;
  device?: string | null;
  cuda_fallback?: boolean;
}
