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
  /** The model whose prediction this line still holds (null once reviewed or never predicted). */
  predicted_by_model_id?: number | null;
  /** What that model predicted (the snapshot review effort is measured against). */
  predicted_text?: string | null;
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
  /** Stable id of the failure (for example "file_missing") and what the user can do about it. */
  error_code?: string | null;
  error_hint?: string | null;
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
  /** Advice only (never blocks): enough new ground truth since the last round (or enough in total for a first one). */
  retrain_suggested: boolean;
  retrain_reason: string;
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
  /** true for POST /jobs/round: the train job's result.next_job_id names the chained predict job. */
  round?: boolean;
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
  /** Visible, non-fatal remarks, for example "running on the CPU because no GPU was found". */
  notices?: string[];
  next_job_id?: number;
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
  notices?: string[];
}

// ---- review queue and effort (M5) ----

export interface ReviewItem {
  line: Line;
  page_id: number;
  crop_url: string;
  predicted_text: string;
  confidence: number;
  model_id: number | null;
}

export interface ReviewNext {
  items: ReviewItem[];
  remaining: number;
}

export interface ReviewQueue {
  total: number;
  buckets: { min: number; max: number; count: number }[];
  mean_confidence: number | null;
  lowest?: ReviewItem[];
}

export interface ReviewRound {
  round: number;
  model_id: number | null;
  model_name: string | null;
  kind: string | null;
  trained_at: string | null;
  cer: number | null;
  wer: number | null;
  reviewed: number;
  accepted: number;
  corrected: number;
  ignored: number;
  accepted_rate: number | null;
  mean_edits_per_line: number | null;
  mean_edits_per_char: number | null;
}

export interface ReviewStats {
  rounds: ReviewRound[];
  total_reviewed: number;
  accepted_rate: number | null;
}

// ---- M6: abbreviations, export, project bundle ----

export interface AbbreviationCandidate {
  id: number;
  expansion: string;
}

export interface Abbreviation {
  id: number;
  written: string;
  expansion: string;
  notes: string | null;
  case_sensitive: boolean;
  /** Occurrences in the text of non-ignored lines (computed live). */
  count: number;
  /** Another row can match the same token with a different expansion. */
  ambiguous: boolean;
  /** All competing expansions (this row's own first) when ambiguous, else empty. */
  candidates: AbbreviationCandidate[];
  created_at: string | null;
  updated_at: string | null;
}

export interface AbbreviationInput {
  written: string;
  expansion: string;
  notes?: string | null;
  case_sensitive?: boolean;
}

export interface Occurrence {
  line_id: number;
  page_id: number;
  document_id: number;
  document_name: string;
  page_filename: string;
  line_index: number;
  text: string;
  text_source: "manual" | "predicted" | null;
  start: number;
  end: number;
  ambiguous: boolean;
}

export interface OccurrencePage {
  total: number;
  limit: number;
  offset: number;
  items: Occurrence[];
}

export type SpanStatus = "expanded" | "ambiguous" | "unmatched";

export interface ExpandSpan {
  start: number;
  end: number;
  written: string;
  expansion: string | null;
  status: SpanStatus;
  candidates: string[];
  entry_ids: number[];
}

export interface ExpandResult {
  text: string;
  spans: ExpandSpan[];
}

export interface PageTextLine {
  line_id: number;
  index: number;
  raw: string;
  expanded: string;
  text_source: "manual" | "predicted" | null;
  is_ground_truth: boolean;
  confidence: number | null;
  spans: ExpandSpan[];
}

export interface PageText {
  page_id: number;
  document_id: number;
  expanded: boolean;
  ambiguous: "raw" | "mark";
  include_predicted: boolean;
  lines: PageTextLine[];
  text: string;
}

export interface ExportOptions {
  format: "txt" | "pagexml";
  expanded: boolean;
  /** How ambiguous abbreviations appear in expanded text: as written, or marked `[?]`. */
  ambiguous: "raw" | "mark";
  document_id?: number;
  include_predicted: boolean;
}

export interface BundleImportResult {
  imported: boolean;
  counts: Record<string, number>;
  schema_version: number;
  migrated_from: number | null;
  bundle_created_at: string | null;
  bundle_app_version: string | null;
  files: number;
  missing_files: string[];
}

// ---- health and error format (M7) ----

export interface HealthCheck {
  id: string;
  label: string;
  status: "ok" | "warn" | "error";
  message: string;
  hint: string | null;
}

export interface HealthDetails {
  status: "ok" | "warning" | "error";
  version: string;
  checks: HealthCheck[];
}

// ---- optional local-LLM abbreviation suggestions (M7) ----

export interface Suggestion {
  id: number;
  written: string;
  expansion: string;
  rationale: string;
  line_id: number | null;
  /** The line the suggestion was made from. */
  context: string;
  model: string | null;
  status: "pending" | "accepted" | "rejected";
  abbreviation_id: number | null;
  created_at: string;
  resolved_at: string | null;
}

export interface SuggestionList {
  enabled: boolean;
  backend: { api: string; url: string; model: string } | null;
  items: Suggestion[];
}

export interface SuggestOptions {
  tokens?: string[];
  max_tokens?: number;
  context_lines?: number;
}

export interface SuggestionEdits {
  written?: string;
  expansion?: string;
  notes?: string;
  case_sensitive?: boolean;
}
