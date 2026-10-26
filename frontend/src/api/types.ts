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
}
