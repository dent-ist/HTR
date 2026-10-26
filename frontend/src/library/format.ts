import type { Job, Page, PageStatus } from "../api/types";

const LABELS: Record<PageStatus, string> = {
  unsegmented: "Unsegmented",
  segmented: "Segmented",
  partially_transcribed: "Partially transcribed",
  done: "Done",
};

export const statusLabel = (s: PageStatus): string => LABELS[s] ?? s;

/** A failed job's message followed by what the user can do about it. */
export const jobProblem = (j: Pick<Job, "error" | "error_hint">): string =>
  [j.error, j.error_hint].filter(Boolean).join(" ");

export const jobIsActive = (j: Job): boolean => j.status === "queued" || j.status === "running";

export function jobLabel(j: Job): string {
  if (j.status === "queued") return "Queued for segmentation...";
  if (j.status === "running") return `Segmenting... ${Math.round(j.progress * 100)}%`;
  if (j.status === "failed") return "Failed";
  if (j.status === "cancelled") return "Cancelled";
  return "Done";
}

/** Active (non-ignored) lines and how many of them are ground truth. */
export function progressOf(p: Pick<Page, "line_count" | "ignored_count" | "active_ground_truth_count">): { done: number; total: number } {
  return { done: p.active_ground_truth_count, total: Math.max(0, p.line_count - p.ignored_count) };
}
