import type { Job, Model } from "../api/types";

/** 0.1234 -> "12.3%"; null/undefined -> "-" (nothing was held out, or no model yet). */
export function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || Number.isNaN(v) ? "-" : `${(v * 100).toFixed(digits)}%`;
}

export function dateLabel(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export const isActive = (j: Job | null): boolean => j !== null && (j.status === "queued" || j.status === "running");

export function modelLabel(m: Model): string {
  return m.name ?? `model ${m.id}`;
}

/** Fine-tuned models that have a measured CER, oldest first: the rounds of the comparison chart. */
export function chartRounds(models: Model[]): Model[] {
  return models
    .filter((m) => m.kind === "finetuned" && m.cer !== null)
    .sort((a, b) => (a.trained_at ?? "").localeCompare(b.trained_at ?? "") || a.id - b.id);
}

/** "Nice" y-axis maximum and tick values for a CER chart (fractions). */
export function yAxis(maxValue: number): { max: number; ticks: number[] } {
  const steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5];
  const target = Math.max(maxValue * 1.1, 0.05);
  const step = steps.find((s) => target / s <= 5) ?? 5;
  const max = Math.ceil(target / step) * step;
  const ticks: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(Math.round(v * 1000) / 1000);
  return { max, ticks };
}
