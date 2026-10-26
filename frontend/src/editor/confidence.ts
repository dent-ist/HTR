// Confidence bands for predicted lines. Never colour alone: every band also has a word, a canvas dash pattern and an
// icon-like prefix, so the indicator survives colour blindness and monochrome printing.
import type { Line } from "../api/types";

export type Band = "high" | "medium" | "low";

export const BAND_MIN: Record<Band, number> = { high: 0.9, medium: 0.7, low: 0 };

export function bandOf(confidence: number): Band {
  return confidence >= BAND_MIN.high ? "high" : confidence >= BAND_MIN.medium ? "medium" : "low";
}

export const BAND_COLOR: Record<Band, string> = { high: "#1c7a8a", medium: "#b7791f", low: "#c62828" };
/** Canvas outline dash per band: solid, dashed, dotted. */
export const BAND_DASH: Record<Band, number[]> = { high: [], medium: [8, 4], low: [2, 3] };
export const BAND_MARK: Record<Band, string> = { high: "●", medium: "◐", low: "○" }; // full, half, empty
export const BAND_LABEL: Record<Band, string> = { high: "high", medium: "medium", low: "low" };

/** A line that holds a model prediction nobody has reviewed yet. */
export const isPrediction = (l: Line): boolean =>
  l.text_source === "predicted" && !l.is_ground_truth && !l.ignored && l.confidence !== null;

export const confidencePct = (c: number): string => `${Math.round(c * 100)}%`;

export const LEGEND: { band: Band; range: string }[] = [
  { band: "high", range: "90% and up" },
  { band: "medium", range: "70 to 89%" },
  { band: "low", range: "under 70%" },
];
