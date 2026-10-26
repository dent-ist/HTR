import { describe, expect, it } from "vitest";
import type { Line } from "../api/types";
import { BAND_DASH, bandOf, confidencePct, isPrediction } from "./confidence";

const line = (over: Partial<Line>): Line => ({
  id: 1,
  page_id: 1,
  index: 0,
  baseline: [],
  polygon: [],
  text: "x",
  text_source: "predicted",
  confidence: 0.8,
  is_ground_truth: false,
  ignored: false,
  crop_url: "",
  ...over,
});

describe("confidence bands", () => {
  it("splits at 70% and 90%", () => {
    expect(bandOf(0.95)).toBe("high");
    expect(bandOf(0.9)).toBe("high");
    expect(bandOf(0.89)).toBe("medium");
    expect(bandOf(0.7)).toBe("medium");
    expect(bandOf(0.69)).toBe("low");
    expect(bandOf(0)).toBe("low");
    expect(confidencePct(0.876)).toBe("88%");
  });

  it("is not colour only: every band has its own outline pattern", () => {
    const patterns = new Set(Object.values(BAND_DASH).map((d) => d.join(",")));
    expect(patterns.size).toBe(3);
  });

  it("only unreviewed predictions count", () => {
    expect(isPrediction(line({}))).toBe(true);
    expect(isPrediction(line({ is_ground_truth: true }))).toBe(false);
    expect(isPrediction(line({ ignored: true }))).toBe(false);
    expect(isPrediction(line({ text_source: "manual", confidence: null }))).toBe(false);
    expect(isPrediction(line({ confidence: null }))).toBe(false);
  });
});
