import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReviewRound, ReviewStats } from "../api/types";
import { EffortCharts, niceAxis } from "./EffortCharts";

const round = (n: number, over: Partial<ReviewRound> = {}): ReviewRound => ({
  round: n,
  model_id: n,
  model_name: `fine-tune ${n}`,
  kind: "finetuned",
  trained_at: "",
  cer: 0.1 / n,
  wer: 0.2 / n,
  reviewed: 20,
  accepted: 10 * n,
  corrected: 20 - 10 * n,
  ignored: 0,
  accepted_rate: (10 * n) / 20,
  mean_edits_per_line: 3 / n,
  mean_edits_per_char: 0.1 / n,
  ...over,
});

describe("EffortCharts", () => {
  it("draws both series per round and a table with the numbers behind them", () => {
    const stats: ReviewStats = { rounds: [round(1), round(2)], total_reviewed: 40, accepted_rate: 0.75 };
    render(<EffortCharts stats={stats} />);
    const edits = screen.getByRole("img", { name: /Mean edits per reviewed line/ });
    expect(edits).toHaveAccessibleDescription("Round 1: 3; Round 2: 1.5");
    expect(screen.getAllByTestId("effort-edits-point")).toHaveLength(2);
    expect(screen.getByRole("img", { name: /Accepted unchanged/ })).toHaveAccessibleDescription("Round 1: 50%; Round 2: 100%");
    const table = screen.getByTestId("effort-table");
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(table).toHaveTextContent("3.00");
    expect(table).toHaveTextContent("10.0%");
  });

  it("copes with one round, ignored-only rounds and no data", () => {
    const { rerender } = render(<EffortCharts stats={{ rounds: [round(1, { accepted_rate: null, mean_edits_per_line: null, mean_edits_per_char: null, ignored: 4, reviewed: 4 })], total_reviewed: 4, accepted_rate: null }} />);
    expect(screen.queryByTestId("effort-edits")).toBeNull();
    expect(screen.getByTestId("effort-table")).toHaveTextContent("4 (4 ignored)");
    rerender(<EffortCharts stats={{ rounds: [], total_reviewed: 0, accepted_rate: null }} />);
    expect(screen.getByText(/No reviews yet/)).toBeInTheDocument();
  });

  it("picks axes that contain the data", () => {
    for (const v of [0, 0.4, 1, 3, 7.5, 40]) {
      const { max, ticks } = niceAxis(v, 1);
      expect(max).toBeGreaterThanOrEqual(v);
      expect(ticks[0]).toBe(0);
      expect(ticks.length).toBeLessThanOrEqual(7);
    }
  });
});
