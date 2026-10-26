import type { Model } from "../api/types";
import { chartRounds, modelLabel, pct, yAxis } from "./format";

const W = 600;
const H = 240;
const M = { left: 52, right: 16, top: 16, bottom: 44 };

/**
 * CER by training round, as inline SVG (no chart library). Two series, told apart by line style and marker shape as
 * well as colour: the new model's CER on the held-out lines (solid, circles) and the CER its start model had on the
 * same lines (dashed, squares), so every round shows its own before and after. The model table is the text equivalent.
 */
export function CerChart({ models }: { models: Model[] }) {
  const rounds = chartRounds(models);
  if (rounds.length === 0) {
    return <p className="muted">No accuracy numbers yet: train a model with at least 8 ground-truth lines to measure CER.</p>;
  }
  const values = rounds.flatMap((m) => [m.cer ?? 0, m.base_cer ?? 0]);
  const { max, ticks } = yAxis(Math.max(...values));
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const x = (i: number) => M.left + (rounds.length === 1 ? plotW / 2 : (plotW * i) / (rounds.length - 1));
  const y = (v: number) => M.top + plotH - (v / max) * plotH;
  const after = rounds.map((m, i) => ({ m, x: x(i), y: y(m.cer ?? 0) }));
  const before = rounds.flatMap((m, i) => (m.base_cer === null ? [] : [{ m, x: x(i), y: y(m.base_cer) }]));
  const points = (ps: { x: number; y: number }[]) => ps.map((p) => `${p.x},${p.y}`).join(" ");
  const summary = rounds.map((m, i) => `round ${i + 1}: ${pct(m.cer)} (start model ${pct(m.base_cer)})`).join("; ");
  return (
    <figure className="cer-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Character error rate by training round" aria-describedby="cer-chart-desc" data-testid="cer-chart">
        <title id="cer-chart-title">Character error rate by training round</title>
        <desc id="cer-chart-desc">{summary}. Lower is better.</desc>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle" className="tick">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {rounds.map((m, i) => (
          <text key={m.id} x={x(i)} y={H - M.bottom + 18} textAnchor="middle" className="tick">
            Round {i + 1}
          </text>
        ))}
        <text x={M.left + plotW / 2} y={H - 6} textAnchor="middle" className="tick">
          CER on the held-out lines (lower is better)
        </text>
        {before.length > 1 && <polyline points={points(before)} className="series before" fill="none" />}
        {after.length > 1 && <polyline points={points(after)} className="series after" fill="none" />}
        {before.map((p) => (
          <rect key={`b${p.m.id}`} x={p.x - 4} y={p.y - 4} width={8} height={8} className="marker before" data-testid="cer-before">
            <title>{`${modelLabel(p.m)}: start model ${pct(p.m.base_cer)}`}</title>
          </rect>
        ))}
        {after.map((p) => (
          <g key={`a${p.m.id}`}>
            <circle cx={p.x} cy={p.y} r={5} className="marker after" data-testid="cer-after">
              <title>{`${modelLabel(p.m)}: ${pct(p.m.cer)} on ${p.m.eval_lines ?? 0} held-out lines`}</title>
            </circle>
            <text x={p.x} y={p.y - 10} textAnchor="middle" className="value">
              {pct(p.m.cer)}
            </text>
          </g>
        ))}
      </svg>
      <figcaption className="legend small">
        <span className="swatch after" aria-hidden="true" /> New model (solid, circles)
        <span className="swatch before" aria-hidden="true" /> Its start model on the same lines (dashed, squares)
      </figcaption>
    </figure>
  );
}
