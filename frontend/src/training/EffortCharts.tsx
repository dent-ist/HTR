import type { ReviewStats } from "../api/types";
import { pct } from "./format";

const W = 420;
const H = 220;
const M = { left: 44, right: 34, top: 18, bottom: 40 };

/** A "nice" axis maximum and ticks for values from 0 up to `max`. */
export function niceAxis(max: number, floor: number): { max: number; ticks: number[] } {
  const target = Math.max(max * 1.1, floor);
  const mag = 10 ** Math.floor(Math.log10(target));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => target / s <= 5) ?? 10 * mag;
  const top = Math.ceil(target / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let v = 0; v <= top + 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return { max: top, ticks };
}

interface Point {
  label: string;
  value: number;
}

function LineChart({ title, id, points, format, floor }: { title: string; id: string; points: Point[]; format(v: number): string; floor: number }) {
  if (points.length === 0) return null;
  const { max, ticks } = niceAxis(Math.max(...points.map((p) => p.value)), floor);
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const x = (i: number) => M.left + (points.length === 1 ? plotW / 2 : (plotW * i) / (points.length - 1));
  const y = (v: number) => M.top + plotH - (v / max) * plotH;
  const desc = points.map((p) => `${p.label}: ${format(p.value)}`).join("; ");
  return (
    <figure className="cer-chart effort-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} aria-describedby={`${id}-desc`} data-testid={id}>
        <title>{title}</title>
        <desc id={`${id}-desc`}>{desc}</desc>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle" className="tick">
              {format(t)}
            </text>
          </g>
        ))}
        {points.length > 1 && <polyline points={points.map((p, i) => `${x(i)},${y(p.value)}`).join(" ")} className="series after" fill="none" />}
        {points.map((p, i) => (
          <g key={p.label}>
            <circle cx={x(i)} cy={y(p.value)} r={5} className="marker after" data-testid={`${id}-point`}>
              <title>{`${p.label}: ${format(p.value)}`}</title>
            </circle>
            <text x={x(i)} y={y(p.value) - 10} textAnchor="middle" className="value">
              {format(p.value)}
            </text>
            <text x={x(i)} y={H - M.bottom + 18} textAnchor="middle" className="tick">
              {p.label}
            </text>
          </g>
        ))}
      </svg>
      <figcaption className="small muted">{title}</figcaption>
    </figure>
  );
}

/** Review effort across rounds: does correcting predictions get cheaper? Two small charts and the numbers behind them. */
export function EffortCharts({ stats }: { stats: ReviewStats }) {
  const rounds = stats.rounds;
  if (rounds.length === 0) {
    return <p className="muted">No reviews yet. Accept or correct predicted lines (Review page or editor) and the effort per round shows up here.</p>;
  }
  const label = (r: (typeof rounds)[number]) => `Round ${r.round}`;
  const edits = rounds.filter((r) => r.mean_edits_per_line !== null).map((r) => ({ label: label(r), value: r.mean_edits_per_line! }));
  const rate = rounds.filter((r) => r.accepted_rate !== null).map((r) => ({ label: label(r), value: r.accepted_rate! }));
  return (
    <div className="effort">
      <div className="effort-grid">
        <LineChart title="Mean edits per reviewed line (lower is better)" id="effort-edits" points={edits} format={(v) => (Math.round(v * 10) / 10).toString()} floor={1} />
        <LineChart title="Accepted unchanged (higher is better)" id="effort-accepted" points={rate} format={(v) => `${Math.round(v * 100)}%`} floor={1} />
      </div>
      <table className="models" data-testid="effort-table">
        <caption className="small muted">Review effort by round (a round is the model whose predictions were reviewed)</caption>
        <thead>
          <tr>
            <th scope="col">Round</th>
            <th scope="col">Model</th>
            <th scope="col">Reviewed</th>
            <th scope="col">Accepted unchanged</th>
            <th scope="col">Edits / line</th>
            <th scope="col">Edits / character</th>
            <th scope="col">Held-out CER</th>
          </tr>
        </thead>
        <tbody>
          {rounds.map((r) => (
            <tr key={r.round}>
              <th scope="row">{r.round}</th>
              <td>{r.model_name ?? "-"}</td>
              <td>
                {r.reviewed}
                {r.ignored > 0 ? ` (${r.ignored} ignored)` : ""}
              </td>
              <td>{pct(r.accepted_rate, 0)}</td>
              <td>{r.mean_edits_per_line === null ? "-" : r.mean_edits_per_line.toFixed(2)}</td>
              <td>{r.mean_edits_per_char === null ? "-" : r.mean_edits_per_char.toFixed(3)}</td>
              <td>{pct(r.cer)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
