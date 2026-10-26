import { BAND_COLOR, BAND_LABEL, BAND_MARK, LEGEND, bandOf, confidencePct } from "./confidence";

/** Confidence as number + word + mark, tinted by band. Colour is an extra, never the only signal. */
export function ConfidenceBadge({ confidence }: { confidence: number }) {
  const band = bandOf(confidence);
  return (
    <span className={`conf conf-${band}`} style={{ color: BAND_COLOR[band] }} data-band={band} title={`Confidence ${confidencePct(confidence)} (${BAND_LABEL[band]})`}>
      <span aria-hidden="true">{BAND_MARK[band]}</span> {confidencePct(confidence)} {BAND_LABEL[band]}
    </span>
  );
}

export function ConfidenceLegend() {
  return (
    <ul className="conf-legend small muted" aria-label="Confidence legend">
      {LEGEND.map(({ band, range }) => (
        <li key={band}>
          <span style={{ color: BAND_COLOR[band] }} aria-hidden="true">
            {BAND_MARK[band]}
          </span>{" "}
          {BAND_LABEL[band]}: {range}
        </li>
      ))}
      <li>Outline: solid, dashed, dotted</li>
    </ul>
  );
}
