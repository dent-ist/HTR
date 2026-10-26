import { useEffect, useState } from "react";
import type { ApiClient } from "../api/client";
import type { TrainingSummary } from "../api/types";

/** Small, non-blocking hint shown in the library and review pages when enough new ground truth has piled up. */
export function RetrainNotice({ api }: { api: ApiClient }) {
  const [summary, setSummary] = useState<TrainingSummary | null>(null);
  useEffect(() => {
    let live = true;
    api
      .getTrainingSummary()
      .then((s) => live && setSummary(s))
      .catch(() => undefined); // a hint only: never an error
    return () => {
      live = false;
    };
  }, [api]);
  if (!summary?.retrain_suggested) return null;
  return (
    <p className="notice retrain" role="status" data-testid="retrain-notice">
      <strong>Retraining suggested:</strong> {summary.retrain_reason}. <a href="#/training">Open Training</a> (you can also keep going without it).
    </p>
  );
}
