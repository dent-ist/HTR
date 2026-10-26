import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { HealthDetails } from "../api/types";
import { useToasts } from "./Toasts";

type View = "loading" | "ok" | "warning" | "error" | "offline";

const LABEL: Record<View, string> = {
  loading: "Checking...",
  ok: "Ready",
  warning: "Ready, with notes",
  error: "Needs attention",
  offline: "Server offline",
};

export interface StatusBarProps {
  api: ApiClient;
  /** How often to re-check while the server answers (ms) and while it does not. */
  pollMs?: number;
  retryMs?: number;
}

/**
 * The strip at the top of every screen: app name, a status indicator fed by GET /health/details (database, disk,
 * GPU, base model, worker) and, when the local server cannot be reached, a banner with a Retry button. Server errors
 * (5xx) and lost connections from any screen are reported once as a toast.
 */
export function StatusBar({ api, pollMs = 30_000, retryMs = 4_000 }: StatusBarProps) {
  const [view, setView] = useState<View>("loading");
  const [details, setDetails] = useState<HealthDetails | null>(null);
  const [open, setOpen] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const toasts = useToasts();
  const viewRef = useRef<View>("loading");
  viewRef.current = view;

  const check = useCallback(async () => {
    setRetrying(true);
    try {
      const d = await api.getHealthDetails();
      setDetails(d);
      setView(d.status);
    } catch {
      setView("offline");
    } finally {
      setRetrying(false);
    }
  }, [api]);

  useEffect(() => {
    void check();
  }, [check]);

  useEffect(() => {
    const timer = setInterval(() => void check(), view === "offline" ? retryMs : pollMs);
    return () => clearInterval(timer);
  }, [check, view, pollMs, retryMs]);

  useEffect(
    () =>
      api.subscribe((e) => {
        if (e.type === "network") {
          if (viewRef.current !== "offline") toasts.error(e.message);
          setView("offline");
        } else if (e.type === "server") {
          toasts.error(e.message);
        } else if (e.type === "ok" && viewRef.current === "offline") {
          void check();
        }
      }),
    [api, check, toasts],
  );

  const problems = details?.checks.filter((c) => c.status !== "ok") ?? [];

  return (
    <>
      <header className="statusbar">
        <a className="brand" href="#/">
          HTR
        </a>
        <button
          className={`status-pill status-${view}`}
          aria-expanded={open}
          aria-controls="health-panel"
          onClick={() => setOpen((o) => !o)}
          data-testid="status-indicator"
        >
          <span className="dot" aria-hidden="true" />
          {LABEL[view]}
          {view !== "offline" && problems.length > 0 && <span className="count"> ({problems.length})</span>}
        </button>
      </header>
      {view === "offline" && (
        <p role="alert" className="error banner offline-banner" data-testid="offline-banner">
          <span>
            The app cannot reach its local server. Is it running? Start it with <code>make dev</code> (or{" "}
            <code>make serve</code>) in the project folder. Nothing you saved is lost.
          </span>
          <button onClick={() => void check()} disabled={retrying}>
            {retrying ? "Retrying..." : "Retry"}
          </button>
        </p>
      )}
      {open && details && view !== "offline" && (
        <section id="health-panel" className="panel health-panel" aria-label="System status">
          <ul>
            {details.checks.map((c) => (
              <li key={c.id} className={`check-${c.status}`}>
                <strong>{c.label}</strong>: {c.message}
                {c.hint && <span className="muted"> {c.hint}</span>}
              </li>
            ))}
          </ul>
          <p className="muted small">
            Version {details.version}. Checks run on this computer only; nothing is sent anywhere.
          </p>
        </section>
      )}
    </>
  );
}
