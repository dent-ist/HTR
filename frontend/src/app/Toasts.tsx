import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

export type ToastKind = "error" | "info" | "warning";
export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

interface ToastApi {
  push: (kind: ToastKind, text: string) => void;
  error: (text: string) => void;
}

const noop: ToastApi = { push: () => undefined, error: () => undefined };
const Ctx = createContext<ToastApi>(noop);

/** Show a non-blocking message from anywhere below <ToastProvider>. Outside a provider it does nothing. */
export const useToasts = (): ToastApi => useContext(Ctx);

const MAX_VISIBLE = 4;
const AUTO_DISMISS_MS = 8000;

/**
 * A small stack of dismissible messages in the corner. Identical consecutive messages are merged, errors stay a
 * little longer than notices, and nothing steals focus.
 */
export function ToastProvider({ children, autoDismissMs = AUTO_DISMISS_MS }: { children: ReactNode; autoDismissMs?: number }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setToasts((cur) => cur.filter((t) => t.id !== id)), []);

  const push = useCallback(
    (kind: ToastKind, text: string) => {
      setToasts((cur) => {
        if (cur.some((t) => t.kind === kind && t.text === text)) return cur;
        const id = nextId.current++;
        if (autoDismissMs > 0) setTimeout(() => dismiss(id), kind === "error" ? autoDismissMs * 2 : autoDismissMs);
        return [...cur, { id, kind, text }].slice(-MAX_VISIBLE);
      });
    },
    [autoDismissMs, dismiss],
  );

  const api = useMemo<ToastApi>(() => ({ push, error: (text) => push("error", text) }), [push]);

  return (
    <Ctx.Provider value={api}>
      {children}
      <div className="toasts" aria-live="polite" data-testid="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`} role={t.kind === "error" ? "alert" : "status"}>
            <span>{t.text}</span>
            <button aria-label="Dismiss message" onClick={() => dismiss(t.id)}>
              Dismiss
            </button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
