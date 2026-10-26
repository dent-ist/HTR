// Tiny hash router: "#/" is the library, "#/pages/<id>" the page editor (optionally "?line=<line id>" to open on that
// line), "#/pages/<id>/text" the raw / expanded text view, "#/training", "#/review", "#/abbreviations" and "#/export"
// the other pages. No dependency needed for a handful of screens.
import { useEffect, useState } from "react";

export type Route =
  | { name: "library" }
  | { name: "page"; id: number; line?: number }
  | { name: "text"; id: number }
  | { name: "training" }
  | { name: "review" }
  | { name: "abbreviations" }
  | { name: "export" };

export const TRAINING_HASH = "#/training";
export const REVIEW_HASH = "#/review";
export const ABBREVIATIONS_HASH = "#/abbreviations";
export const EXPORT_HASH = "#/export";

export function parseHash(hash: string): Route {
  if (/^#\/training\/?$/.test(hash)) return { name: "training" };
  if (/^#\/review\/?$/.test(hash)) return { name: "review" };
  if (/^#\/abbreviations\/?$/.test(hash)) return { name: "abbreviations" };
  if (/^#\/export\/?$/.test(hash)) return { name: "export" };
  const t = /^#\/pages\/(\d+)\/text\/?$/.exec(hash);
  if (t) return { name: "text", id: Number(t[1]) };
  const m = /^#\/pages\/(\d+)\/?(?:\?line=(\d+))?$/.exec(hash);
  if (!m) return { name: "library" };
  return m[2] ? { name: "page", id: Number(m[1]), line: Number(m[2]) } : { name: "page", id: Number(m[1]) };
}

export const pageHash = (id: number, lineId?: number): string => `#/pages/${id}${lineId !== undefined ? `?line=${lineId}` : ""}`;
export const textHash = (id: number): string => `#/pages/${id}/text`;

let guard: (() => boolean) | null = null;

/** The editor registers a function returning true when leaving is OK (or the user confirmed). */
export function setNavigationGuard(fn: (() => boolean) | null): void {
  guard = fn;
}

export function navigate(hash: string): void {
  if (guard && !guard()) return;
  window.location.hash = hash;
}

export function useRoute(): Route {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    let current = window.location.hash;
    const onChange = () => {
      const next = window.location.hash;
      if (guard && !guard()) {
        // Back/forward button while there are unsaved changes and the user said no: undo the navigation.
        history.replaceState(null, "", current || "#/");
        return;
      }
      current = next;
      setHash(next);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return parseHash(hash);
}
