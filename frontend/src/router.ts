// Tiny hash router: "#/" is the library, "#/pages/<id>" the page editor, "#/training" the training page.
// No dependency needed for three screens.
import { useEffect, useState } from "react";

export type Route = { name: "library" } | { name: "page"; id: number } | { name: "training" };

export const TRAINING_HASH = "#/training";

export function parseHash(hash: string): Route {
  if (/^#\/training\/?$/.test(hash)) return { name: "training" };
  const m = /^#\/pages\/(\d+)\/?$/.exec(hash);
  return m ? { name: "page", id: Number(m[1]) } : { name: "library" };
}

export const pageHash = (id: number): string => `#/pages/${id}`;

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
