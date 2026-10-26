import { useEffect, useRef } from "react";
import { SHORTCUTS } from "./transcription";

export function HelpOverlay({ onClose }: { onClose(): void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
  }, []);
  return (
    <div className="overlay-backdrop" onClick={onClose}>
      <div
        className="overlay"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-title"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape" || e.key === "?" || e.key === "F1") {
            e.preventDefault();
            e.stopPropagation();
            onClose();
          }
        }}
      >
        <h2 id="help-title">Keyboard shortcuts</h2>
        <table>
          <tbody>
            {SHORTCUTS.map((s) => (
              <tr key={s.keys}>
                <th scope="row">
                  <kbd>{s.keys}</kbd>
                </th>
                <td>{s.what}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button ref={closeRef} onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
