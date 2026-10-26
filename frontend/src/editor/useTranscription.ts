import { useEffect, useReducer, useRef } from "react";
import type { ApiClient } from "../api/client";
import { type EditorAction, type EditorState, initialState, reducer } from "./transcription";

/** Runs the pure reducer and sends its outbox of PATCH requests, reporting each result back. */
export function useTranscription(api: ApiClient): [EditorState, (a: EditorAction) => void] {
  const [state, dispatch] = useReducer(reducer, initialState);
  const sent = useRef(new Set<number>());

  useEffect(() => {
    for (const req of state.outbox) {
      if (sent.current.has(req.reqId)) continue;
      sent.current.add(req.reqId);
      dispatch({ type: "dispatched", reqId: req.reqId });
      api
        .patchLine(req.lineId, req.patch)
        .then((line) => dispatch({ type: "saveOk", reqId: req.reqId, line }))
        .catch((e: unknown) =>
          dispatch({ type: "saveFail", reqId: req.reqId, message: e instanceof Error ? e.message : String(e) }),
        );
    }
  }, [state.outbox, api]);

  return [state, dispatch];
}
