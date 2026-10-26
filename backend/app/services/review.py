"""Active learning: the review queue (lowest-confidence predictions first) and the log of what reviewing cost.

A *prediction* is a line with `text_source = 'predicted'`, not ground truth, not ignored. `predicted_by_model_id` and
`predicted_text` (set by the predict job) remember who predicted what. When the user turns such a line into ground
truth, the line change and a `review_events` row are written in the same transaction:

* `accepted`  - the final text equals the predicted text (one keystroke in the UI)
* `corrected` - it differs; `edit_distance` is the number of characters changed (Levenshtein)
* `ignored`   - the line was ignored instead

The event also clears `predicted_by_model_id`, so one prediction yields at most one event. Correction effort per model
(`stats`) is what shows whether review gets cheaper from round to round.
"""

from __future__ import annotations

import sqlite3

from ..db import now_iso
from ..pipeline.evaluate import edit_distance
from .rows import line_dict

CANDIDATE_SQL = (
    "FROM lines JOIN pages ON pages.id = lines.page_id WHERE lines.text_source = 'predicted' "
    "AND lines.is_ground_truth = 0 AND lines.ignored = 0 AND lines.confidence IS NOT NULL"
)
BUCKETS = 10


def record_if_reviewed(
    conn: sqlite3.Connection, cur: sqlite3.Row, new_text: str | None, new_gt, new_ignored
) -> str | None:
    """Call inside the transaction that applies a line patch, *before* the UPDATE, with the row as it is now and the
    patch's text / is_ground_truth / ignored values (None when absent). Returns the recorded action, if any."""
    if cur["predicted_by_model_id"] is None:
        return None
    predicted = cur["predicted_text"] if cur["predicted_text"] is not None else cur["text"]
    confidence = cur["confidence"] if cur["text_source"] == "predicted" else None
    if new_gt and not cur["is_ground_truth"]:
        final = new_text if new_text is not None else cur["text"]
        dist = edit_distance(predicted, final)
        action = "accepted" if dist == 0 else "corrected"
    elif new_ignored and not cur["ignored"]:
        final, dist, action = None, None, "ignored"
    else:
        return None
    conn.execute(
        "INSERT INTO review_events (line_id, model_id, predicted_text, confidence, final_text, edit_distance, "
        "predicted_chars, action, reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cur["id"],
            cur["predicted_by_model_id"],
            predicted,
            confidence,
            final,
            dist,
            len(predicted),
            action,
            now_iso(),
        ),
    )
    conn.execute("UPDATE lines SET predicted_by_model_id = NULL WHERE id = ?", (cur["id"],))
    return action


def _exclude_clause(exclude: list[int]) -> tuple[str, list]:
    if not exclude:
        return "", []
    return f" AND lines.id NOT IN ({','.join('?' for _ in exclude)})", list(exclude)


def _item(r: sqlite3.Row) -> dict:
    return {
        "line": line_dict(r),
        "page_id": r["page_id"],
        "crop_url": f"/lines/{r['id']}/crop.png",
        "predicted_text": r["text"],
        "confidence": r["confidence"],
        "model_id": r["predicted_by_model_id"],
    }


def next_items(conn: sqlite3.Connection, exclude: list[int], limit: int) -> dict:
    """The `limit` lowest-confidence predictions (ties by id) that are not in `exclude`, and how many remain."""
    clause, args = _exclude_clause(exclude)
    rows = conn.execute(
        f"SELECT lines.* {CANDIDATE_SQL}{clause} ORDER BY lines.confidence ASC, lines.id ASC LIMIT ?", (*args, limit)
    ).fetchall()
    remaining = conn.execute(f"SELECT COUNT(*) {CANDIDATE_SQL}{clause}", args).fetchone()[0]
    return {"items": [_item(r) for r in rows], "remaining": remaining}


def queue_overview(conn: sqlite3.Connection, lowest: int = 0) -> dict:
    """Queue size, a histogram of confidence (10 buckets of 0.1; 1.0 falls in the last) and optionally the N lowest."""
    confs = [r[0] for r in conn.execute(f"SELECT lines.confidence {CANDIDATE_SQL}")]
    counts = [0] * BUCKETS
    for c in confs:
        counts[min(BUCKETS - 1, int(max(0.0, min(1.0, c)) * BUCKETS))] += 1
    out = {
        "total": len(confs),
        "buckets": [{"min": i / BUCKETS, "max": (i + 1) / BUCKETS, "count": n} for i, n in enumerate(counts)],
        "mean_confidence": sum(confs) / len(confs) if confs else None,
    }
    if lowest:
        out["lowest"] = next_items(conn, [], lowest)["items"]
    return out


def stats(conn: sqlite3.Connection) -> dict:
    """Correction effort per model (= per training round), oldest first, joined with the model's held-out CER."""
    rows = conn.execute(
        "SELECT e.model_id, e.action, e.edit_distance, e.predicted_chars, e.confidence FROM review_events e"
    ).fetchall()
    per: dict[int | None, dict] = {}
    for r in rows:
        d = per.setdefault(r["model_id"], {"accepted": 0, "corrected": 0, "ignored": 0, "edits": 0, "chars": 0})
        d[r["action"]] += 1
        if r["action"] != "ignored":
            d["edits"] += r["edit_distance"] or 0
            d["chars"] += r["predicted_chars"]
    models = {
        m["id"]: m
        for m in conn.execute(
            "SELECT id, name, kind, trained_at, train_lines, eval_lines, cer, wer, base_cer FROM models"
        ).fetchall()
    }
    out = []
    for model_id, d in per.items():
        m = models.get(model_id)
        judged = d["accepted"] + d["corrected"]
        out.append(
            {
                "model_id": model_id,
                "model_name": (m["name"] or f"model {model_id}") if m else None,
                "kind": m["kind"] if m else None,
                "trained_at": m["trained_at"] if m else None,
                "train_lines": m["train_lines"] if m else None,
                "cer": m["cer"] if m else None,
                "wer": m["wer"] if m else None,
                "reviewed": judged + d["ignored"],
                "accepted": d["accepted"],
                "corrected": d["corrected"],
                "ignored": d["ignored"],
                "accepted_rate": d["accepted"] / judged if judged else None,
                "mean_edits_per_line": d["edits"] / judged if judged else None,
                "mean_edits_per_char": d["edits"] / d["chars"] if d["chars"] else None,
            }
        )
    out.sort(key=lambda s: (s["trained_at"] or "", s["model_id"] or 0))
    for i, s in enumerate(out, 1):
        s["round"] = i
    total_judged = sum(s["accepted"] + s["corrected"] for s in out)
    return {
        "rounds": out,
        "total_reviewed": sum(s["reviewed"] for s in out),
        "accepted_rate": sum(s["accepted"] for s in out) / total_judged if total_judged else None,
    }
