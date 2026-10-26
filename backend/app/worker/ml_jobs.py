"""The `train` and `predict` job handlers.

train:   export ground truth -> score the start model on the held-out lines ("before") -> fine-tune -> score the
         best epoch on the same lines ("after") -> register the model. Progress 0-1 across those phases; the live
         status (phase, epoch, loss, validation accuracy) is the running job's `result`.
predict: read every line that is not ground truth, not ignored and has no manual text, page by page, and write text +
         confidence (text_source='predicted') through the transactional PAGE-XML sync. Never touches ground truth or
         manual text, not even if the user edits a line while the job runs (the UPDATE re-checks under the lock).
Both poll for cancellation (batch / page granularity) and turn a CUDA out-of-memory into a plain failure message.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import unicodedata
from pathlib import Path

from PIL import Image

from ..db import write_tx
from ..errors import Invalid, JobError
from ..pipeline import wordcrop
from ..pipeline.evaluate import evaluate
from ..pipeline.recognize import (
    Cancelled,
    EpochInfo,
    OutOfMemory,
    RecognitionError,
    TrainObserver,
    TrainRequest,
    extract_line_image,
    free_cuda,
    load_recognizer,
    make_trainer,
)
from ..services import models as models_svc
from ..services import sync
from ..services import training as training_svc
from . import queue
from .handlers import JobContext

log = logging.getLogger(__name__)

DEFAULT_EPOCHS = 15
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PATIENCE = 5
MAX_TEXT = 2000


def clean_prediction(text: str) -> str:
    """A predicted line as the API accepts text: no control characters, at most 2000 characters, NFC untouched."""
    return "".join(c for c in text if unicodedata.category(c) != "Cc")[:MAX_TEXT]


def _device(ctx: JobContext) -> str:
    """The device to run on. When a GPU is wanted ("auto" or "cuda...") but CUDA is not available, fall back to the
    CPU and leave a visible notice on the job (it ends up in `result.notices`)."""
    device = ctx.payload.get("device") or ctx.settings.device
    if device == "cpu":
        return device
    import torch

    if torch.cuda.is_available():
        return device
    note = (
        "No usable GPU (CUDA) was found, so this job ran on the CPU, which is much slower. "
        "Run `make check` to see why the GPU is not visible."
    )
    if note not in ctx.notices:
        ctx.notices.append(note)
    return "cpu"


class _Throttle:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self.last >= self.seconds:
            self.last = now
            return True
        return False


class _Observer(TrainObserver):
    """Turns trainer callbacks into job progress / live info and polls for cancellation."""

    def __init__(self, ctx: JobContext, epochs: int, base_info: dict):
        self.ctx, self.epochs, self.base = ctx, epochs, base_info
        self.history: list[dict] = []
        self.loss: float | None = None
        self._info = _Throttle(1.0)
        self._cancel = _Throttle(0.5)
        self._cancelled = False

    def _publish(self, epoch: int, batch: int, n_batches: int, finished_epoch: bool = False) -> None:
        finished = epoch if finished_epoch else (epoch - 1) + batch / max(1, n_batches)
        self.ctx.progress(0.12 + 0.78 * min(1.0, finished / max(1, self.epochs)))
        self.ctx.info(
            {
                **self.base,
                "phase": "training",
                "epoch": epoch,
                "epochs": self.epochs,
                "batch": batch,
                "batches": n_batches,
                "loss": self.loss,
                "val_accuracy": self.history[-1]["val_accuracy"] if self.history else None,
                "history": self.history,
            }
        )

    def on_batch(self, epoch, epochs, batch, n_batches, loss):
        if loss is not None:
            self.loss = loss
        if self._info.ready():
            self._publish(epoch, batch, n_batches)

    def on_epoch(self, info: EpochInfo):
        if info.train_loss is not None:
            self.loss = info.train_loss
        self.history.append(info.as_dict())
        self._publish(info.epoch, 1, 1, finished_epoch=True)

    def cancelled(self) -> bool:
        if not self._cancelled and self._cancel.ready():
            self._cancelled = self.ctx.cancel_requested()
        return self._cancelled


def _payload_int(p: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = p.get(key)
    v = default if v is None else v
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise Invalid(f"{key} must be an integer between {lo} and {hi}")
    return v


def _check_preprocess(model_row, page_keys) -> str:
    """The one line preprocessing of the pages being used; JobError when the pages disagree with each other or with
    the model (a model must only ever read pages prepared like the pages it was trained on, pipeline/wordcrop.py)."""
    keys = sorted(set(page_keys))
    if len(keys) > 1:
        raise JobError(
            f"the ground-truth pages were prepared in different ways ({', '.join(keys)}); a model can only be "
            "trained on one kind"
        )
    key = keys[0] if keys else wordcrop.PAGE_KEY
    model_key = models_svc.line_preprocess_key(model_row)
    if not wordcrop.compatible(model_key, key):
        name = model_row["name"] or f"model {model_row['id']}"
        raise JobError(
            f"{name} was trained for '{model_key}' pages but these pages are '{key}'; using it would silently "
            "degrade accuracy. Pick a model that matches (or start from a base model)."
        )
    return key


def train_model(ctx: JobContext) -> dict:
    """payload: {epochs?, batch_size?, learning_rate?, base_model_id?, device?, num_workers?, patience?, augment?,
    warnings?}. See module docstring."""
    p, conn, s = ctx.payload, ctx.conn, ctx.settings
    epochs = _payload_int(p, "epochs", DEFAULT_EPOCHS, 1, 500)
    batch_size = _payload_int(p, "batch_size", s.train_batch_size, 1, 256)
    patience = _payload_int(p, "patience", DEFAULT_PATIENCE, 1, 100)
    workers = _payload_int(p, "num_workers", s.train_workers, 0, 16)
    lr = p.get("learning_rate", DEFAULT_LEARNING_RATE)
    augment = bool(p.get("augment", True))
    t_start = time.time()

    start = models_svc.resolve_model(conn, s, p.get("base_model_id"))
    start_path = models_svc.model_file(s, start)
    engine = start["engine"]
    start_name = start["name"] or f"model {start['id']}"

    rows = training_svc.usable_lines(conn)
    if not rows:
        raise JobError("there are no usable ground-truth lines to train on")
    line_pp = _check_preprocess(start, training_svc.page_preprocess_keys(conn, {r["page_id"] for r in rows}).values())
    ids = [r["id"] for r in rows]
    splits = training_svc.assign_splits(conn, ids)
    train_ids = [i for i in ids if splits[i] == "train"]
    if not train_ids:
        raise JobError("every ground-truth line is in the held-out set; add more transcribed lines")
    val_ids, val_is_train = training_svc.choose_validation(train_ids)
    run_id = f"run-{ctx.job['id']}"
    ctx.progress(0.02)
    ctx.info({"phase": "exporting", "ground_truth_lines": len(ids)})
    hyper = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patience": patience,
        "augment": augment,
        "num_workers": workers,
        "device": _device(ctx),
        "optimizer": "AdamW",
        "seed": 1234,
        "line_preprocess": line_pp,
        "new_characters": "added to the output layer (resize=union)",
    }
    export = training_svc.export_run(
        conn,
        s,
        run_id,
        {i: splits[i] for i in ids},
        set(val_ids),
        val_is_train,
        {"job_id": ctx.job["id"], "start_model": {"id": start["id"], "name": start_name}, "hyperparameters": hyper},
    )
    if not export.train_files:
        raise JobError("no training line could be extracted from the page images")
    ctx.check_cancelled()
    base_info = {
        "start_model": start_name,
        "train_lines": len(train_ids),
        "validation_lines": len(val_ids),
        "held_out_lines": len(export.eval_samples),
    }

    # ---- before: the start model on the held-out lines
    before = None
    if export.eval_samples:
        ctx.progress(0.05)
        ctx.info({**base_info, "phase": "evaluating the start model"})
        before = _score(ctx, engine, start_path, export.eval_samples)
        ctx.progress(0.12)
    ctx.check_cancelled()

    # ---- train
    models_dir = s.data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pending = models_dir / f"pending-{ctx.job['id']}.mlmodel"
    pending.unlink(missing_ok=True)
    observer = _Observer(ctx, epochs, base_info)
    req = TrainRequest(
        start_model=start_path,
        out_path=pending,
        train_files=export.train_files,
        val_files=export.val_files,
        work_dir=export.run_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(lr),
        patience=patience,
        device=_device(ctx),
        num_workers=workers,
        augment=augment,
    )
    try:
        outcome = make_trainer(engine).train(req, observer)
    except Cancelled:
        pending.unlink(missing_ok=True)
        raise queue.JobCancelled({**base_info, "phase": "cancelled", "history": observer.history}) from None
    except OutOfMemory as e:
        pending.unlink(missing_ok=True)
        raise JobError(str(e)) from e
    except RecognitionError as e:
        pending.unlink(missing_ok=True)
        raise JobError(str(e)) from e
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    try:
        ctx.check_cancelled({**base_info, "phase": "cancelled", "history": observer.history})
        # ---- after: the best epoch on the same held-out lines
        after = None
        if export.eval_samples:
            ctx.progress(0.92)
            ctx.info({**base_info, "phase": "evaluating the new model", "history": observer.history})
            after = _score(ctx, engine, outcome.model_path, export.eval_samples)
        model_id, activated, note = _register(
            ctx,
            start,
            outcome,
            hyper,
            base_info,
            observer.history,
            before,
            after,
            splits,
            val_ids,
            val_is_train,
            export,
            engine,
        )
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    result = {
        **base_info,
        "phase": "done",
        "model_id": model_id,
        "activated": activated,
        "epochs_run": outcome.epochs_run,
        "best_epoch": outcome.best_epoch,
        "stopped_early": outcome.stopped_early,
        "best_val_accuracy": outcome.best_val_accuracy,
        "before": before.as_dict() if before else None,
        "after": after.as_dict() if after else None,
        "cer": after.cer if after else None,
        "wer": after.wer if after else None,
        "base_cer": before.cer if before else None,
        "seconds": round(time.time() - t_start, 1),
        "training_seconds": round(outcome.seconds, 1),
        "device": outcome.device,
        "peak_vram_mb": round(outcome.peak_vram_mb, 1) if outcome.peak_vram_mb is not None else None,
        "new_characters": outcome.alphabet_added,
        "history": observer.history,
        "run_dir": str(export.run_dir.relative_to(s.data_dir)),
        "warnings": [*p.get("warnings", []), *([note] if note else [])],
    }
    if p.get("then_predict"):
        # A "round" (POST /jobs/round): predict every non-ground-truth line with the new model, queued right behind.
        if activated:
            nxt = queue.enqueue(conn, "predict", {"model_id": model_id, "round_job_id": ctx.job["id"]})
            result["next_job_id"] = nxt["id"]
        else:
            result["warnings"].append("Round: the new model was not activated, so no prediction job was started.")
    return result


def _score(ctx: JobContext, engine: str, model_path: Path, samples):
    """Metrics of a model on (line id, crop, text) samples, via the same recognizer the predict job uses."""
    try:
        rec = load_recognizer(engine, model_path, _device(ctx), ctx.settings.predict_batch_size)
    except Exception as e:  # noqa: BLE001
        raise JobError(f"could not load model {model_path.name}: {e}") from e
    try:
        preds = rec.predict([im for _, im, _ in samples], should_stop=ctx.cancel_requested)
    except Cancelled:
        raise queue.JobCancelled() from None
    except OutOfMemory as e:
        raise JobError(str(e)) from e
    finally:
        rec.close()
    return evaluate([(text, pred.text) for (_, _, text), pred in zip(samples, preds, strict=True)])


def _register(
    ctx, start, outcome, hyper, base_info, history, before, after, splits, val_ids, val_is_train, export, engine
):
    """Insert the models row, move the file to models/<id>.mlmodel, decide on activation. Returns (id, activated,
    note)."""
    conn, s = ctx.conn, ctx.settings
    n_prev = conn.execute("SELECT COUNT(*) FROM models WHERE kind = 'finetuned'").fetchone()[0]
    fields = {
        "path": outcome.model_path.relative_to(s.data_dir).as_posix(),
        "name": f"fine-tune {n_prev + 1}",
        "engine": engine,
        "base_model": start["name"] or f"model {start['id']}",
        "parent_model_id": start["id"],
        "train_lines": base_info["train_lines"],
        "eval_lines": len(export.eval_samples),
        "cer": after.cer if after else None,
        "wer": after.wer if after else None,
        "base_cer": before.cer if before else None,
        "base_wer": before.wer if before else None,
        "hyperparameters": json.dumps({**hyper, "epochs_run": outcome.epochs_run, "device": outcome.device}),
        "split_info": json.dumps(
            {
                "rule": training_svc.SPLIT_RULE_VERSION,
                "held_out_percent": training_svc.EVAL_PERMILLE_X10 / 100,
                "train_lines": base_info["train_lines"],
                "validation_lines": len(val_ids),
                "validation_is_training_set": val_is_train,
                "held_out_lines": len(export.eval_samples),
                "run_dir": str(export.run_dir.relative_to(s.data_dir)),
            }
        ),
        "metrics": json.dumps(
            {
                "history": history,
                "best_epoch": outcome.best_epoch,
                "best_val_accuracy": outcome.best_val_accuracy,
                "seconds": round(outcome.seconds, 1),
                "peak_vram_mb": outcome.peak_vram_mb,
                "before": before.as_dict() if before else None,
                "after": after.as_dict() if after else None,
            }
        ),
        "job_id": ctx.job["id"],
        "license": start["license"],
    }
    lines = [(lid, "eval" if sp == "eval" else "train") for lid, sp in splits.items()]
    note = None
    final = outcome.model_path
    with write_tx(conn):
        model_id = models_svc.create_finetuned(conn, fields, lines)
        final = s.data_dir / "models" / f"{model_id}.mlmodel"
        shutil.move(str(outcome.model_path), final)
        conn.execute("UPDATE models SET path = ? WHERE id = ?", (final.relative_to(s.data_dir).as_posix(), model_id))
        worse = after is not None and before is not None and (after.cer or 0) > (before.cer or 0)
        activated = not worse
        if activated:
            conn.execute("UPDATE models SET is_active = 0 WHERE is_active = 1")
            conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        else:
            note = (
                f"The new model is worse than its start model on the held-out lines (CER {after.cer:.3f} vs "
                f"{before.cer:.3f}); it was kept but not activated."
            )
    return model_id, activated, note


# ---- predict ---------------------------------------------------------------------------------------------------


def predict_lines(ctx: JobContext) -> dict:
    """payload: {model_id?, page_ids?, batch_size?, device?}. Predicts every line that is not ground truth, not
    ignored and carries no manual text. Result: counts, device, mean confidence."""
    p, conn, s = ctx.payload, ctx.conn, ctx.settings
    batch = _payload_int(p, "batch_size", s.predict_batch_size, 1, 256)
    row = models_svc.resolve_model(conn, s, p.get("model_id"))
    page_ids = p.get("page_ids")
    sql = (
        'SELECT lines.id, lines.page_id, lines."index", lines.baseline, lines.polygon FROM lines '
        "JOIN pages ON pages.id = lines.page_id WHERE lines.is_ground_truth = 0 AND lines.ignored = 0 "
        "AND (lines.text_source IS NULL OR lines.text_source = 'predicted') AND pages.segmented_at IS NOT NULL"
    )
    args: list = []
    if page_ids is not None:
        sql += f" AND lines.page_id IN ({','.join('?' for _ in page_ids)})"
        args = list(page_ids)
    sql += ' ORDER BY lines.page_id, lines."index"'
    todo = conn.execute(sql, args).fetchall()
    by_page: dict[int, list] = {}
    for r in todo:
        by_page.setdefault(r["page_id"], []).append(r)
    if todo:
        _check_preprocess(row, training_svc.page_preprocess_keys(conn, {r["page_id"] for r in todo}).values())
    total = len(todo)
    model_name = row["name"] or f"model {row['id']}"
    counts = {"lines": total, "predicted": 0, "unreadable": 0, "changed_meanwhile": 0, "empty": 0}
    confidences: list[float] = []
    t0 = time.time()

    def result(phase: str) -> dict:
        return {
            "phase": phase,
            "model_id": row["id"],
            "model": model_name,
            **counts,
            "pages": len(by_page),
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "seconds": round(time.time() - t0, 1),
        }

    if not total:
        return {**result("done"), "device": None}
    ctx.info(result("loading the model"))
    try:
        rec = load_recognizer(row["engine"], models_svc.model_file(s, row), _device(ctx), batch)
    except Exception as e:  # noqa: BLE001
        raise JobError(f"could not load model {model_name}: {e}") from e
    done = 0
    try:
        for page_id, rows in by_page.items():
            ctx.check_cancelled(result("cancelled"))
            page = conn.execute("SELECT working_path FROM pages WHERE id = ?", (page_id,)).fetchone()
            crops: list[tuple[int, Image.Image]] = []
            with Image.open(s.data_dir / page["working_path"]) as im:
                im.load()
                for r in rows:
                    crop = extract_line_image(
                        im,
                        [tuple(pt) for pt in json.loads(r["baseline"])],
                        [tuple(pt) for pt in json.loads(r["polygon"])],
                    )
                    if crop is None:
                        counts["unreadable"] += 1
                    else:
                        crops.append((r["id"], crop))
            try:
                preds = rec.predict([c for _, c in crops], should_stop=ctx.cancel_requested)
            except Cancelled:
                raise queue.JobCancelled(result("cancelled")) from None
            except OutOfMemory as e:
                raise JobError(str(e)) from e
            written = _write_predictions(
                ctx, page_id, [(lid, pr) for (lid, _), pr in zip(crops, preds, strict=True)], row["id"]
            )
            counts["predicted"] += written["written"]
            counts["changed_meanwhile"] += written["skipped"]
            counts["empty"] += written["empty"]
            confidences.extend(written["confidences"])
            done += len(rows)
            ctx.progress(done / total)
            ctx.info(result("predicting"))
        out = {**result("done"), "device": rec.device, "cuda_fallback": getattr(rec, "fell_back_to_cpu", False)}
    finally:
        rec.close()
        free_cuda()
    return out


def _write_predictions(ctx: JobContext, page_id: int, preds, model_id: int | None = None) -> dict:
    """Write one page's predictions in one transaction (DB + PAGE-XML). Only lines that are *still* eligible are
    updated: the WHERE clause is evaluated under the write lock, so a line the user transcribed or marked as ground
    truth in the meantime is left alone."""
    written = skipped = empty = 0
    confs: list[float] = []
    with sync.page_transaction(ctx.conn, ctx.settings, page_id):
        for line_id, pr in preds:
            text = clean_prediction(pr.text)
            conf = min(1.0, max(0.0, float(pr.confidence))) if text else None
            cur = ctx.conn.execute(
                "UPDATE lines SET text = ?, text_source = ?, confidence = ?, predicted_by_model_id = ?, "
                "predicted_text = ? WHERE id = ? AND is_ground_truth = 0 "
                "AND ignored = 0 AND (text_source IS NULL OR text_source = 'predicted')",
                (text, "predicted" if text else None, conf, model_id if text else None, text or None, line_id),
            )
            if cur.rowcount:
                written += 1
                if text:
                    confs.append(conf)
                else:
                    empty += 1
            else:
                skipped += 1
    return {"written": written, "skipped": skipped, "empty": empty, "confidences": confs}
