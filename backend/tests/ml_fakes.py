"""Fast stand-ins for the Kraken recognizer / trainer, so the job lifecycle can be tested without a GPU."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.pipeline.recognize import (
    Cancelled,
    EpochInfo,
    OutOfMemory,
    Prediction,
    Recognizer,
    Trainer,
    TrainObserver,
    TrainOutcome,
    TrainRequest,
)

REF = "hello world"  # the ground-truth text used by the tests


class FakeRecognizer(Recognizer):
    """Reads every line as `text_for(model_path)` with confidence 0.9. The default makes the model whose file name
    contains 'base' slightly wrong (1 edit in 11 characters) and every other model perfect."""

    engine = "kraken"
    instances: list[FakeRecognizer] = []
    on_predict: Callable[[FakeRecognizer, int], None] | None = None  # hook called with the batch size

    def __init__(self, path, device="auto", batch_size=4):
        self.path = Path(path)
        self.device = "cpu"
        self.batch_size = batch_size
        self.closed = False
        self.calls = 0
        FakeRecognizer.instances.append(self)

    def text(self) -> str:
        return "hello wxrld" if "base" in self.path.name or "base" in str(self.path.parent) else REF

    def predict(self, line_images, should_stop=None):
        self.calls += 1
        if FakeRecognizer.on_predict:
            FakeRecognizer.on_predict(self, len(line_images))
        if should_stop and should_stop():
            raise Cancelled()
        return [Prediction(self.text(), 0.9) for _ in line_images]

    def close(self):
        self.closed = True


class FakeTrainer(Trainer):
    """Writes a dummy model file and reports `epochs` epochs. Behaviour switches (class attributes) let a test make
    it block until cancelled, run out of memory, or take a while."""

    engine = "kraken"
    block_until_cancelled = False
    oom = False
    started = threading.Event()
    last_request: TrainRequest | None = None

    def train(self, request: TrainRequest, observer: TrainObserver | None = None) -> TrainOutcome:
        observer = observer or TrainObserver()
        FakeTrainer.last_request = request
        FakeTrainer.started.set()
        if FakeTrainer.oom:
            raise OutOfMemory(
                f"the GPU ran out of memory at batch size {request.batch_size}. Retry with a smaller batch size"
            )
        history = []
        for e in range(1, request.epochs + 1):
            if FakeTrainer.block_until_cancelled:
                while not observer.cancelled():
                    time.sleep(0.01)
            if observer.cancelled():
                raise Cancelled()
            observer.on_batch(e, request.epochs, 1, 1, 10.0 / e)
            info = EpochInfo(e, request.epochs, 10.0 / e, 0.5 + 0.05 * e, 0.01)
            history.append(info)
            observer.on_epoch(info)
        request.out_path.write_bytes(b"fake model")
        return TrainOutcome(
            model_path=request.out_path,
            best_epoch=len(history),
            best_val_accuracy=history[-1].val_accuracy,
            epochs_run=len(history),
            stopped_early=False,
            history=history,
            seconds=0.1,
            device="cpu",
        )

    @classmethod
    def reset(cls):
        cls.block_until_cancelled = False
        cls.oom = False
        cls.started = threading.Event()
        cls.last_request = None
