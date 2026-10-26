"""Fine-tuning a Kraken recognition model (Kraken 7.1.1: `kraken.train.VGSLRecognitionModel` on PyTorch Lightning).

What this does, in Kraken terms (the same as `ketos train --load <model> -f page ...`):

* loads the start model's weights and codec; characters in the training data that the model does not know are
  *added* to the output layer (`resize='union'`), nothing is removed;
* trains on PAGE-XML baseline data with Kraken's `augment` option on (random mild warp/rotation/scale, brightness
  and contrast, blur, noise; see `kraken.lib.dataset.augment.DefaultAugmenter`) and AdamW at a small learning rate;
* after every epoch scores a validation set (1 - CER, Kraken's `val_accuracy`), keeps the best checkpoint, and stops
  early after `patience` epochs without improvement;
* text is used exactly as written: no Unicode normalization, whitespace normalization or bidi reordering, so that
  training, prediction and evaluation all see the same strings.

Memory: the network is small (about 4M parameters) but activations grow with line width. The default batch size is
conservative; a CUDA out-of-memory becomes an `OutOfMemory` error with advice, never a crash.
"""

from __future__ import annotations

import logging
import shutil
import time

from .recognize import (
    Cancelled,
    EpochInfo,
    OutOfMemory,
    RecognitionError,
    Trainer,
    TrainObserver,
    TrainOutcome,
    TrainRequest,
    free_cuda,
    is_oom,
    torch_device,
)

log = logging.getLogger(__name__)


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class KrakenTrainer(Trainer):
    engine = "kraken"

    def train(self, request: TrainRequest, observer: TrainObserver | None = None) -> TrainOutcome:
        observer = observer or TrainObserver()
        try:
            return self._train(request, observer)
        except Cancelled:
            raise
        except RecognitionError:
            raise
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                free_cuda()
                raise OutOfMemory(
                    f"the GPU ran out of memory at batch size {request.batch_size}. Retry with a smaller batch size "
                    f"(for example {max(1, request.batch_size // 2)}) or train on the CPU (device 'cpu')."
                ) from e
            raise
        finally:
            free_cuda()

    def _train(self, req: TrainRequest, observer: TrainObserver) -> TrainOutcome:
        import lightning as L
        import torch
        from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig
        from kraken.models.convert import convert_models
        from kraken.train import KrakenTrainer as KTrainer
        from kraken.train import VGSLRecognitionDataModule, VGSLRecognitionModel
        from lightning.pytorch.callbacks import Callback, ModelCheckpoint

        if not req.train_files or not req.val_files:
            raise RecognitionError("training needs at least one training and one validation page file")
        if req.out_path.exists():
            raise RecognitionError(f"{req.out_path} already exists")
        dev = torch_device(req.device)
        on_gpu = dev.startswith("cuda")
        accelerator = "gpu" if on_gpu else "cpu"
        devices = [int(dev.split(":")[1]) if ":" in dev else 0] if on_gpu else 1
        ckpt_dir = req.work_dir / "checkpoints"
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        L.seed_everything(req.seed, workers=True)

        cfg = VGSLRecognitionTrainingConfig(
            epochs=req.epochs,
            batch_size=req.batch_size,
            lrate=req.learning_rate,
            optimizer="AdamW",
            schedule="constant",
            quit="early",
            lag=req.patience,
            resize="union",
            freq=1.0,
            checkpoint_path=str(ckpt_dir),
        )
        data_cfg = VGSLRecognitionTrainingDataConfig(
            training_data=[str(p) for p in req.train_files],
            evaluation_data=[str(p) for p in req.val_files],
            format_type="xml",
            augment=req.augment,
            num_workers=req.num_workers,
            batch_size=req.batch_size,
            normalize_whitespace=False,
            bidi_reordering=False,
            normalization=None,
        )

        history: list[EpochInfo] = []
        state = {"t0": time.time(), "cancelled": False}

        class Progress(Callback):
            def on_train_epoch_start(self, trainer, module):
                state["t0"] = time.time()

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = _f(trainer.callback_metrics.get("train_loss_step"))
                observer.on_batch(
                    trainer.current_epoch + 1, req.epochs, batch_idx + 1, trainer.num_training_batches, loss
                )
                if observer.cancelled():
                    state["cancelled"] = True
                    raise Cancelled()

            def on_train_epoch_end(self, trainer, module):
                m = trainer.callback_metrics
                info = EpochInfo(
                    epoch=trainer.current_epoch + 1,
                    epochs=req.epochs,
                    train_loss=_f(m.get("train_loss_epoch")),
                    val_accuracy=_f(m.get("val_accuracy")),
                    seconds=time.time() - state["t0"],
                )
                history.append(info)
                observer.on_epoch(info)
                if observer.cancelled():
                    state["cancelled"] = True
                    raise Cancelled()

        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor="val_metric",
            mode="max",
            auto_insert_metric_name=False,
            filename="best-{epoch:02d}-{val_metric:.4f}",
        )
        trainer = KTrainer(
            accelerator=accelerator,
            devices=devices,
            precision="32-true",
            max_epochs=req.epochs,
            min_epochs=0,
            enable_progress_bar=False,
            enable_summary=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            gradient_clip_val=cfg.gradient_clip_val,
            callbacks=[ckpt_cb, Progress()],
            default_root_dir=str(req.work_dir),
            check_val_every_n_epoch=1,
        )
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with trainer.init_module(empty_init=False):
            model = VGSLRecognitionModel.load_from_weights(str(req.start_model), config=cfg)
        data_module = VGSLRecognitionDataModule(data_cfg)
        labels_before = model.net.codec.max_label if model.net.codec else 0
        trainer.fit(model, data_module)
        if state["cancelled"]:
            raise Cancelled()
        best = ckpt_cb.best_model_path
        if not best:
            raise RecognitionError("training produced no checkpoint (is the validation set empty?)")
        score = _f(ckpt_cb.best_model_score)
        added = (model.net.codec.max_label if model.net.codec else 0) - labels_before
        out = convert_models([best], req.out_path, weights_format="coreml")
        peak = torch.cuda.max_memory_allocated() / 2**20 if on_gpu else None
        best_epoch = max(history, key=lambda h: h.val_accuracy if h.val_accuracy is not None else -1).epoch
        outcome = TrainOutcome(
            model_path=out,
            best_epoch=best_epoch if history else 0,
            best_val_accuracy=score,
            epochs_run=len(history),
            stopped_early=len(history) < req.epochs,
            history=history,
            seconds=time.time() - t0,
            device=dev,
            peak_vram_mb=peak,
            alphabet_added=max(0, added),
        )
        del trainer, model, data_module
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return outcome
