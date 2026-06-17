"""Training pipeline for Penny.

Configures logging, builds the diffusion process / datasets / model, trains
with AdamW + warmup-cosine scheduling and early stopping, saves the best
checkpoint, then runs the full test suite.

Entry point: ``scripts/train_penny.py``.
"""

from __future__ import annotations

import logging
import math
import os

from datetime import datetime

import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import evaluate
from .dataset import build_datasets
from .diffusion import Diffusion
from .model import Penny, count_parameters

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> None:
    """Configure the root logger to write to stdout and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"penny_{stamp}.log")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    logger.info("logging to %s", path)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device, warning and falling back if CUDA is absent."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("config requested cuda but no GPU is available; using cpu")
        return torch.device("cpu")
    return torch.device(requested)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup followed by cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def train(config: dict) -> None:
    """Run the full training + evaluation pipeline."""
    device = resolve_device(config["device"])
    logger.info("device: %s", device)

    diffusion = Diffusion(config, device)
    train_ds, val_ds, test_ds, columns, trend_sigma = build_datasets(
        config, diffusion, device
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    model = Penny(config).to(device)
    logger.info("model parameter count: %d", count_parameters(model))

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    total_steps = max(config["epochs"] * len(train_loader), 1)
    scheduler = build_scheduler(optimizer, config["warmup_steps"], total_steps)

    best_val = float("inf")
    patience = 0
    ckpt_dir = os.path.dirname(config["checkpoint_path"])
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            total, diff_loss, trend_loss, _, _ = evaluate.compute_losses(
                model, diffusion, batch, config
            )
            if torch.isnan(total):
                logger.warning(
                    "NaN loss at epoch %d step %d; skipping batch", epoch, step
                )
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += total.item()
            logger.debug(
                "epoch %d step %d total=%.5f diff=%.5f trend=%.5f lr=%.2e",
                epoch,
                step,
                total.item(),
                diff_loss.item(),
                trend_loss.item(),
                scheduler.get_last_lr()[0],
            )

        train_loss = running / max(len(train_loader), 1)
        val = evaluate.run_validation(model, diffusion, val_loader, config)
        logger.info(
            "epoch %d | train=%.5f | val total=%.5f diff=%.5f trend=%.5f "
            "| acc=%.4f f1=%.4f kappa=%.4f",
            epoch,
            train_loss,
            val["total_loss"],
            val["diff_loss"],
            val["trend_loss"],
            val["accuracy"],
            val["f1_macro"],
            val["kappa"],
        )
        if "confusion_matrix" in val:
            logger.info(
                "epoch %d val confusion matrix:\n%s", epoch, val["confusion_matrix"]
            )

        if val["total_loss"] < best_val:
            best_val = val["total_loss"]
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                config["checkpoint_path"],
            )
            logger.info(
                "saved checkpoint at epoch %d with val loss %.5f", epoch, best_val
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info(
                    "early stopping at epoch %d (best val %.5f)", epoch, best_val
                )
                break

    logger.info("loading best checkpoint for final test evaluation")
    ckpt = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    evaluate.run_test(model, diffusion, test_loader, config, columns, trend_sigma)
