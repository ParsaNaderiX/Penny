"""Train StableLOB: improved-DDPM joint diffusion-classifier (feature-only inference).

Same joint structure as ``crypto.train_jumpgatelob`` — a shared trunk trained on two
**separate passes** so the trend head always sees the clean-window distribution it
sees at inference — but the generative branch uses the **improved-DDPM** noise/denoise
recipe (cosine schedule + learned variance + hybrid loss; :mod:`models.iddpm`) instead
of the Lévy ε-prediction:

    L_cls    = CE(classify(x0), label)                 # clean pass, t = 0
    L_hybrid = L_simple + lambda_vlb * L_vlb            # noised pass, sampled t
             = MSE(ε̂, ε)  +  λ_vlb · VLB(ε̂.detach, v̂)  # v̂ trains the reverse variance
    L        = L_cls + lambda_diff * L_hybrid

The diffusion head predicts ``(ε̂, v̂)``; ``L_simple`` trains the mean (ε) and the VLB
trains only the variance (``v̂``) via the stop-grad-on-mean trick.  Model selection and
early stopping are on **trend-head macro-F1** (feature-only), not the diffusion loss;
train and val F1 are both logged so the noise-fitting gap is visible.

Modes:
  * default    — joint (both losses each step).
  * --baseline — plain classifier: ``L_cls`` only, no diffusion head.

Usage::

    uv run python -m crypto.train_stablelob configs/crypto/nobitex/stablelob/btcirt_ofi_k10.json
    uv run python -m crypto.train_stablelob ... --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.iddpm import ImprovedDiffusion
from models.stablelob import StableLOB, count_parameters
from utils.evaluate import run_test
from utils.flops import log_gflops
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


def _train_epoch(model, diff, loader, optimizer, lr_sched, config, device, do_diff):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_diff = config.get("lambda_diff", 1.0)
    lam_vlb = config.get("lambda_vlb", 1.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    t_max = diff.num_timesteps

    tot = clsm = difm = vlbm = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]

        # clean pass — trend head sees exactly what inference sees
        logits = model.classify(x0)
        cls_loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
        loss = cls_loss
        diff_loss = torch.zeros((), device=device)
        parts = {"simple": 0.0, "vlb": 0.0}

        if do_diff:
            t = torch.randint(0, t_max, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = diff.q_sample(x0, t, noise)
            eps_hat, v_hat = model.diffuse(x_t, t)
            diff_loss, parts = diff.hybrid_loss(
                eps_hat, v_hat, x0, x_t, t, noise, lambda_vlb=lam_vlb
            )
            loss = loss + lam_diff * diff_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()
        lr_sched.step()

        tot += loss.item()
        clsm += cls_loss.item()
        difm += diff_loss.item()
        vlbm += parts["vlb"]
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "cls": clsm / n, "diff": difm / n, "vlb": vlbm / n}


@torch.no_grad()
def _f1_ce_acc(model, loader, device, max_batches=None):
    """Feature-only macro-F1 / CE / accuracy (F1 drives selection)."""
    model.eval()
    ce, n = 0.0, 0
    y_true, y_pred = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        ce += F.cross_entropy(logits, label).item()
        y_true.extend(label.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        n += len(label)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(n, 1)
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    return f1, ce / max(n, 1), acc


@torch.no_grad()
def _per_class_report(model, dataset, config, device) -> dict:
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_true, y_pred = [], []
    for batch in loader:
        logits = model.predict(batch, device)
        y_true.extend(batch["label"].tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    logger.info(
        "TEST per-class P/R/F1:\n{}",
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["down", "stationary", "up"],
            zero_division=0,
        ),
    )
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "stationary", "up"],
        zero_division=0,
        output_dict=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/nobitex/stablelob/btcirt_ofi_k10.json",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="plain classifier: L_cls only, no diffusion head",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "baseline" if args.baseline else "joint"
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"stablelob_{mode}_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info("StableLOB  symbol={} mode={} schedule=cosine", config["symbol"], mode)
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  label_thr={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = StableLOB(config).to(device)
    diff = ImprovedDiffusion(
        num_timesteps=config.get("T_max", 1000),
        cosine_s=config.get("cosine_s", 0.008),
    )
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  lambda_diff={} lambda_vlb={}  device={}",
        count_parameters(model) / 1e6,
        log_gflops(model, train_ds, device),
        config.get("lambda_diff", 1.0),
        config.get("lambda_vlb", 1.0),
        device,
    )

    nw = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    train_eval_batches = max(1, len(val_loader))

    epochs = config["epochs"]
    do_diff = not args.baseline
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    lr_sched = build_cosine_schedule(optimizer, config, epochs * len(train_loader))

    best, patience, history = float("-inf"), 0, []
    for epoch in range(epochs):
        tr = _train_epoch(
            model, diff, train_loader, optimizer, lr_sched, config, device, do_diff
        )
        val_f1, val_ce, val_acc = _f1_ce_acc(model, val_loader, device)
        train_f1, _, _ = _f1_ce_acc(model, train_loader, device, train_eval_batches)
        row = {
            "epoch": epoch,
            **tr,
            "val_f1": val_f1,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "f1_gap": train_f1 - val_f1,
        }
        logger.info(
            "ep {} | cls={:.4f} diff={:.4f} vlb={:.4f}"
            " | val_f1={:.4f} acc={:.4f} | train_f1={:.4f} gap={:+.4f}",
            epoch,
            tr["cls"],
            tr["diff"],
            tr["vlb"],
            val_f1,
            val_acc,
            train_f1,
            train_f1 - val_f1,
        )
        history.append(row)

        if val_f1 > best:
            best, patience = val_f1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    metrics = run_test(model, test_ds, config, device)
    report = _per_class_report(model, test_ds, config, device)
    (ckpt_dir / "metrics.json").write_text(
        json.dumps({"test": metrics, "per_class": report}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
