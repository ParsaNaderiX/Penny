"""Train ScoreJacobLOB on a single crypto symbol (two-phase).

Phase 1  Diffusion pretraining — BiN + 1-D U-Net backbone trained with the
         v-parameterisation + min-SNR loss.  No classification signal.
Phase 2  Classifier fine-tuning — backbone frozen; score-Jacobian attention
         saliencies + bottleneck are extracted once per sample (cached), then
         only the gated head is trained with cross-entropy.  Best checkpoint
         selected by validation macro-F1.

Usage::

    uv run python -m crypto.train_scorejacoblob configs/crypto/binance/scorejacoblob/btcusdt_ofi.json
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
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from crypto.dataset import build_datasets
from models.scorejacoblob import ScoreJacobLOB, count_parameters
from utils.evaluate import run_test
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


# ── cached score-Jacobian features ────────────────────────────────────────────


class CachedFeatureDataset(Dataset):
    """Holds precomputed ``(w_t, w_f, h, label)`` so phase 2 trains fast."""

    def __init__(self, wt, wf, h, labels) -> None:
        self.wt, self.wf, self.h, self.labels = wt, wf, h, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "wt": self.wt[i],
            "wf": self.wf[i],
            "h": self.h[i],
            "label": int(self.labels[i]),
        }


@torch.no_grad()
def _precompute(model, dataset, config, device) -> CachedFeatureDataset:
    """Extract score-Jacobian saliencies + bottleneck for every sample once."""
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    wts, wfs, hs, labels = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device).float()
        wt, wf, h = model.compute_features(x)  # enable_grad handled internally
        wts.append(wt.cpu())
        wfs.append(wf.cpu())
        hs.append(h.half().cpu())  # bottleneck stored fp16 to save memory
        labels.extend(batch["label"].tolist())
    return CachedFeatureDataset(
        torch.cat(wts), torch.cat(wfs), torch.cat(hs), torch.tensor(labels)
    )


# ── phase 1: diffusion pretraining ────────────────────────────────────────────


def _pretrain_epoch(model, loader, optimizer, scheduler, device, t_max, grad_clip):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x0 = batch["x"].to(device).float().squeeze(1)  # (B, T, F)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(x0)
        loss = model.diffusion_loss(x0, t, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _val_diffusion(model, loader, device, t_max):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x0 = batch["x"].to(device).float().squeeze(1)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(x0)
        total += model.diffusion_loss(x0, t, noise).item()
        n += 1
    return total / max(n, 1)


# ── phase 2: head fine-tuning ─────────────────────────────────────────────────


def _finetune_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.head_logits(
            batch["wt"].to(device).float(),
            batch["wf"].to(device).float(),
            batch["h"].to(device).float(),
        )
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _val_f1(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits = model.predict(batch, device)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    acc = float((torch.tensor(y_true) == torch.tensor(y_pred)).float().mean())
    return f1, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/binance/scorejacoblob/btcusdt_ofi.json",
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
    grad_clip = config.get("grad_clip", 1.0)
    t_max = config.get("T_max", 1000)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"scorejacoblob_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    model = ScoreJacobLOB(config).to(device)
    logger.info(
        "ScoreJacobLOB  symbol={}  mode={}",
        config["symbol"],
        config.get("feature_mode"),
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  alpha={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )
    logger.info(
        "  features={}  params={:.2f}M  t_star={}  K={}",
        meta["n_features"],
        count_parameters(model) / 1e6,
        model.t_star,
        len(model.t_star),
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

    history = {"pretrain": [], "finetune": []}

    # ── Phase 1: diffusion pretraining ───────────────────────────────────────
    pre_epochs = config.get("sjl_pretrain_epochs", 60)
    pre_patience = config.get("sjl_pretrain_patience", config.get("patience", 15))
    optim1 = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    sched1 = build_cosine_schedule(optim1, config, pre_epochs * len(train_loader))
    logger.info("── phase 1: diffusion pretraining ({} epochs) ──", pre_epochs)
    best, patience = float("inf"), 0
    for epoch in range(pre_epochs):
        tr = _pretrain_epoch(
            model, train_loader, optim1, sched1, device, t_max, grad_clip
        )
        vl = _val_diffusion(model, val_loader, device, t_max)
        logger.info("  [pre] epoch {} | train={:.4f} val={:.4f}", epoch, tr, vl)
        history["pretrain"].append({"epoch": epoch, "train": tr, "val": vl})
        if vl < best:
            best, patience = vl, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                ckpt_dir / "backbone_best.pt",
            )
        else:
            patience += 1
            if patience >= pre_patience:
                logger.info("  [pre] early stop at epoch {}", epoch)
                break

    ckpt = torch.load(
        ckpt_dir / "backbone_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model"])
    model.freeze_backbone()

    # ── precompute score-Jacobian features (once) ────────────────────────────
    logger.info("── extracting score-Jacobian attention features ──")
    cached_train = _precompute(model, train_ds, config, device)
    cached_val = _precompute(model, val_ds, config, device)
    cached_test = _precompute(model, test_ds, config, device)
    logger.info(
        "  cached train={} val={} test={}",
        len(cached_train),
        len(cached_val),
        len(cached_test),
    )

    ft_train_loader = DataLoader(
        cached_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    ft_val_loader = DataLoader(
        cached_val, batch_size=config["batch_size"], shuffle=False
    )

    # ── Phase 2: head fine-tuning ────────────────────────────────────────────
    ft_epochs = config.get("sjl_finetune_epochs", 40)
    ft_patience = config.get("sjl_finetune_patience", config.get("patience", 15))
    ft_lr = config.get("sjl_finetune_lr", config["lr"])
    head_params = [p for p in model.parameters() if p.requires_grad]
    optim2 = AdamW(head_params, lr=ft_lr, weight_decay=config["weight_decay"])
    sched2 = build_cosine_schedule(optim2, config, ft_epochs * len(ft_train_loader))
    logger.info(
        "── phase 2: head fine-tuning ({} epochs, {} trainable params) ──",
        ft_epochs,
        sum(p.numel() for p in head_params),
    )
    best_f1, patience = -1.0, 0
    for epoch in range(ft_epochs):
        tr = _finetune_epoch(model, ft_train_loader, optim2, sched2, device, grad_clip)
        val_f1, val_acc = _val_f1(model, ft_val_loader, device)
        logger.info(
            "  [ft] epoch {} | ce={:.4f} val_f1={:.4f} val_acc={:.4f}",
            epoch,
            tr,
            val_f1,
            val_acc,
        )
        history["finetune"].append(
            {"epoch": epoch, "ce": tr, "val_f1": val_f1, "val_acc": val_acc}
        )
        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
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
            if patience >= ft_patience:
                logger.info("  [ft] early stop at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("best val macro-F1 = {:.4f}", best_f1)
    run_test(model, cached_test, config, device)


if __name__ == "__main__":
    main()
