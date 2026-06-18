"""Penny training entry point (spec sections 7, 11, 12).

Usage::

    uv run python scripts/train_penny.py [configs/config.json]

Loads the config, builds datasets (calibrating alpha + gamma), constructs the
UNet and trend head, runs the training loop with early stopping on validation
diffusion loss, and saves the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penny import evaluate, train as train_mod  # noqa: E402
from penny.dataset import build_datasets  # noqa: E402
from penny.diffusion import Diffusion  # noqa: E402
from penny.model import TrendHead, build_unet, count_parameters  # noqa: E402


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.warning("cuda requested but unavailable; using mps")
            return torch.device("mps")
        logger.warning("cuda requested but unavailable; using cpu")
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("mps requested but unavailable; using cpu")
        return torch.device("cpu")
    return torch.device(requested)


def print_summary(config, meta, gamma, alpha, n_params, device, ckpt_dir) -> None:
    cb = meta["class_balance"]
    iv = config["snapshot_interval_sec"]
    total = meta["total_snapshots"]
    approx_days = total * iv / 86400
    tf, vf = config["train_frac"], config["val_frac"]
    f = config["unet_filters"]
    logger.info("Penny — LOB inpainting diffusion with trend loss")
    logger.info("  feature_mode      : {}", config["feature_mode"])
    logger.info(
        "  exchange          : {}  |  pair: {}", config["exchange"], config["pair"]
    )
    logger.info(
        "  snapshot_interval : {}s  |  total: {:,} snapshots (~{:.1f} days)",
        iv,
        total,
        approx_days,
    )
    logger.info(
        "  window            : T_past={} ({} min)  T_future={} ({} min)",
        config["T_past"],
        round(config["T_past"] * iv / 60),
        config["T_future"],
        round(config["T_future"] * iv / 60),
    )
    logger.info(
        "  stride            : {}  ({}-min step between windows)",
        config["stride"],
        round(config["stride"] * iv / 60),
    )
    logger.info(
        "  splits            : {:.0%}/{:.0%}/{:.0%}  →  train={} val={} test={}",
        tf,
        vf,
        1 - tf - vf,
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )
    logger.info(
        "  label_k           : {}  ({}s smoothing each side)",
        config["label_k"],
        config["label_k"] * iv,
    )
    logger.info("  label_alpha       : {:.6f}", alpha)
    logger.info(
        "  class balance     : down={:.1%} stationary={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )
    if config["feature_mode"] == "ofi":
        logger.info("  ofi gamma         : {:.6g}", gamma)
    logger.info(
        "  model             : UNet2DModel  {} blocks  {}",
        len(f),
        "-".join(map(str, f)),
    )
    logger.info(
        "  params            : ~{:.0f}M (UNet) + 6 (trend head)", n_params / 1e6
    )
    logger.info("  dropout           : {}", config["dropout"])
    logger.info("  lambda_trend      : {}", config["lambda_trend"])
    logger.info("  device            : {}", device)
    logger.info("  checkpoint dir    : {}/", ckpt_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Penny inpainting diffusion.")
    parser.add_argument("config", nargs="?", default="configs/config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(config["checkpoint_root"]) / (
        f"{config['feature_mode']}_{config['exchange']}_{config['pair']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, normalizer, gamma, alpha, meta = build_datasets(config)
    meta["total_snapshots"] = int(train_ds.rows.shape[0])
    level_starts = meta["level_starts"]

    diffusion = Diffusion(config, device)
    unet = build_unet(config).to(device)
    trend_head = TrendHead().to(device)

    print_summary(config, meta, gamma, alpha, count_parameters(unet), device, ckpt_dir)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    total_steps = max(config["epochs"] * len(train_loader), 1)
    optimizer, scheduler = train_mod.build_optimizer_scheduler(
        unet, trend_head, config, total_steps
    )

    best_val = float("inf")
    patience = 0
    history = []
    for epoch in range(config["epochs"]):
        tr = train_mod.train_one_epoch(
            unet,
            trend_head,
            diffusion,
            train_loader,
            optimizer,
            scheduler,
            config,
            normalizer,
            level_starts,
            gamma,
            device,
        )
        val_diff = train_mod.validate_diffusion(
            unet, diffusion, val_loader, config, device
        )
        val_acc = train_mod.validate_label_accuracy(
            unet,
            diffusion,
            val_ds,
            config,
            normalizer,
            level_starts,
            gamma,
            alpha,
            device,
            config["val_eval_windows"],
            seed=epoch,
        )
        logger.info(
            "epoch {} | train total={:.5f} diff={:.5f} trend={:.5f} | val diff={:.5f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["diff"],
            tr["trend"],
            val_diff,
            val_acc["accuracy"],
        )
        history.append(
            {"epoch": epoch, **tr, "val_diff": val_diff, "val_acc": val_acc["accuracy"]}
        )

        if val_diff < best_val:
            best_val = val_diff
            patience = 0
            torch.save(
                {
                    "unet": unet.state_dict(),
                    "trend_head": trend_head.state_dict(),
                    "config": config,
                    "normalizer": normalizer.to_dict(),
                    "gamma": gamma,
                    "alpha": alpha,
                    "label_k": config["label_k"],
                    "level_starts": level_starts,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info(
                "saved checkpoint at epoch {} (val diff {:.5f})", epoch, best_val
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info(
                    "early stopping at epoch {} (best val diff {:.5f})", epoch, best_val
                )
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    np.savez(ckpt_dir / "normalizer.npz", **normalizer.to_dict())

    logger.info("loading best checkpoint for test evaluation")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet"])
    evaluate.run_test(
        unet, diffusion, test_ds, config, normalizer, level_starts, gamma, alpha, device
    )


if __name__ == "__main__":
    main()
