"""Training entry point for Penny.

Usage::

    uv run python scripts/train_penny.py [configs/config.json]

``configs/config.json`` is used by default when not specified.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from penny.train import setup_logging, train

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments, load config, configure logging and launch training."""
    parser = argparse.ArgumentParser(
        description="Train the Penny LOB diffusion model."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/config.json",
        help="Path to config.json (default: configs/config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        config = json.load(f)

    setup_logging(config["log_dir"])
    logger.info("loaded config from %s", config_path)

    try:
        train(config)
    except Exception:
        logger.exception("unrecoverable error during training")
        raise


if __name__ == "__main__":
    main()
