from __future__ import annotations

import logging
import sys


def build_logger(date: str) -> logging.Logger:
    logger = logging.getLogger("hw2")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(f"log-{date}.txt", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
