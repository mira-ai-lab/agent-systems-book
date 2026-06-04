from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_executor_logger(env_prefix: str, logger_name: str) -> logging.Logger:
    """
    Logger for A2A executor request tracing (aligned with HR agent).

    Reads:
      - {env_prefix}_LOG_LEVEL (default INFO), e.g. SWE_AGENT_LOG_LEVEL
      - {env_prefix}_LOG_DIR (optional), e.g. SWE_AGENT_LOG_DIR
    """
    level_name = (os.getenv(f"{env_prefix}_LOG_LEVEL") or "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = os.getenv(f"{env_prefix}_LOG_DIR")
    return get_agent_logger(logger_name, level=level, log_dir=log_dir or None)


def get_agent_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create a per-agent logger that logs to both console and a rotating file:
      <project_root>/logs/<name>.log

    Safe to call multiple times; handlers are added only once per logger name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Resolve project root as: <project_root>/agents/logging_utils.py
    agents_dir = Path(__file__).resolve().parent
    project_root = agents_dir.parent
    out_dir = Path(log_dir) if log_dir else (project_root / "logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = out_dir / f"{name}.log"
    fh = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Reduce noisy libraries unless user explicitly configures otherwise
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))

    return logger

