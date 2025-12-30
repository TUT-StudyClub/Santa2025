from __future__ import annotations

import logging
import time
from logging import FileHandler, StreamHandler
from pathlib import Path


def get_logger(name: str, log_dir: Path | str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"{time.strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s : %(levelname)s - %(filename)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
