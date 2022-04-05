from typing import Optional
from utils.resource_util import get_repo_dir
from datetime import datetime
from logging import Logger
import logging
import os


def get_logger(logger_name: Optional[str] = None, log_filename: Optional[str] = None) -> Logger:
    logger_name = logger_name or "ds"
    log_filename = log_filename or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not logger.hasHandlers():
        # console logging
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(handler_formatter)
        logger.addHandler(handler)

        # file logging
        log_dir = os.path.join(get_repo_dir(), "log")
        log_filepath = os.path.join(log_dir, f"{log_filename}.log")
        handler = logging.FileHandler(log_filepath, "w")
        handler.setLevel(logging.INFO)
        handler.setFormatter(handler_formatter)
        logger.addHandler(handler)

    return logger
