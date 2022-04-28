from typing import Optional
from utils.resource_util import get_repo_dir
from datetime import datetime
from logging import Logger
import logging
import os


def get_logger(logs_dir: Optional[str] = None) -> Logger:
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)
    handler_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not logger.hasHandlers():
        # console logging
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(handler_formatter)
        logger.addHandler(handler)

        # file logging
        log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = logs_dir or os.path.join(get_repo_dir(), "logs")
        log_filepath = os.path.join(logs_dir, f"{log_filename}.log")
        handler = logging.FileHandler(log_filepath, "w")
        handler.setLevel(logging.INFO)
        handler.setFormatter(handler_formatter)
        logger.addHandler(handler)

    return logger
