import logging
from typing import Any

LOGGER_NAME = "linear_relational"

logger = logging.getLogger(LOGGER_NAME)


def log_or_print(msg: Any, verbose: bool, level: int = logging.INFO) -> None:
    if verbose:
        print(msg)
    else:
        logger.log(level, msg)
