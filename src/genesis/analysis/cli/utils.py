import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name and a standardized configuration."""
    log = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)
    return log


log = get_logger(__name__)
