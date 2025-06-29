import logging

__all__ = ["get_logger"]


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger with the specified name.

    Parameters
    ----------
    name : str, optional
        The name of the logger. Defaults to the module's name.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
