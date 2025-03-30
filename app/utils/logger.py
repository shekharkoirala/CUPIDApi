import logging
from typing import Optional


class CustomLogger:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            raise RuntimeError(
                "Logger has not been initialized. Call setup_logger first."
            )
        return cls._instance

    @classmethod
    def setup_logger(cls, log_level: str) -> logging.Logger:
        if cls._instance is not None:
            return cls._instance

        logger = logging.getLogger("cupid-api")
        logger.setLevel(log_level.upper())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level.upper())
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        cls._instance = logger
        return logger
