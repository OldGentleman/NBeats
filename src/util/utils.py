from loguru import logger
from src.util.paths import Paths


def setup_logger() -> None:
    logger.add(Paths().logs_filepath, rotation="5 MB")
