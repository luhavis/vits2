import sys
from vits2.utils import loguru_logger

loguru_logger.info(f"python version: {sys.version}")
for path in sys.path:
    loguru_logger.info(path)