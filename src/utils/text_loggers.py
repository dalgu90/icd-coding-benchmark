import logging
import sys

logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs/logs.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(
    logging.Formatter("\n%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)
