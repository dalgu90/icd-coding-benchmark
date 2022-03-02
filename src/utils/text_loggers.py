import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("logs/log.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(module)s: %(message)s")
)
logger.addHandler(file_handler)


logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(
    logging.Formatter("\n%(asctime)s:%(levelname)s:%(module)s: %(message)s")
)
logger.addHandler(stdout_handler)
