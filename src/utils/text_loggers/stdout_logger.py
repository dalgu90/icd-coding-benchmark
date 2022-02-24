import logging
import sys

stdout_logger = logging.getLogger(__name__)
stdout_logger.setLevel(logging.INFO)
file_hander = logging.StreamHandler(sys.stdout)
file_hander.setFormatter(
    logging.Formatter("\n%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
stdout_logger.addHandler(file_hander)
