import logging
import sys

stdout_logger = logging.getstdout_logger(__name__)
stdout_logger.setLevel(logging.INFO)
file_hander = logging.StreamHandler(sys.stdout)
file_hander.setFormatter(
    logging.Formatter("\n%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
stdout_logger.addHandler(file_hander)


datasets_logger = logging.getLogger(__name__)
datasets_logger.setLevel(logging.DEBUG)
file_hander = logging.FileHandler("logs/dataset.log")
file_hander.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
datasets_logger.addHandler(file_hander)
