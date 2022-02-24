import logging
import sys

datasets_logger = logging.getLogger(__name__)
datasets_logger.setLevel(logging.DEBUG)
file_hander = logging.FileHandler("logs/dataset.log")
file_hander.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
datasets_logger.addHandler(file_hander)
