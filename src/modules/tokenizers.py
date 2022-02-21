"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""
import logging
import sys

from src.utils.mapper import ConfigMapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_hander = logging.FileHandler("logs/dataset.log")
file_hander.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
logger.addHandler(file_hander)


@ConfigMapper.map("tokenizers", "spacetokenizer")
class SpaceTokenizer:
    def __init__(self, config):
        if config:
            logger.info(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config.as_dict())
            )
        else:
            logger.info(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config)
            )
        self._config = config

    def tokenize_list(self, lst):
        lst = [self.tokenize(text) for text in lst]
        return lst

    def tokenize(self, text):
        return text.split(" ")
