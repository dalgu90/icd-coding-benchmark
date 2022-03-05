"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""
import logging
import sys

from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("tokenizers", "spacetokenizer")
class SpaceTokenizer:
    def __init__(self, config):
        if config:
            logger.debug(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config.as_dict())
            )
        else:
            logger.debug(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config)
            )
        self._config = config

    def tokenize_list(self, lst):
        lst = [self.tokenize(text) for text in lst]
        return lst

    def tokenize(self, text):
        return text.split(" ")
