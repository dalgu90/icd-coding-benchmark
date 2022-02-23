"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""
import logging
import sys

from src.utils.mapper import ConfigMapper
from src.utils.text_logger import datasets_logger


@ConfigMapper.map("tokenizers", "spacetokenizer")
class SpaceTokenizer:
    def __init__(self, config):
        if config:
            datasets_logger.info(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config.as_dict())
            )
        else:
            datasets_logger.info(
                "Using Space Tokenizer to tokenize the data with the following "
                "config: {}".format(config)
            )
        self._config = config

    def tokenize_list(self, lst):
        lst = [self.tokenize(text) for text in lst]
        return lst

    def tokenize(self, text):
        return text.split(" ")
