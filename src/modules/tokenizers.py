"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""

from src.utils.mapper import ConfigMapper


@ConfigMapper.map("tokenizers", "spacetokenizer")
class SpaceTokenizer:
    def __init__(self, config):
        self._config = config

    def tokenize_list(self, lst):
        lst = [self.tokenize(text) for text in lst]
        return lst

    def tokenize(self, text):
        return text.split(" ")
