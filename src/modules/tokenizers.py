"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""

import torch
from torchtext.data import Field, TabularDataset
from torchtext.vocab import GloVe
from transformers import AutoTokenizer

from src.utils.mapper import ConfigMapper


@ConfigMapper.map("tokenizers", "spacetokenizer")
class SpaceTokenizer:
    def __init__(self, config):
        self._config = config

    def tokenize_dict(self, input_dataset, attr_name):
        for sample in input_dataset:
            sample[attr_name] = self.tokenize(sample[attr_name])
        return input_dataset

    def tokenize(self, text):
        return text.split(" ")
