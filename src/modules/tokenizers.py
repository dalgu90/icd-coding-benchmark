"""Contains tokenizers like GloveTokenizers and BERT Tokenizer."""

import torch
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset
from src.utils.mapper import configmapper
from transformers import AutoTokenizer


class Tokenizer:
    """Abstract Class for Tokenizers."""

    def tokenize(self):
        """Abstract Method for tokenization."""


@configmapper.map("tokenizers", "glove")
class GloveTokenizer(Tokenizer):
    """Implement GloveTokenizer for tokenizing text for Glove Embeddings.

    Attributes:
        embeddings (torchtext.vocab.Vectors): Loaded pre-trained embeddings.
        text_field (torchtext.data.Field): Text_field for vector creation.

    Methods:
        __init__(self, name='840B', dim='300', cache='../embeddings/') : Constructor method
        initialize_vectors(fix_length=4, tokenize='spacy', file_path="../data/imperceptibility
                           /Concreteness Ratings/train/forty.csv",
                           file_format='tsv', fields=None): Initialize vocab vectors based on data.

        tokenize(x_input, **initializer_params): Tokenize given input and return the output.
    """

    def __init__(self, name="840B", dim="300", cache="../embeddings/"):
        """Construct GloveTokenizer.

        Args:
            name (str): Name of the GloVe embedding file
            dim (str): Dimensions of the Glove embedding file
            cache (str): Path to the embeddings directory
        """
        super(GloveTokenizer, self).__init__()
        self.embeddings = GloVe(name=name, dim=dim, cache=cache)
        self.text_field = None

    def initialize_vectors(
        self,
        fix_length=4,
        tokenize="spacy",
        tokenizer_file_paths=None,
        file_format="tsv",
        fields=None,
    ):
        """Initialize words/sequences based on GloVe embedding.

        Args:
            fields (list): The list containing the fields to be taken
                                     and processed from the file (see documentation for
                                      torchtext.data.TabularDataset)
            fix_length (int): The length of the tokenized text,
                              padding or cropping is done accordingly
            tokenize (function or string): Method to tokenize the data.
                                           If 'spacy' uses spacy tokenizer,
                                           else the specified method.
            tokenizer_file_paths (list of str): The paths of the files containing the data
            format (str): The format of the file : 'csv', 'tsv' or 'json'
        """
        text_field = Field(batch_first=True, fix_length=fix_length, tokenize=tokenize)
        tab_dats = [
            TabularDataset(
                i, format=file_format, fields={k: (k, text_field) for k in fields}
            )
            for i in tokenizer_file_paths
        ]
        text_field.build_vocab(*tab_dats)
        text_field.vocab.load_vectors(self.embeddings)
        self.text_field = text_field

    def tokenize(self, x_input, **init_vector__params):
        """Tokenize given input based on initialized vectors.

        Initialize the vectors with given parameters if not already initialized.

        Args:
            x_input (str): Unprocessed input text to be tokenized
            **initializer_params (Keyword arguments): Parameters to initialize vectors

        Returns:
            x_output (str): Processed and tokenized text
        """
        if self.text_field is None:
            self.initialize_vectors(**init_vector__params)
        try:
            x_output = torch.squeeze(
                self.text_field.process([self.text_field.preprocess(x_input)])
            )
        except Exception as e:
            print(x_input)
            print(self.text_field.preprocess(x_input))
            print(e)
        return x_output


@configmapper.map("tokenizers", "AutoTokenizer")
class AutoTokenizer(AutoTokenizer):
    def __init__(self, *args):
        super(AutoTokenizer, self).__init__()
