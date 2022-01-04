"""Contains various kinds of embeddings like Glove, BERT, etc."""

from torch.nn import Module, Embedding, Flatten
from src.utils.mapper import configmapper


@configmapper.map("embeddings", "glove")
class GloveEmbedding(Module):
    """Implement Glove based Word Embedding."""

    def __init__(self, embedding_matrix, padding_idx, static=True):
        """Construct GloveEmbedding.

        Args:
            embedding_matrix (torch.Tensor): The matrix contrainining the embedding weights
            padding_idx (int): The padding index in the tokenizer.
            static (bool): Whether or not to freeze embeddings.
        """
        super(GloveEmbedding, self).__init__()
        self.embedding = Embedding.from_pretrained(embedding_matrix)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.required_grad = False
        self.flatten = Flatten(start_dim=1)

    def forward(self, x_input):
        """Pass the input through the embedding.

        Args:
            x_input (torch.Tensor): The numericalized tokenized input

        Returns:
            x_output (torch.Tensor): The output from the embedding
        """
        x_output = self.embedding(x_input)
        x_output = self.flatten(x_output)
        return x_output
