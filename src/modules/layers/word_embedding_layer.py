import torch
import torch.nn as nn

from src.utils.mapper import ConfigMapper


class WordEmbeddingLayer(nn.Module):
    """
    A Word Embedding Layer. This layer loads a pre-trained word embedding matrix
    , and copies its weights to an nn.Embedding layer.

    Args:
        embed_dir (str): A directory containing the pre-trained word embedding
                         matrix, among other things. Please see
                         https://github.com/dalgu90/icd-coding-benchmark/blob/main/src/modules/embeddings.py#L17
                         for more details.
        dropout (float): The dropout probability.
    """

    def __init__(self, embed_dir, dropout):
        super(WordEmbeddingLayer, self).__init__()
        # Note: This should be changed, since we won't always use Word2Vec.
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")

        W = torch.Tensor(embedding_cls.load_emb_matrix(embed_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.embedding_size = self.embed.embedding_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.embed(x)
        x = self.dropout(embedding)
        return x
