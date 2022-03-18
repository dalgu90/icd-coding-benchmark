import torch
import torch.nn as nn

from src.utils.caml_utils import load_embeddings
from src.utils.mapper import ConfigMapper


class WordEmbeddingLayer(nn.Module):
    def __init__(self, embed_dir, dropout):
        super(WordEmbeddingLayer, self).__init__()
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")

        W = torch.Tensor(
            load_embeddings(embedding_cls.load_emb_matrix(embed_dir))
        )
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.embedding_size = self.embed.embedding_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.embed(x)
        x = self.dropout(embedding)
        return x
