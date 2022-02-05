"""Contains various kinds of embeddings like Glove, BERT, etc."""

import os

import gensim
import numpy as np

from src.utils.file_loaders import load_json, save_json
from src.utils.mapper import ConfigMapper


@ConfigMapper.map("embeddings", "word2vec")
class Word2VecEmbedding:
    def __init__(self, config):
        self._config = config

        if not os.path.exists(self._config.embedding_dir):
            os.makedirs(self._config.embedding_dir)

    def train(self, corpus):
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            corpus, **self._config.word2vec_params.as_dict()
        )
        model.wv.save(
            os.path.join(self._config.embedding_dir, "word2vec.wordvectors")
        )

        token_to_idx_dict = model.wv.key_to_index
        token_to_idx_dict[self._config.unk_token] = len(token_to_idx_dict)
        token_to_idx_dict[self._config.pad_token] = len(token_to_idx_dict)

        save_json(
            token_to_idx_dict,
            os.path.join(self._config.embedding_dir, "token_to_idx.json"),
        )

        embedding_matrix = model.wv.get_normed_vectors()
        unk_emb = np.expand_dims(np.mean(embedding_matrix, axis=0), axis=0)
        pad_emb = np.zeros((1, embedding_matrix.shape[1]))
        embedding_matrix = np.concatenate(
            (embedding_matrix, unk_emb, pad_emb), axis=0
        )
        np.save(
            os.path.join(self._config.embedding_dir, "embedding_matrix.npy"),
            model.wv.get_normed_vectors(),
        )

    def load_vocab_emb_matrix(self, dir_path):
        vocab = load_json(os.path.join(dir_path, "token_to_idx.json"))
        embedding_matrix = np.load(
            os.path.join(dir_path, "embedding_matrix.npy")
        )
        return (vocab, embedding_matrix)
