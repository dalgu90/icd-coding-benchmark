"""Contains various kinds of embeddings like Glove, BERT, etc."""
import logging
import os
import sys

import gensim
import numpy as np

from anemic.utils.file_loaders import load_json, save_json
from anemic.utils.mapper import ConfigMapper
from anemic.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("embeddings", "word2vec")
class Word2VecEmbedding:
    def __init__(self, config):
        logger.debug(
            "Using Word2Vec to train embeddings on clinical notes with the "
            "following config: {}".format(config.as_dict())
        )
        self._config = config

        if not os.path.exists(self._config.embedding_dir):
            os.makedirs(self._config.embedding_dir)

    def train(self, corpus):
        logger.debug("Training Word2Vec on clinical notes")
        # build vocabulary and train model
        # Resulting model doesn't have <pad> and <unk> which will be added
        model = gensim.models.Word2Vec(
            corpus, **self._config.word2vec_params.as_dict()
        )
        model.wv.save(
            os.path.join(self._config.embedding_dir, "word2vec.wordvectors")
        )

        # Vocab: {<pad>: 0, <unk>: 1, word1: 2, word2: 3, ... }
        words = [
            self._config.pad_token,
            self._config.unk_token,
        ] + model.wv.index_to_key
        token_to_idx_dict = {token: idx for idx, token in enumerate(words)}

        save_json(
            token_to_idx_dict,
            os.path.join(self._config.embedding_dir, "token_to_idx.json"),
        )

        # Add <PAD> and <UNK> to the embedding matrix
        embedding_matrix = model.wv.get_normed_vectors()
        unk_emb = np.expand_dims(np.mean(embedding_matrix, axis=0), axis=0)
        pad_emb = np.zeros((1, embedding_matrix.shape[1]))
        embedding_matrix = np.concatenate(
            (pad_emb, unk_emb, embedding_matrix), axis=0
        )
        np.save(
            os.path.join(self._config.embedding_dir, "embedding_matrix.npy"),
            embedding_matrix,
        )

    @staticmethod
    def load_vocab(dir_path):
        logger.debug("Loading Word2Vec vocab from {}".format(dir_path))
        vocab = load_json(os.path.join(dir_path, "token_to_idx.json"))
        return vocab

    @staticmethod
    def load_emb_matrix(dir_path):
        logger.debug("Loading Word2Vec embedding from {}".format(dir_path))
        embedding_matrix = np.load(
            os.path.join(dir_path, "embedding_matrix.npy")
        )
        return embedding_matrix
