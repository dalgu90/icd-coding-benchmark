"""Contains various kinds of embeddings like Glove, BERT, etc."""

import os

import gensim

from src.utils.mapper import ConfigMapper


@ConfigMapper.map("embeddings", "word2vec")
class Word2VecEmbedding:
    def __init__(self, config):
        self._config = config
        self.save_path = os.path.join(
            self._config.embedding_dir, self._config.model_file_name
        )

    def train(self, corpus):
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            corpus, **self._config.word2vec_params.as_dict()
        )
        model.wv.save(self.save_path)

    def load_emb_matrix(self):
        wv = gensim.models.KeyedVectors.load_word2vec_format(
            self.save_path, mmap="r"
        )
        return wv
