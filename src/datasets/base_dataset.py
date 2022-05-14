import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.file_loaders import load_csv_as_df, load_json
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config

        # Load vocab (dict of {word: idx})
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        self.vocab = embedding_cls.load_vocab(self._config.word2vec_dir)
        self.vocab_size = len(self.vocab)
        assert self.vocab_size == max(self.vocab.values()) + 1
        self.pad_idx = self.vocab[self._config.pad_token]
        self.unk_idx = self.vocab[self._config.unk_token]
        self.inv_vocab = {i: w for w, i in self.vocab.items()}

        # Load labels (dict of {code: idx})
        label_path = os.path.join(
            self._config.dataset_dir, self._config.label_file
        )
        self.all_labels = load_json(label_path)
        self.num_labels = len(self.all_labels)
        assert self.num_labels == max(self.all_labels.values()) + 1
        self.inv_labels = {i: c for c, i in self.all_labels.items()}
        logger.debug(
            "Loaded {} ICD code labels from {}".format(
                self.num_labels, label_path
            )
        )

        # To-do: This class currently deals with only JSON files. We can extend
        # this to deal with other file types (.csv, .xlsx, etc.).

        # Load data (JSON)
        data_path = os.path.join(
            self._config.dataset_dir, self._config.data_file
        )
        self.df = pd.DataFrame.from_dict(load_json(data_path))
        logger.info(
            "Loaded dataset from {} ({} examples)".format(
                data_path, len(self.df)
            )
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_note = row[self._config.column_names.clinical_note]
        codes = row[self._config.column_names.labels].split(";")

        # Note (list) -> word idxs (UNK is assigned at the last word)
        token_idxs = self.encode_tokens(clinical_note)

        # ICD codes -> binary labels
        labels = self.encode_labels(codes)
        one_hot_labels = np.zeros(self.num_labels, dtype=np.int32)
        for l in labels:
            one_hot_labels[l] = 1

        return (token_idxs, one_hot_labels)

    def encode_tokens(self, tokens):
        """Convert list of words into list of token idxs, and truncate"""
        token_idxs = [
            self.vocab[w] if w in self.vocab else self.unk_idx
            for w in tokens
        ]
        token_idxs = token_idxs[: self._config.max_length]
        return token_idxs

    def decode_tokens(self, token_idxs):
        """Convert list of token idxs into list of words"""
        return [self.inv_vocab[idx] for idx in token_idxs]

    def encode_labels(self, codes):
        """Convert list of ICD codes into labels"""
        return [self.all_labels[c] for c in codes]

    def decode_labels(self, labels):
        """Convert labels into list of ICD codes"""
        return [self.inv_labels[l] for l in labels]

    def collate_fn(self, examples):
        """Concatenate examples into note and label tensors"""
        notes, labels = zip(*examples)

        # Pad notes
        max_note_len = max(map(len, notes))
        notes = [
            note + [self.pad_idx] * (max_note_len - len(note)) for note in notes
        ]

        # Convert into Tensor
        notes = torch.tensor(notes)
        labels = torch.tensor(labels)

        return notes, labels
