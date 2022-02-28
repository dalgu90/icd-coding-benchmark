import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.file_loaders import load_csv_as_df, load_json
from src.utils.mapper import ConfigMapper


@ConfigMapper.map("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config

        # Load vocab (dict of {word: idx})
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        self.vocab = embedding_cls.load_vocab(self._config.word2vec_dir)
        self.vocab_size = len(self.vocab)
        assert self.vocab_size == max(self.vocab.values()) + 1
        self.pad_idx = self.vocab.index(self._config.pad_token)
        self.unk_idx = self.vocab.index(self._config.unk_token)

        # Load labels (dict of {code: idx})
        label_path = os.path.join(self._config.dataset_dir,
                                  self._config.label_file)
        self.all_labels = load_json(label_path)
        self.num_labels = len(self.all_labels)
        assert self.num_labels == max(self.all_labels.values()) + 1

        # To-do: This class currently deals with only JSON files. We can extend
        # this to deal with other file types (.csv, .xlsx, etc.).

        # Load data (JSON)
        data_path = os.path.join(self._config.dataset_dir,
                                 self._config.data_file)
        print(f'Load dataset from {data_path}')
        self.df = pd.DataFrame.from_dict(load_json(data_path))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_note = row[self._config.column_names.clinical_note]
        codes = row[self._config.column_names.labels].split(";")

        # Note (list) -> word idxs (UNK is assigned at the last word)
        clinical_note = [self.vocab[w] if w in self.vocab else self.unk_idx
                         for w in clinical_note]

        # ICD codes -> binary labels
        labels = np.zeros(self.num_labels, dtype=np.int32)
        for code in codes:
            labels[self.all_labels[code]] = 1

        return (clinical_note, labels)

    def collate_fn(self, examples):
        """ Concatenate examples into note and label tensors """
        notes, labels = zip(*examples)

        # Pad notes
        max_note_len = max(map(len, notes))
        notes = [note + [self.pad_idx] * (max_note_len - len(note)) for note in notes]

        # Convert into Tensor
        notes = torch.tensor(notes)
        labels = torch.tensor(labels)

        return notes, labels

