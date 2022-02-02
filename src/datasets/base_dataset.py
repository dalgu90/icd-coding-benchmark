import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.file_loaders import load_json
from src.utils.mapper import ConfigMapper


@ConfigMapper.map("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config
        data_path = self._config.data_file

        # Load vocab (dict of {word: idx})
        self.vocab = load_json(config.vocab_file)
        self.vocab_size = len(self.vocab)
        assert self.vocab_size == max(self.vocab.values()) + 1

        # Load labels (dict of {code: idx})
        self.all_labels = load_json(config.label_file)
        self.num_labels = len(self.all_labels)
        assert self.num_labels == max(self.all_labels.values()) + 1

        # To-do: This class currently deals with only CSV files. We can extend
        # this to deal with other file types (.json, .xlsx, etc.).

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_note = row[self._config.column_names.clinical_note]
        codes = row[self._config.column_names.label].split(";")

        # Note -> word idxs (UNK is assigned after the last word)
        clinical_note = [self.vocab[w] if w in self.vocab else self.vocab_size
                         for w in clinical_note.split()]

        # ICD codes -> binary labels
        labels = np.zeros(self.num_labels, dtype=np.int32)
        for code in codes:
            labels[self.all_labels[code]] = 1

        return (clinical_note, labels)

    def get_collate_fn(self):
        def _collate_fn(examples):
            notes, labels = zip(*examples)

            # Pad notes
            max_note_len = max(map(len, notes))
            notes = [note + [0] * (max_note_len - len(note)) for note in notes]

            # Convert into Tensor
            notes = torch.tensor(notes)
            labels = torch.tensor(labels)

            return notes, labels
        return _collate_fn


