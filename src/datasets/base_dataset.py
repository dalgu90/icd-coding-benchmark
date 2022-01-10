import pandas as pd
from torch.utils.data import Dataset

from src.utils.mapper import configmapper


@configmapper("datasets", "base_dataset")
class BaseDataset(Dataset):
    def __init__(self, config):
        self._config = config
        data_path = self._config.data_file

        # To-do: This class currently deals with only CSV files. We can extend
        # this to deal with other file types (.json, .xlsx, etc.).

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_note = row[self._config.column_names.clinical_note]
        labels = row[self._config.column_names.label].split(";")
        return (clinical_note, labels)