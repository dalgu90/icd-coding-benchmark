import os

from src.utils.file_loaders import load_json
from src.utils.mapper import ConfigMapper


@ConfigMapper.map("dataset_splitters", "caml_official_split")
class CamlOfficialSplit:
    def __init__(self, config):
        self.train_split = load_json(os.path.join(config.hadm_dir,
                                                  config.train_hadm_ids_name))
        self.val_split = load_json(os.path.join(config.hadm_dir,
                                                config.val_hadm_ids_name))
        self.test_split = load_json(os.path.join(config.hadm_dir,
                                                 config.test_hadm_ids_name))

    def __call__(self, df, hadm_id_col_name):
        print("\nSplitting Data Based on Official Split...")
        train_df = df[df[hadm_id_col_name].isin(self.train_split)]
        val_df = df[df[hadm_id_col_name].isin(self.val_split)]
        test_df = df[df[hadm_id_col_name].isin(self.test_split)]
        return (train_df, val_df, test_df)
