import logging
import sys
from collections import Counter

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.file_loaders import save_json
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


class TopKCodes:
    def __init__(self, k, labels_save_path, labels_freq_save_path=None):
        logger.debug(
            "Finding top-k codes with the following args: k = {}, "
            "label_save_path = {}".format(k, labels_save_path)
        )
        self.k = k
        self.top_k_codes = []
        self.labels_save_path = labels_save_path
        self.labels_freq_save_path = labels_freq_save_path

    def __call__(self, label_col_name, code_df):
        self.find_top_k_codes(label_col_name, code_df)
        save_json(
            {v: k for k, v in enumerate(self.top_k_codes)},
            self.labels_save_path,
        )

        label_freq = {k: 0 for k in self.top_k_codes}

        if self.k == 0:
            if self.labels_freq_save_path:
                for idx, row in code_df.iterrows():
                    codes = set(row[label_col_name].split(";"))
                    for code in codes:
                        label_freq[code] += 1
                save_json(label_freq, self.labels_freq_save_path)
            return code_df

        indices_to_delete = []
        top_k_codes_set = set(self.top_k_codes)
        for idx, row in code_df.iterrows():
            filtered_indices = set(row[label_col_name].split(";")).intersection(
                top_k_codes_set
            )
            if len(filtered_indices) > 0:
                if self.labels_freq_save_path is not None:
                    for filtered_idx in filtered_indices:
                        label_freq[filtered_idx] += 1
                row[label_col_name] = ";".join(filtered_indices)
            else:
                indices_to_delete.append(idx)

        if self.labels_freq_save_path is not None:
            save_json(label_freq, self.labels_freq_save_path)

        code_df.drop(indices_to_delete, inplace=True)
        return code_df

    def find_top_k_codes(self, label_col_name, code_df):
        counts = Counter()
        for _, row in code_df.iterrows():
            for label in row[label_col_name].split(";"):
                counts[label] += 1
        if self.k == 0:
            self.top_k_codes = [code for code, _ in counts.items()]
        else:
            self.top_k_codes = [code for code, _ in counts.most_common(self.k)]

        logger.debug("top-k codes: {}".format(self.top_k_codes))

    def find_code_freq(self, label_lst):
        mlb = MultiLabelBinarizer(classes=label_lst)
