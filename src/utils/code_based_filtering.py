import logging
import sys
from collections import Counter

import pandas as pd

from src.utils.file_loaders import save_json
from src.utils.text_loggers import logger


class TopKCodes:
    def __init__(self, k, labels_save_path):
        logger.debug(
            "Finding top-k codes with the following args: k = {}, "
            "label_save_path = {}".format(k, labels_save_path)
        )
        self.k = k
        self.top_k_codes = []
        self.labels_save_path = labels_save_path

    def __call__(self, label_col_name, code_df):
        self.find_top_k_codes(label_col_name, code_df)
        save_json(
            {v: k for k, v in enumerate(self.top_k_codes)},
            self.labels_save_path,
        )
        if self.k == 0:
            return code_df
        indices_to_delete = []
        top_k_codes_set = set(self.top_k_codes)
        for idx, row in code_df.iterrows():
            filtered_indices = set(row[label_col_name].split(";")).intersection(
                top_k_codes_set
            )
            if len(filtered_indices) > 0:
                row[label_col_name] = ";".join(filtered_indices)
            else:
                indices_to_delete.append(idx)
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
