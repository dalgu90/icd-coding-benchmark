from collections import Counter

from src.utils.file_loaders import save_json


class TopKCodes:
    def __init__(self, k, labels_save_path):
        self.k = k
        self.top_k_codes = []
        self.labels_save_path = labels_save_path

    def __call__(self, label_col_name, code_df):
        print(f"\nFiltering Dataset Samples Based on Top-{self.k} Codes...")
        self.find_top_k_codes(label_col_name, code_df)
        save_json(
            {v: k for k, v in enumerate(self.top_k_codes)},
            self.labels_save_path,
        )
        if self.k == 0:
            return code_df
        indices_to_delete = []
        for idx, row in code_df.iterrows():
            if set(row[label_col_name].split(",")).issubset(
                set(self.top_k_codes)
            ):
                indices_to_delete.append(idx)
        code_df.drop(indices_to_delete, inplace=True)
        return code_df

    def find_top_k_codes(self, label_col_name, code_df):
        counts = Counter()
        for _, row in code_df.iterrows():
            for label in row[label_col_name].split(";"):
                counts[label] += 1
        if self.k == 0:
            self.top_k_codes = [code for code, _ in counts]
        else:
            self.top_k_codes = [code for code, _ in counts.most_common(self.k)]
