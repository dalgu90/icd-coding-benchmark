from collections import Counter


class TopKCodes:
    def __init__(self, k):
        self.k = k
        self.top_k_codes = []

    def __call__(self, label_col_name, code_df):
        counts = self.find_top_k_codes(label_col_name, code_df)
        self.top_k_codes = [code for code, _ in counts]
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
        return counts.most_common(self.k)
