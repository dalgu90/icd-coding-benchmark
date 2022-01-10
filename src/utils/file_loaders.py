import pandas as pd


def load_csv_as_df(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, on_bad_lines="skip", low_memory=True)
    elif file_path.endswith(".gz"):
        df = pd.read_csv(
            file_path, compression="gzip", on_bad_lines="skip", low_memory=True
        )
    return df
