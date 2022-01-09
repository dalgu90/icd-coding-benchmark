import pandas as pd


def load_csv_as_df(file_path):
    if(file_path.endswith(".csv")):
        df = pd.read_csv(file_path, error_bad_lines=False)
    elif(file_path.endswith(".gz")):
        df = pd.read_csv(file_path, compression="gzip", error_bad_lines=False)
    return df