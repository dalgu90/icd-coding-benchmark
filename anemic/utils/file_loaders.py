import json
import logging
import sys

import pandas as pd

from anemic.utils.text_loggers import get_logger

logger = get_logger(__name__)


def load_csv_as_df(file_path, dtype=None):
    logger.debug("Loading file {} as dataframe".format(file_path))
    if file_path.endswith(".csv"):
        df = pd.read_csv(
            file_path, on_bad_lines="skip", low_memory=True, dtype=dtype
        )
    elif file_path.endswith(".gz"):
        df = pd.read_csv(
            file_path,
            compression="gzip",
            on_bad_lines="skip",
            low_memory=True,
            dtype=dtype,
        )
    return df


def save_df(df, file_path):
    logger.debug("Saving dataframe as CSV file {}".format(file_path))
    df.to_csv(file_path, index=False)


def load_json(file_path):
    logger.debug("Loading JSON file {} as dictionary".format(file_path))
    with open(file_path, "r") as f:
        ret_dict = json.load(f)
    return ret_dict


def save_json(d, file_path):
    logger.debug("Saving dictionary as JSON file {}".format(file_path))
    with open(file_path, "w") as f:
        json.dump(d, f, indent=4)
