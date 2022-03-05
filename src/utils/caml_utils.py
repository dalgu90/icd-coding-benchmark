"""
    CAML utils (Mullenbach et al. 2018)
    https://github.com/jamesmullenbach/caml-mimic
"""
import csv
import os
from collections import defaultdict

from src.modules.preprocessors import CodeProcessor
from src.utils.file_loaders import load_csv_as_df, load_json
from src.utils.mapper import ConfigMapper


def load_lookups(
    dataset_dir,
    mimic_dir,
    static_dir,
    word2vec_dir,
    label_file="labels.json",
    version="mimic3",
):
    """
    Inputs:
        args: Input arguments
        desc_embed: true if using DR-CAML
    Outputs:
        vocab lookups, ICD code lookups, description lookup
        vector lookup
    """
    # get vocab lookups
    embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
    w2ind = embedding_cls.load_vocab(word2vec_dir)
    ind2w = {i: w for w, i in w2ind.items()}

    # get codes
    c2ind = load_json(os.path.join(dataset_dir, label_file))
    ind2c = {i: c for c, i in c2ind.items()}

    # get description lookups
    desc_dict = load_code_descriptions(
        mimic_dir=mimic_dir, static_dir=static_dir, version=version
    )

    # In this implementation, we don't use dv_dict ({code: desc token idxs}).
    # Instead, we tokenize/embed on the code description on the fly.

    dicts = {
        "ind2w": ind2w,
        "w2ind": w2ind,
        "ind2c": ind2c,
        "c2ind": c2ind,
        "desc": desc_dict,
    }
    return dicts


def load_code_descriptions(mimic_dir, static_dir, version="mimic3"):
    # We use the code processor for reformat (adding period in ICD codes)
    reformat_fn = CodeProcessor.reformat_icd_code

    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == "mimic2":
        with open(os.path.join(static_dir, "MIMIC_ICD9_mapping"), "r") as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        diag_df = load_csv_as_df(
            os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv.gz"),
            dtype={"ICD9_CODE": str},
        )
        for _, row in diag_df.iterrows():
            desc_dict[reformat_fn(row.ICD9_CODE, True)] = row.LONG_TITLE

        proc_df = load_csv_as_df(
            os.path.join(mimic_dir, "D_ICD_PROCEDURES.csv.gz"),
            dtype={"ICD9_CODE": str},
        )
        for _, row in proc_df.iterrows():
            desc_dict[reformat_fn(row.ICD9_CODE, True)] = row.LONG_TITLE

        with open(
            os.path.join(static_dir, "icd9_descriptions.txt"), "r"
        ) as labelfile:
            for i, row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = " ".join(row[1:])
    return desc_dict


def pad_desc_vecs(desc_vecs):
    # In this implementation, padding is performed not in the in-place manner
    # pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        pad_vecs.append(vec + [0] * (desc_len - len(vec)))
    return pad_vecs
