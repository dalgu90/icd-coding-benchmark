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
import numpy as np
import codecs
import re

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



def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num



def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()

    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break

                embedd_dict[word] = word_vector

    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
            for line in file:
                # logging.info(line)
                line = line.strip()
                if len(line) == 0:
                    continue
                # tokens = line.split()
                tokens = re.split(r"\s+", line)
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    # assert (embedd_dim + 1 == len(tokens))
                    if embedd_dim + 1 != len(tokens):
                        continue
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd


    return embedd_dict, embedd_dim

def build_pretrain_embedding(embedding_path, word_alphabet, norm):

    embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([len(word_alphabet)+2, embedd_dim], dtype=np.float32)  # add UNK (last) and PAD (0)
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1

        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1

        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1

        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1

        else:
            if norm:
                pretrain_emb[index, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
            else:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    # initialize pad and unknown
    pretrain_emb[0, :] = np.zeros([1, embedd_dim], dtype=np.float32)
    if norm:
        pretrain_emb[-1, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
    else:
        pretrain_emb[-1, :] = np.random.uniform(-scale, scale, [1, embedd_dim])


    print("pretrained word emb size {}".format(len(embedd_dict)))
    print("prefect match:%.2f%%, case_match:%.2f%%, dig_zero_match:%.2f%%, "
                 "case_dig_zero_match:%.2f%%, not_match:%.2f%%"
                 %(perfect_match*100.0/len(word_alphabet), case_match*100.0/len(word_alphabet), digits_replaced_with_zeros_found*100.0/len(word_alphabet),
                   lowercase_and_digits_replaced_with_zeros_found*100.0/len(word_alphabet), not_match*100.0/len(word_alphabet)))

    return pretrain_emb, embedd_dim

