#!/usr/bin/env python
"""
    The interactive demo of ICD coding benchmark (prototype)

TODOs:
    1. Make config hashable by streamlit, so config and other modules can be
    cached.
    2. Organize demo yaml file and the code so that users can easily modify
    3. Support multiple models in the config file and more visualization

"""
import argparse
import csv

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.datasets import *
from src.models import *
from src.modules.embeddings import *
from src.modules.preprocessors import ClinicalNotePreprocessor
from src.utils.checkpoint_savers import *
from src.utils.configuration import Config
from src.utils.file_loaders import load_json
from src.utils.mapper import ConfigMapper

# Title
st.title("ICD Coding Benchmark Demo")
status = st.empty()  # Displaying status


# @st.cache
def load_config():
    parser = argparse.ArgumentParser(description="Here we put app desc")
    parser.add_argument(
        "--config_path",
        type=str,
        action="store",
        help="Path to the config file",
    )
    args = parser.parse_args()
    config = Config(path=args.config_path)
    return config


# @st.cache
def load_model(config):
    # Load preprocessor
    preprocessor = ClinicalNotePreprocessor(
        config.model.clinical_note_preprocessing
    )

    # Load vocab (dict of {word: idx})
    embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
    vocab = embedding_cls.load_vocab(config.model.word2vec_dir)
    vocab_size = len(vocab)
    inv_vocab = {i: k for k, i in vocab.items()}
    assert vocab_size == max(vocab.values()) + 1

    # Load labels
    labels = load_json(config.model.label_file)
    labels = list(zip(*sorted(labels.items(), key=lambda x: x[1])))[0]

    # Icd description
    icd_desc = list(csv.reader(open(config.demo.icd_desc_file), delimiter="\t"))
    icd_desc = {r[0]: r[1] for r in icd_desc}

    # Load model
    model = ConfigMapper.get_object("models", config.model.name)(
        config.model.params
    )
    ckpt_saver = ConfigMapper.get_object(
        "checkpoint_savers", config.model.checkpoint_saver.name
    )(config.model.checkpoint_saver.params)
    best_ckpt = ckpt_saver.get_best_checkpoint()
    if best_ckpt is None:
        raise ValueError("Best ckpt not found")
    ckpt_saver.load_ckpt(model, best_ckpt, optimizer=None)
    model.eval()

    # if config.demo.use_gpu:
    # model.cuda()

    return preprocessor, vocab, inv_vocab, labels, icd_desc, model


status.text("Loading model...")
config = load_config()
preprocessor, vocab, inv_vocab, labels, icd_desc, model = load_model(config)
status.text(f"Model loaded ({config.model.name})")

# Input text
text = st.text_area(label="Discharge summary note", height=200)

# Preprocessed text
text2 = preprocessor(text)
st.text_area(label="Preprocessed note", value=text2, height=200, disabled=True)

# Token indices
unk_idx = vocab[config.model.unk_token]
text3 = [vocab[w] if w in vocab else unk_idx for w in text2.split()]
text3 = text3[: config.model.max_length]
text4 = " ".join([inv_vocab[idx] for idx in text3])
st.text_area(label="Tokens", value=text4, height=200, disabled=True)

st.write("Output")
if text3:
    # Forward pass
    batch_input = torch.tensor([text3])
    # if config.demo.use_gpu:
    # batch_input = batch_input.cuda()
    with torch.no_grad():
        batch_output = model(batch_input)
    probs = torch.sigmoid(batch_output[0].cpu()).numpy()
    top_k_pred = np.argsort(probs)[-1 : -(config.demo.top_k + 1) : -1]

    # Outputs
    output_df = pd.DataFrame(
        {
            "ICD-9 Code": [labels[p] for p in top_k_pred],
            "Probability": [probs[p] for p in top_k_pred],
            "Description": [icd_desc[labels[p]] for p in top_k_pred],
        }
    )
    st.dataframe(output_df.style.format({"Probability": "{:.4f}"}))
else:
    st.markdown("**[No input]**")
