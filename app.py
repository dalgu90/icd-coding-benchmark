#!/usr/bin/env python
"""
    The interactive demo of ICD coding benchmark (prototype)
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
from src.utils.mapper import ConfigMapper

hash_funcs = {
    Config: lambda x: hash(str(x)),
    torch.nn.parameter.Parameter: lambda x: hash(x.shape),
}


@st.cache(hash_funcs=hash_funcs, allow_output_mutation=True)
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


@st.cache(hash_funcs=hash_funcs, allow_output_mutation=True)
def load_modules(config):
    # Load preprocessor
    preprocessor = ClinicalNotePreprocessor(config.clinical_note_preprocessing)

    # Load dataset
    dataset = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params
    )

    # Load model
    model = ConfigMapper.get_object("models", config.model.name)(
        config.model.params
    )

    # Restore ckpt
    ckpt_saver = ConfigMapper.get_object(
        "checkpoint_savers", config.checkpoint_saver.name
    )(config.checkpoint_saver.params)
    best_ckpt = ckpt_saver.get_best_checkpoint()
    if best_ckpt is None:
        raise ValueError("Best ckpt not found")
    ckpt_saver.load_ckpt(model, best_ckpt, optimizer=None)

    # Set model GPU & eval mode
    if config.demo.use_gpu:
        model.cuda()
    model.eval()

    return preprocessor, dataset, model


@st.cache(hash_funcs=hash_funcs, allow_output_mutation=True)
def load_icd_desc(config):
    # Icd description
    icd_desc = list(csv.reader(open(config.demo.icd_desc_file), delimiter="\t"))
    icd_desc = {r[0]: r[1] for r in icd_desc}
    return icd_desc


# Title
st.title("ICD Coding Interactive Demo")
status = st.empty()  # Displaying status

# Load config, modules, and icd descriptions
config = load_config()
preprocessor, dataset, model = load_modules(config)
icd_desc = load_icd_desc(config)
status.text(f"Model loaded ({config.model.name})")

# Input text
input_text = st.text_area(label="Discharge summary note", height=200)

# Preprocess text
preprocessed_text = preprocessor(input_text)
st.text_area(
    label="Preprocessed note",
    value=preprocessed_text,
    height=200,
    disabled=True,
)

# Tokenize text with vocab
token_idxs = dataset.encode_tokens(preprocessed_text.split())
token_text = " ".join(dataset.decode_tokens(token_idxs))
st.text_area(label="Tokens", value=token_text, height=200, disabled=True)

# Model prediction
st.write("Output")
if token_idxs:
    # Forward pass
    batch_input = torch.tensor([token_idxs])
    if config.demo.use_gpu:
        batch_input = batch_input.cuda()
    with torch.no_grad():
        batch_output = model(batch_input)
    probs = torch.sigmoid(batch_output[0].cpu()).numpy()
    top_k_preds = np.argsort(probs)[-1 : -(config.demo.top_k + 1) : -1]
    top_k_probs = [probs[p] for p in top_k_preds]
    top_k_codes = dataset.decode_labels(top_k_preds)
    top_k_descs = [icd_desc[c] for c in top_k_codes]

    # Output as table
    output_df = pd.DataFrame(
        {
            "ICD-9 Code": top_k_codes,
            "Probability": top_k_probs,
            "Description": top_k_descs,
        }
    )
    st.table(output_df.style.format({"Probability": "{:.4f}"}))
else:
    st.markdown("**[No input]**")
