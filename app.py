#!/usr/bin/env python
"""
    The interactive demo of ICD coding benchmark (prototype)
"""
import argparse
import copy
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
    # 1. Load preprocessor
    # First, we change the config to initialize default params and subclasses
    pp_config = copy.deepcopy(config.clinical_note_preprocessing)
    if not pp_config.remove_numeric.perform:
        pp_config.remove_numeric.set_value(
            "replace_numerics_with_letter", False
        )
    if not pp_config.remove_stopwords.perform:
        pp_config.remove_stopwords = Config(
            dic={
                "perform": True,
                "params": Config(
                    dic={
                        "stopwords_file_path": None,
                        "remove_common_medical_terms": False,
                    }
                ),
            }
        )
    if not pp_config.stem_or_lemmatize.perform:
        pp_config.stem_or_lemmatize = Config(
            dic={
                "perform": True,
                "params": Config(
                    dic={"stemmer_name": "nltk.WordNetLemmatizer"}
                ),
            }
        )
    preprocessor = ClinicalNotePreprocessor(pp_config)
    # Restore the original preprocessor config back
    preprocessor._config.remove_stopwords.set_value(
        "perform", config.clinical_note_preprocessing.remove_stopwords.perform
    )
    preprocessor._config.stem_or_lemmatize.set_value(
        "perform", config.clinical_note_preprocessing.stem_or_lemmatize.perform
    )

    # 2. Load dataset
    dataset = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params
    )

    # 3. Load model
    model = ConfigMapper.get_object("models", config.model.name)(
        config.model.params
    )
    # Load ckpt
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
    # Load ICD description (dict of {code: desc})
    icd_desc = list(csv.reader(open(config.demo.icd_desc_file), delimiter="\t"))
    icd_desc = {r[0]: r[1] for r in icd_desc}
    return icd_desc


# Page setup
st.set_page_config(
    page_title="ICD Coding Interactive Demo",
    page_icon=":syringe:",
    layout="wide",
)

# Title & status line
st.title("🩺 ICD Coding Interactive Demo")
status = st.empty()


# Displaying status
def set_status(text):
    status.text(f"💡 {text}")


set_status("")

# App info
info_str = """
- This is an interactive app to run automatic diagnostic coding models and
  visualize the diagnosis code prediction with the importance score of the
  input.
- To run the model, please put a discharge summary in the "Discharge summary
  note" box and hit the Submit button.
- For more models, please specify available models (in `configs/demo/`.
  Corresponding checkpoints need to be downloaded) in the command-line argument,
  or train your own model.
- This app is built in Streamlit, and the format is taken from a Streamlit demo
  app ([BERT Keyword
  Extractor](https://github.com/streamlit/example-app-bert-keyword-extractor)).
- For more help, please checkout our
  [ICD Coding Benchmark](https://github.com/dalgu90/icd-coding-benchmark) repo.
  Thanks!
"""
with st.expander("ℹ️  About the app", expanded=False):
    st.write(info_str)
    st.markdown("")

# Load config, modules, and icd descriptions
config = load_config()
preprocessor, dataset, model = load_modules(config)
icd_desc = load_icd_desc(config)
set_status(f"Model loaded ({config.model.name})")

# Main form
with st.form("my_form"):
    # Layout
    _, col1, _, col2, _ = st.columns([0.07, 1, 0.07, 4, 0.07])
    with col1:
        # Model selection
        # TODO: Support multiple models
        model_type = st.radio(
            "Choose model",
            ["CAML"],
            help="""You can add more models by adding configs when running the
                    app""",
        )

        # K selection
        top_k = st.slider(
            "Top-k prediction",
            min_value=1,
            max_value=min(50, len(icd_desc)),
            value=config.demo.top_k,
            help="""The number of predictions with highest scores. Between 1 and
                    maximum number of output codes""",
        )

        # Input visualization selection
        vis_score_options = [
            "NO",
            "Integrated Gradient",
        ]
        # if hasattr(model, "get_input_attention"):
        # vis_score_options.append("Attention score")

        vis_score = st.radio(
            "Visualize input score",
            vis_score_options,
            help="Visualizing input score."
            # help="""Visualizing input score. Attention is available only for
            # attention-based models"""
        )

        vis_code = st.selectbox(
            "Select ICD code to visualize score",
            ["N/A"],
            index=0,
            help="""Code to visualize the input. You can choose from top-k
                    predictions which are available after the model run""",
        )

        # Preprocessing option selection (truncation is not controlled)
        pp_config = config.clinical_note_preprocessing
        pp_lower_case = st.checkbox(
            "Lowercase",
            value=pp_config.to_lower.perform,
        )
        pp_remove_punc = st.checkbox(
            "Remove punctuation",
            value=pp_config.remove_punctuation.perform,
        )
        pp_remove_numeric = st.checkbox(
            "Remove numeric words",
            value=pp_config.remove_numeric.perform,
        )
        pp_remove_stopwords = st.checkbox(
            "Remove stopwords",
            value=pp_config.remove_stopwords.perform,
        )
        pp_stem = st.checkbox(
            "Stem / lemmatize words",
            value=pp_config.stem_or_lemmatize.perform,
        )

        submitted = st.form_submit_button("🚀 Submit")
    with col2:
        # Input text
        input_text = st.text_area(label="Discharge summary note", height=200)
        input_text = input_text.strip()
        if input_text:
            set_status("Processing...")

        # Preprocess text
        preprocessor._config.to_lower.set_value("perform", pp_lower_case)
        preprocessor._config.remove_punctuation.set_value(
            "perform", pp_remove_punc
        )
        preprocessor._config.remove_numeric.set_value(
            "perform", pp_remove_numeric
        )
        preprocessor._config.remove_stopwords.set_value(
            "perform", pp_remove_stopwords
        )
        preprocessor._config.stem_or_lemmatize.set_value("perform", pp_stem)
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
        st.text_area(
            label="Tokens", value=token_text, height=200, disabled=True
        )

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
            top_k_preds = np.argsort(probs)[-1 : -(top_k + 1) : -1]
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
            output_df.index += 1
            st.table(output_df.style.format({"Probability": "{:.4f}"}))
        else:
            st.markdown("**[No input]**")

        if input_text:
            set_status("Done!")
