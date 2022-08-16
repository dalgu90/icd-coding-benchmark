#!/usr/bin/env python
"""
    The interactive demo of ICD coding benchmark (prototype)
"""
import argparse
import copy
import csv

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from captum.attr import LayerIntegratedGradients

from anemic.modules.preprocessors import ClinicalNotePreprocessor
from anemic.utils.configuration import Config
from anemic.utils.mapper import ConfigMapper
from anemic.utils.misc import html_word_importance

hash_funcs = {
    Config: lambda x: hash(str(x)),
    torch.nn.parameter.Parameter: lambda x: hash(x.shape),
}


@st.cache(hash_funcs=hash_funcs, allow_output_mutation=True)
def load_config():
    parser = argparse.ArgumentParser(
        description="Demo app for automatic ICD coding"
    )
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
                        "remove_common_medical_terms": True,
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
    model_dict = {}
    lig_dict = {}
    if hasattr(config, "models"):
        model_configs = config.models
    else:
        model_configs = [config]

    for model_config in model_configs:
        model = ConfigMapper.get_object("models", model_config.model.name)(
            model_config.model.params
        )
        # Load ckpt
        ckpt_saver = ConfigMapper.get_object(
            "checkpoint_savers", model_config.checkpoint_saver.name
        )(model_config.checkpoint_saver.params)
        best_ckpt = ckpt_saver.get_best_checkpoint()
        if best_ckpt is None:
            raise ValueError("Best ckpt not found.")
        ckpt_saver.load_ckpt(model, best_ckpt, optimizer=None)

        # Set model GPU & eval mode
        if config.demo.use_gpu:
            model.cuda()
        model.eval()

        # 4. Captum attribute module
        try:
            embed_layer_name = getattr(
                model_config.model, "embed_layer_name", "embed"
            )
            embed_layer = getattr(model, embed_layer_name)
        except:
            raise ValueError(
                f"Config for {model_config.model.name} does not"
                "specify name of the embedding layer."
            )
        lig = LayerIntegratedGradients(model, embed_layer)

        model_dict[model_config.model.name] = model
        lig_dict[model_config.model.name] = lig

    return preprocessor, dataset, model_dict, lig_dict


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

st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 75, 75);
    color: white;
    width: 100%;
    border: 0px;
    padding-right: 20px;
}
.streamlit-expanderHeader { font-size: medium; }
</style>""",
    unsafe_allow_html=True,
)

# Title & status line
st.title("🩺 ICD Coding Interactive Demo")
status = st.empty()


# Displaying status
def set_status(text):
    status.text(f"💡 {text}")


set_status("Loading model...")

# App info
info_str = """
- This is an interactive app to run automatic diagnostic coding models and
  visualize the diagnosis code prediction with the importance score of the
  input.
- To run the model, please put a discharge summary in the "Discharge summary
  note" box and hit the Submit button. Try different options of preprocessing
  and visualization!
- To run other models, please specify the config (in `configs/demo/`) of an
  available model from the command-line argument. Checkpoints of the
  corresponding models need to be downloaded.
- For more help, please check out our
  [ICD Coding Benchmark](https://github.com/dalgu90/icd-coding-benchmark) repo.
  Thanks!
"""
with st.expander("ℹ️  About the app", expanded=False):
    st.write(info_str)
    st.markdown("")

# Load config, modules, and icd descriptions
config = load_config()
preprocessor, dataset, model_dict, lig_dict = load_modules(config)
icd_desc = load_icd_desc(config)
set_status(f"Model loaded ({', '.join(model_dict.keys())})")

# Main form
with st.form("my_form"):
    # Layout
    _, col1, _, col2, _ = st.columns([0.07, 1, 0.07, 4, 0.07])
    with col1:
        # Model selection
        # TODO: Support multiple models
        model_name = st.radio(
            "Choose model",
            list(model_dict.keys()),
            help="""Currently, running multiple models is not supported.""",
        )

        # K selection
        top_k = st.slider(
            "Top-k prediction",
            min_value=1,
            max_value=min(50, len(icd_desc)),
            value=config.demo.top_k,
            help="""The number of predictions with highest scores. Between 1 and
                    maximum number of output codes (or 50 if label space is too
                    too large).""",
        )

        # Input visualization selection
        vis_score_options = [
            "NO",
            "Integrated Gradients",
        ]
        if any(
            hasattr(model, "get_input_attention")
            for model in model_dict.values()
        ):
            vis_score_options.append("Attention score")

        vis_score = st.radio(
            "Visualize attribution score",
            vis_score_options,
            help="""Interpretability visualization methods. Attention score is
            available only for attention-based models.""",
        )

        vis_code_options = ["Choose ICD code"]
        vis_code_options += dataset.decode_labels(range(dataset.num_labels))
        vis_code = st.selectbox(
            "ICD code to compute attribution score",
            vis_code_options,
            index=0,
            help="""Code to visualize the attribution. It will be used when the
                    interpretability method is other than "NO".""",
        )

        # Preprocessing option selection (truncation is not controlled)
        st.markdown(
            """<p style="font-size: small;"> Preprocessing </p>""",
            unsafe_allow_html=True,
        )
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

        submitted = st.form_submit_button("🚀 SUBMIT!")
    with col2:
        # Input text
        css_str = "line-height:1; margin-top:1rem; margin-bottom:-2rem;"
        st.markdown(
            f"""<div style="{css_str}">Discharge summary note</div>""",
            unsafe_allow_html=True,
        )
        input_text = st.text_area(label="", height=200)
        # input_text = st.text_area(label="Discharge summary note", height=200)
        input_text = input_text.strip()
        if input_text:
            set_status("Processing...")

        # Preprocess text
        with st.expander("Preprocessed text", expanded=False):
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

        with st.expander("Input tokens", expanded=False):
            # Tokenize text with vocab
            token_idxs = dataset.encode_tokens(preprocessed_text.split())
            tokens = dataset.decode_tokens(token_idxs)
            token_text = " ".join(tokens)
            st.text_area(
                label="Tokens", value=token_text, height=200, disabled=True
            )

        # Model prediction
        st.write("ICD code prediction")
        if token_idxs and len(token_idxs) >= config.demo.min_input_len:
            model = model_dict[model_name]
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
                    "ICD_Code": top_k_codes,
                    "Probability": top_k_probs,
                    "Description": top_k_descs,
                }
            )
            output_df.index += 1
            cmap = sns.light_palette("#AC304B", as_cmap=True)
            output_df = output_df.style.background_gradient(
                cmap=cmap, subset=["Probability"], vmin=0.0, vmax=1.0
            ).format({"Probability": "{:.4f}"})
            st.table(output_df)

            # Attribution score:
            target_label = vis_code_options.index(vis_code) - 1  # starts from 1
            with st.expander(f"Attribution score", expanded=True):
                if vis_score == "NO":
                    st.markdown("**[No attribution method selected]**")
                elif target_label == -1:
                    st.markdown("**[No ICD code selected]**")
                elif vis_score == "Attention score" and not hasattr(
                    model, "get_input_attention"
                ):
                    st.markdown("**[Model does not support attention score]**")
                else:
                    if vis_score == "Integrated Gradients":
                        lig = lig_dict[model_name]
                        attrs, approx_error = lig.attribute(
                            batch_input,
                            target=target_label,
                            return_convergence_delta=True,
                        )
                        attrs = attrs.sum(dim=2).squeeze(0)
                        attrs = (
                            (attrs / torch.norm(attrs)).cpu().detach().numpy()
                        )
                    elif vis_score == "Attention score":
                        attrs = model.get_input_attention()
                        attrs = attrs[:, target_label].squeeze(0)
                        attrs /= np.linalg.norm(attrs)
                    else:
                        raise ValueError(f"Wrong model selected.")

                    assert len(attrs) >= len(tokens)
                    html_string = html_word_importance(tokens, attrs)
                    st.markdown(
                        f"**{vis_score}** for **{vis_code}** "
                        f"({icd_desc[vis_code]})"
                    )
                    st.markdown(html_string, unsafe_allow_html=True)
                    st.markdown("")
        elif token_idxs and len(token_idxs) < config.demo.min_input_len:
            st.markdown(f"**[Input too short(<{config.demo.min_input_len})]**")
        else:
            st.markdown("**[No input]**")

        if input_text:
            set_status("Done!")
