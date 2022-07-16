# ICD Coding Benchmark

[![GitHub contributors](https://img.shields.io/github/contributors/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/network)
[![GitHub issues](https://img.shields.io/github/issues/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/issues)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dalgu90/icd-coding-benchmark/issues)


Automatic ICD coding benchmark based on the MIMIC dataset.

**NOTE: 🚧 The repo is under construction. Please see below for available datasets/models.**

Automatic diagnosis coding[^1] in clinical NLP is a task to predict the diagnoses and the procedures during a hospital stay given the summary of the stay (discharge summary).
The labels of the task are mostly represented in ICD (international classification of disease) codes which are alpha-numeric codes widely adopted by hospitals in the US.
The most popular database used in automatic diagnosis coding is the MIMIC-III dataset, but the preprocessing varies among the literature, and some of them are done incorrectly.
Such inconsistency and error make it hard to compare different methods on automatic diagnosis coding and, arguably, results in incorrect evaluations of the methods.

This code repository aims to provide a standardized benchmark of automatic diagnosis coding with the MIMIC-III database.
The benchmark encompasses all the procedures of ICD coding: dataset pre-processing, model training/evaluation, and interactive web demo.

We currently provide (items in parentheses are under development):
- Four preset of preprocessed datasets: MIMIC-III full, top-50, full (old), top-50 (old), where we referred to (old) as the version of CAML[^2].
- ICD coding models: CNN, CAML, MultiResCNN, DCAN, Fusion, TransICD, (LAAT)
- Interactive demo


## Preparation
Please put the MIMIC-III `csv.gz` files (v1.4) under `datasets/mimic3/csv/`. You can also create symbolic links pointing to the files.


## Pre-processing
Please run the following command to generate the MIMIC-III top-50 dataset or generate other versions using the config files in `configs/preprocessing`.
```
$ python run_preprocessing.py --config_path configs/preprocessing/mimic3_50.yml
```


## Training / Testing
Please run the following command to train, or resume training of, the CAML model on the MIMIC-III top-50 dataset. You can evaluate the model with `--test` options and use other config files under `configs`.
```
$ python run.py --config_path configs/caml_mimic3_50.yml         # Train
$ python run.py --config_path configs/caml_mimic3_50.yml --test  # Test
```


## Results
- MIMIC-III full

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@8      | P@15     |
|--------------|-----------|-----------|----------|----------|----------|----------|
| CAML         | 0.890823  | 0.984352  | 0.048890 | 0.498832 | 0.703181 | 0.553875 |
| MultiResCNN  | 0.914338  | 0.986860  | 0.084015 | 0.558163 | 0.739324 | 0.587169 |
| DCAN         | 0.854977  | 0.980384  | 0.057142 | 0.522896 | 0.718491 | 0.568149 |
| Fusion       | 0.912643  | 0.986653  | 0.078532 | 0.556019 | 0.743105 | 0.588988 |

- MIMIC-III top-50

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@5      |
|--------------|-----------|-----------|----------|----------|----------|
| CAML         | 0.917969  | 0.942569  | 0.608479 | 0.688646 | 0.662709 |
| MultiResCNN  | 0.926459  | 0.949508  | 0.644801 | 0.718600 | 0.672604 |
| DCAN         | 0.933557  | 0.953008  | 0.658478 | 0.727083 | 0.681014 |
| Fusion       | 0.932295  | 0.952335  | 0.660386 | 0.726407 | 0.678726 |
| TransICD     | 0.918623  | 0.940373  | 0.603596 | 0.680459 | 0.644898 |

- MIMIC-III full (old)

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@8      | P@15     |
|--------------|-----------|-----------|----------|----------|----------|----------|
| CAML         | 0.880379  | 0.983444  | 0.057407 | 0.500574 | 0.696582 | 0.546777 |
| MultiResCNN  | 0.907786  | 0.986263  | 0.077205 | 0.550036 | 0.736099 | 0.583472 |
| DCAN         | 0.844949  | 0.978251  | 0.063707 | 0.524617 | 0.722160 | 0.570087 |
| Fusion       | 0.907964  | 0.986258  | 0.079416 | 0.559838 | 0.747628 | 0.591281 |

- MIMIC-III top-50 (old)

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@5      |
|--------------|-----------|-----------|----------|----------|----------|
| CAML         | 0.881043  | 0.908731  | 0.519399 | 0.610033 | 0.612955 |
| MultiResCNN  | 0.899234  | 0.927863  | 0.589514 | 0.670533 | 0.640023 |
| DCAN         | 0.913397  | 0.937868  | 0.611135 | 0.688379 | 0.649393 |
| Fusion       | 0.904610  | 0.929229  | 0.611743 | 0.674127 | 0.640023 |
| TransICD     | 0.896408  | 0.924700  | 0.544390 | 0.639996 | 0.621862 |


## Run demo
After you train a model, you can run an interactive demo app of it (CAML on MIMIC-III top-50, for example) by running
```
$ streamlit run app.py -- --config_path configs/demo/caml_mimic3_50.yml
```
You can write own config file specifying modules as same as in pre-processing and training


## Authors
(in alphabetical order)
- Abheesht Sharma [@abheesht17](https://github.com/abheesht17)
- Juyong Kim [@dalgu90](https://github.com/dalgu90)
- Suhas Shanbhogue [@SuhasShanbhogue](https://github.com/SuhasShanbhogue)


## Cite this work
```
@misc{juyong2022icdcodinggithub,
  author = {Juyong Kim and Abheesht Sharma and Suhas Shanbhogue},
  title = {dalgu90/icd-coding-benchmark},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dalgu90/icd-coding-benchmark}},
}
```

[^1]: Also referred to as medical coding, clinical coding, or simply ICD coding in other literature. They may have different meanings in detail.
[^2]: Mullenbach, et al., Explainable Prediction of Medical Codes from Clinical Text, NAACL 2018 ([paper](https://arxiv.org/abs/1802.05695), [code](https://github.com/jamesmullenbach/caml-mimic))
