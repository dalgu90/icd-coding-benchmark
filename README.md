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
- ICD coding models: CNN, CAML, MultiResCNN[^3], DCAN[^4], Fusion[^5], TransICD[^6], (LAAT)
- Interactive demo


## Preparation
Please put the MIMIC-III `csv.gz` files (v1.4) under `datasets/mimic3/csv/`. You can also create symbolic links pointing to the files.


## Pre-processing
Please run the following command to generate the MIMIC-III top-50 dataset or generate other versions using the config files in `configs/preprocessing`.
```
$ python run_preprocessing.py --config_path configs/preprocessing/default/mimic3_50.yml
```


## Training / Testing
Please run the following command to train, or resume training of, the CAML model on the MIMIC-III top-50 dataset. You can evaluate the model with `--test` options and use other config files under `configs`.
```
$ python run.py --config_path configs/caml/caml_mimic3_50.yml         # Train
$ python run.py --config_path configs/caml/caml_mimic3_50.yml --test  # Test
```
Training is logged through TensorBoard graph (located in the output dir under `results/`).
Also, logging through text files is performed on pre-processing, training, and evaluation. Log files will be located under `logs/`.


## Results
- MIMIC-III full

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@8      | P@15     |
|--------------|-----------|-----------|----------|----------|----------|----------|
| CNN          | 0.833543  | 0.974153  | 0.032920 | 0.427462 | 0.619773 | 0.472282 |
| CAML         | 0.890823  | 0.984352  | 0.048890 | 0.498832 | 0.703181 | 0.553875 |
| MultiResCNN  | 0.914338  | 0.986860  | 0.084015 | 0.558163 | 0.739324 | 0.587169 |
| DCAN         | 0.858853  | 0.980687  | 0.060191 | 0.526127 | 0.721456 | 0.572420 |
| Fusion       | 0.912643  | 0.986653  | 0.078532 | 0.556019 | 0.743105 | 0.588988 |
| TransICD     | 0.897434  | 0.984421  | 0.057159 | 0.498058 | 0.665703 | 0.524041 |

- MIMIC-III top-50

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@5      |
|--------------|-----------|-----------|----------|----------|----------|
| CNN          | 0.914614  | 0.937504  | 0.627029 | 0.695571 | 0.649474 |
| CAML         | 0.917969  | 0.942569  | 0.608479 | 0.688646 | 0.662709 |
| MultiResCNN  | 0.926459  | 0.949508  | 0.644801 | 0.718600 | 0.672604 |
| DCAN         | 0.934825  | 0.953576  | 0.659244 | 0.727007 | 0.685034 |
| Fusion       | 0.932295  | 0.952335  | 0.660386 | 0.726407 | 0.678726 |
| TransICD     | 0.918623  | 0.940373  | 0.603596 | 0.680459 | 0.644898 |

- MIMIC-III full (old)

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@8      | P@15     |
|--------------|-----------|-----------|----------|----------|----------|----------|
| CNN          | 0.833935  | 0.973760  | 0.032343 | 0.421507 | 0.608244 | 0.465836 |
| CAML         | 0.880379  | 0.983444  | 0.057407 | 0.500574 | 0.696582 | 0.546777 |
| MultiResCNN  | 0.907786  | 0.986263  | 0.077205 | 0.550036 | 0.736099 | 0.583472 |
| DCAN         | 0.842585  | 0.977854  | 0.062069 | 0.525559 | 0.720529 | 0.570977 |
| Fusion       | 0.907964  | 0.986258  | 0.079416 | 0.559838 | 0.747628 | 0.591281 |
| TransICD     | 0.889693  | 0.983436  | 0.050560 | 0.492136 | 0.660291 | 0.516528 |

- MIMIC-III top-50 (old)

| Model        | macro AUC | micro AUC | macro F1 | micro F1 | P@5      |
|--------------|-----------|-----------|----------|----------|----------|
| CNN          | 0.892395  | 0.918188  | 0.585288 | 0.645080 | 0.619433 |
| CAML         | 0.881043  | 0.908731  | 0.519399 | 0.610033 | 0.612955 |
| MultiResCNN  | 0.899234  | 0.927863  | 0.589514 | 0.670533 | 0.640023 |
| DCAN         | 0.914818  | 0.939139  | 0.613484 | 0.691155 | 0.657259 |
| Fusion       | 0.904610  | 0.929229  | 0.611743 | 0.674127 | 0.640023 |
| TransICD     | 0.896408  | 0.924700  | 0.544390 | 0.639996 | 0.621862 |


## Run demo
After you train a model, you can run an interactive demo app of it (CAML on MIMIC-III top-50, for example) by running
```
$ streamlit run app.py -- --config_path configs/demo/multi_mimic3_50.yml  # CAML, MultiResCNN, DCAN, Fusion on MIMIC-III top-50
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
[^3]: Li and Yu, ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network, AAAI 2020 ([paper](https://arxiv.org/abs/1912.00862), [code](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network))
[^4]: Ji, et al., Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text, Clinical NLP Workshop 2020 ([paper](https://aclanthology.org/2020.clinicalnlp-1.8/), [code](https://github.com/shaoxiongji/DCAN))
[^5]: Luo, et al., Fusion: Towards Automated ICD Coding via Feature Compression, ACL 2020 Findings ([paper](https://aclanthology.org/2021.findings-acl.184/), [code](https://github.com/machinelearning4health/Fusion-Towards-Automated-ICD-Coding))
[^6]: Biswas, et al., TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding, AIME 2021 ([paper](https://arxiv.org/abs/2104.10652), [code](https://github.com/AIMedLab/TransICD))
