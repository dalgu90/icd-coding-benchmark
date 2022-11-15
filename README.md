# AnEMIC: An Error-reduced MIMIC ICD Coding Benchmark

[![GitHub contributors](https://img.shields.io/github/contributors/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/network)
[![GitHub issues](https://img.shields.io/github/issues/dalgu90/icd-coding-benchmark)](https://github.com/dalgu90/icd-coding-benchmark/issues)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dalgu90/icd-coding-benchmark/issues)

[![Try Interactive Demo](https://shields.io/badge/Try_Interactive_Demo-green?logo=jekyll&style=for-the-badge)](https://tinyurl.com/icd-coding-demo)
[![Checkpoints Available Here](https://shields.io/badge/Model_checkpoints-download-lightgray?logo=pytorch&style=for-the-badge)](https://drive.google.com/drive/folders/1kextbHf6DYnnD8cXkU9Z0iK0Zh7Vao68?usp=sharing)

Automatic ICD coding benchmark based on the MIMIC dataset.  
Please check our paper on EMNLP 2022 (demo track): [AnEMIC: A Framework for Benchmarking ICD Coding Models](#) (Available soon)

**NOTE: 🚧 The repo is under active development. Please see below for available datasets/models.**

Automatic diagnosis coding[^1] in clinical NLP is a task to predict the diagnoses and the procedures during a hospital stay given the summary of the stay (discharge summary).
The labels of the task are mostly represented in ICD (international classification of disease) codes which are alpha-numeric codes widely adopted by hospitals in the US.
The most popular database used in automatic diagnosis coding is the MIMIC-III dataset, but the preprocessing varies among the literature, and some of them are done incorrectly.
Such inconsistency and error make it hard to compare different methods on automatic diagnosis coding and, arguably, results in incorrect evaluations of the methods.

This code repository aims to provide a standardized benchmark of automatic diagnosis coding with the MIMIC-III database.
The benchmark encompasses all the procedures of ICD coding: dataset pre-processing, model training/evaluation, and interactive web demo.

We currently provide (items in parentheses are under development):
- Four preset of preprocessed datasets: MIMIC-III full, top-50, full (old), top-50 (old), where we referred to (old) as the version of CAML[^2].
- ICD coding models: CNN, CAML, MultiResCNN[^3], DCAN[^4], TransICD[^5], Fusion[^6], (LAAT)
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


## Run demo
After you train a model, you can run an interactive demo app of it (CAML on MIMIC-III top-50, for example) by running
```
$ streamlit run app.py -- --config_path configs/demo/multi_mimic3_50.yml  # CAML, MultiResCNN, DCAN, Fusion on MIMIC-III top-50
```
You can write own config file specifying modules as same as in pre-processing and training


## Results
- MIMIC-III full

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@8        |        P@15        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.835&plusmn;0.001 | 0.974&plusmn;0.000 | 0.034&plusmn;0.001 | 0.420&plusmn;0.006 | 0.619&plusmn;0.002 | 0.474&plusmn;0.004 |
| CAML         | 0.893&plusmn;0.002 | 0.985&plusmn;0.000 | 0.056&plusmn;0.006 | 0.506&plusmn;0.006 | 0.704&plusmn;0.001 | 0.555&plusmn;0.001 |
| MultiResCNN  | 0.912&plusmn;0.004 | 0.987&plusmn;0.000 | 0.078&plusmn;0.005 | 0.555&plusmn;0.004 | 0.741&plusmn;0.002 | 0.589&plusmn;0.002 |
| DCAN         | 0.848&plusmn;0.009 | 0.979&plusmn;0.001 | 0.066&plusmn;0.005 | 0.533&plusmn;0.006 | 0.721&plusmn;0.001 | 0.573&plusmn;0.000 |
| TransICD     | 0.886&plusmn;0.010 | 0.983&plusmn;0.002 | 0.058&plusmn;0.001 | 0.497&plusmn;0.001 | 0.666&plusmn;0.000 | 0.524&plusmn;0.001 |
| Fusion       | 0.910&plusmn;0.003 | 0.986&plusmn;0.000 | 0.081&plusmn;0.002 | 0.560&plusmn;0.003 | 0.744&plusmn;0.002 | 0.589&plusmn;0.001 |

- MIMIC-III top-50

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@5        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.913&plusmn;0.002 | 0.936&plusmn;0.002 | 0.627&plusmn;0.001 | 0.693&plusmn;0.003 | 0.649&plusmn;0.001 |
| CAML         | 0.918&plusmn;0.000 | 0.942&plusmn;0.000 | 0.614&plusmn;0.005 | 0.690&plusmn;0.001 | 0.661&plusmn;0.002 |
| MultiResCNN  | 0.928&plusmn;0.001 | 0.950&plusmn;0.000 | 0.652&plusmn;0.006 | 0.720&plusmn;0.002 | 0.674&plusmn;0.001 |
| DCAN         | 0.934&plusmn;0.001 | 0.953&plusmn;0.001 | 0.651&plusmn;0.010 | 0.724&plusmn;0.005 | 0.682&plusmn;0.003 |
| TransICD     | 0.917&plusmn;0.002 | 0.939&plusmn;0.001 | 0.602&plusmn;0.002 | 0.679&plusmn;0.001 | 0.643&plusmn;0.001 |
| Fusion       | 0.932&plusmn;0.001 | 0.952&plusmn;0.000 | 0.664&plusmn;0.003 | 0.727&plusmn;0.003 | 0.679&plusmn;0.001 |

- MIMIC-III full (old)

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@8        |        P@15        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.833&plusmn;0.003 | 0.974&plusmn;0.000 | 0.027&plusmn;0.005 | 0.419&plusmn;0.006 | 0.612&plusmn;0.004 | 0.467&plusmn;0.001 |
| CAML         | 0.880&plusmn;0.003 | 0.983&plusmn;0.000 | 0.057&plusmn;0.000 | 0.502&plusmn;0.002 | 0.698&plusmn;0.002 | 0.548&plusmn;0.001 |
| MultiResCNN  | 0.905&plusmn;0.003 | 0.986&plusmn;0.000 | 0.076&plusmn;0.002 | 0.551&plusmn;0.005 | 0.738&plusmn;0.003 | 0.586&plusmn;0.003 |
| DCAN         | 0.837&plusmn;0.005 | 0.977&plusmn;0.001 | 0.063&plusmn;0.002 | 0.527&plusmn;0.002 | 0.721&plusmn;0.001 | 0.572&plusmn;0.001 |
| TransICD     | 0.882&plusmn;0.010 | 0.982&plusmn;0.001 | 0.059&plusmn;0.008 | 0.495&plusmn;0.005 | 0.663&plusmn;0.007 | 0.521&plusmn;0.006 |
| Fusion       | 0.910&plusmn;0.003 | 0.986&plusmn;0.000 | 0.076&plusmn;0.007 | 0.555&plusmn;0.008 | 0.744&plusmn;0.003 | 0.588&plusmn;0.003 |

- MIMIC-III top-50 (old)

| Model        |     macro AUC      |     micro AUC      |      macro F1      |      micro F1      |         P@5        |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CNN          | 0.892&plusmn;0.003 | 0.920&plusmn;0.003 | 0.583&plusmn;0.006 | 0.652&plusmn;0.008 | 0.627&plusmn;0.007 |
| CAML         | 0.865&plusmn;0.017 | 0.899&plusmn;0.008 | 0.495&plusmn;0.035 | 0.593&plusmn;0.020 | 0.597&plusmn;0.016 |
| MultiResCNN  | 0.898&plusmn;0.006 | 0.928&plusmn;0.003 | 0.590&plusmn;0.012 | 0.666&plusmn;0.013 | 0.638&plusmn;0.005 |
| DCAN         | 0.915&plusmn;0.002 | 0.938&plusmn;0.001 | 0.614&plusmn;0.001 | 0.690&plusmn;0.002 | 0.653&plusmn;0.004 |
| TransICD     | 0.895&plusmn;0.003 | 0.924&plusmn;0.002 | 0.541&plusmn;0.010 | 0.637&plusmn;0.003 | 0.617&plusmn;0.005 |
| Fusion       | 0.904&plusmn;0.002 | 0.930&plusmn;0.001 | 0.606&plusmn;0.009 | 0.677&plusmn;0.003 | 0.640&plusmn;0.001 |


## Authors
(in alphabetical order)
- Abheesht Sharma [@abheesht17](https://github.com/abheesht17)
- Juyong Kim [@dalgu90](https://github.com/dalgu90)
- Suhas Shanbhogue [@SuhasShanbhogue](https://github.com/SuhasShanbhogue)


## Cite this work
```
@InProceeding{juyong2022anemic,
  title = {AnEMIC: A Framework for Benchmarking ICD Coding Models},
  author = {Kim, Juyong and Sharma, Abheesht and Shanbhogue, Suhas and Ravikumar, Pradeep and Weiss, Jeremy C},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP), System Demonstrations},
  year = {2022},
  publisher = {ACL},
  url = {https://github.com/dalgu90/icd-coding-benchmark},
}
```

[^1]: Also referred to as medical coding, clinical coding, or simply ICD coding in other literature. They may have different meanings in detail.
[^2]: Mullenbach, et al., Explainable Prediction of Medical Codes from Clinical Text, NAACL 2018 ([paper](https://arxiv.org/abs/1802.05695), [code](https://github.com/jamesmullenbach/caml-mimic))
[^3]: Li and Yu, ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network, AAAI 2020 ([paper](https://arxiv.org/abs/1912.00862), [code](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network))
[^4]: Ji, et al., Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text, Clinical NLP Workshop 2020 ([paper](https://aclanthology.org/2020.clinicalnlp-1.8/), [code](https://github.com/shaoxiongji/DCAN))
[^5]: Biswas, et al., TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding, AIME 2021 ([paper](https://arxiv.org/abs/2104.10652), [code](https://github.com/AIMedLab/TransICD))
[^6]: Luo, et al., Fusion: Towards Automated ICD Coding via Feature Compression, ACL 2020 Findings ([paper](https://aclanthology.org/2021.findings-acl.184/), [code](https://github.com/machinelearning4health/Fusion-Towards-Automated-ICD-Coding))
