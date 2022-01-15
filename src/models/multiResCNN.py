"""
    ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network , 2020
    https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from elmo.elmo import Elmo
import json
from utils import build_pretrain_embedding, load_embeddings
from math import floor

