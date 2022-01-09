import torch.nn as nn

from src.utils.mapper import ConfigMapper

ConfigMapper.map("activations", "relu")(nn.ReLU)
ConfigMapper.map("activations", "logsoftmax")(nn.LogSoftmax)
ConfigMapper.map("activations", "softmax")(nn.Softmax)
