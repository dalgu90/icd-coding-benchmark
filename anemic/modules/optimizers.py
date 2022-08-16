"""Method containing activation functions"""
from torch.optim import SGD, Adam, AdamW

from anemic.utils.mapper import ConfigMapper

ConfigMapper.map("optimizers", "adam")(Adam)
ConfigMapper.map("optimizers", "adam_w")(AdamW)
ConfigMapper.map("optimizers", "sgd")(SGD)
