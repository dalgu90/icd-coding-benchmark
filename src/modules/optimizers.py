" Method containing activation functions"
from torch.optim import Adam, AdamW, SGD
from src.utils.mapper import ConfigMapper

ConfigMapper.map("optimizers", "adam")(Adam)
ConfigMapper.map("optimizers", "adam_w")(AdamW)
ConfigMapper.map("optimizers", "sgd")(SGD)
