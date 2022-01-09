"All criterion functions."
from torch.nn import CrossEntropyLoss, MSELoss

from src.utils.mapper import ConfigMapper

ConfigMapper.map("losses", "mse")(MSELoss)
ConfigMapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)
