"All criterion functions."
from torch.nn import MSELoss, CrossEntropyLoss
from src.utils.mapper import ConfigMapper

ConfigMapper.map("losses", "mse")(MSELoss)
ConfigMapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)
