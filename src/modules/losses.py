"""All criterion functions."""
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.nn.functional as F

from src.utils.mapper import ConfigMapper


ConfigMapper.map("losses", "mse")(MSELoss)
ConfigMapper.map("losses", "CrossEntropyLoss")(CrossEntropyLoss)

@ConfigMapper.map("losses", "BinaryCrossEntropyLoss")
class BinaryCrossEntropyLoss(BCEWithLogitsLoss):
    def __init__(self, config):
        self.config = config
        super().__init__(**(config.as_dict() if config else {}))

    def forward(self, input, target):
        if target.dtype != torch.float:
            target = target.float()
        return super().forward(input=input, target=target)
