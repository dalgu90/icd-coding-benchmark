"""All criterion functions."""
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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


@ConfigMapper.map("losses", "BinaryCrossEntropyWithLabelSmoothingLoss")
class BinaryCrossEntropyWithLabelSmoothingLoss(BCEWithLogitsLoss):
    def __init__(self, config):
        self.config = config
        super().__init__(**(config.as_dict() if config else {}))

    def forward(self, input, target):
        if target.dtype != torch.float:
            target = target.float()
        target = target * (
            1 - self.config.alpha
        ) + self.config.alpha / target.size(1)
        return super().forward(input=input, target=target)
