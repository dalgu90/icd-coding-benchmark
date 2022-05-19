"""All criterion functions."""
import json
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from src.utils.file_loaders import load_json
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

        config_dict = config.as_dict()
        self.alpha = config_dict.pop("alpha")

        super().__init__(**(config_dict if config else {}))

    def forward(self, input, target):
        if target.dtype != torch.float:
            target = target.float()
        target = target * (1 - self.alpha) + self.alpha / target.size(1)
        return super().forward(input=input, target=target)


@ConfigMapper.map("losses", "LDAMLoss")
class LDAMLoss(BCEWithLogitsLoss):
    def __init__(self, config):
        config_dict = config.as_dict()

        label_freq_path = os.path.join(
            config_dict.pop("label_freq_json_dir"),
            config_dict.pop("label_freq_json_name"),
        )
        label_freq = list(load_json(label_freq_path).values())

        self.class_margin = (
            torch.tensor(label_freq, dtype=torch.float32) ** 0.25
        )
        self.class_margin = self.class_margin.masked_fill(
            self.class_margin == 0, 1
        )
        self.class_margin = 1.0 / self.class_margin

        self.C = config_dict.pop("C")

        super().__init__(**(config_dict if config_dict else {}))

    def forward(self, input, target):
        device = input.get_device()
        target = target.to(device)
        self.class_margin = self.class_margin.to(device)
        if target.dtype != torch.float:
            target = target.float()

        ldam_input = (
            input
            - target * Variable(self.class_margin, requires_grad=False) * self.C
        )
        return super().forward(input=ldam_input, target=target)
