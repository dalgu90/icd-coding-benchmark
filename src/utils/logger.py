import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.mapper import ConfigMapper


class LoggerBase:
    def __init__(self, config):
        self.config = config

    def writer_scalar(self, name, value, step=None):
        raise NotImplementedError()


@ConfigMapper.map("loggers", "tensorboard")
class TensorboardLogger(LoggerBase):
    def __init__(self, config):
        super().__init__(config)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def write_scalar(self, name, value, step=None):
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)
        self.writer.flush()


@ConfigMapper.map("loggers", "wandb")
class WandBLogger(LoggerBase):
    def __init__(self, config):
        super().__init__(config)

    def writer_scalar(self, name, value, step=None):
        raise NotImplementedError()
