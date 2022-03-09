import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


class GraphWriterBase:
    def __init__(self, config):
        self.config = config

    def writer_scalar(self, name, value, step=None):
        raise NotImplementedError()


@ConfigMapper.map("graph_writers", "tensorboard")
class TensorboardGraphWriter(GraphWriterBase):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.debug(f"Initializing {cls_name} with config: {config}")

        super().__init__(config)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def write_scalar(self, name, value, step=None):
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)
        self.writer.flush()


@ConfigMapper.map("graph_writers", "wandb")
class WandBGraphWriter(GraphWriterBase):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.debug(f"Initializing {cls_name} with config: {config}")

        super().__init__(config)

    def writer_scalar(self, name, value, step=None):
        raise NotImplementedError()
