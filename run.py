# Imports
import argparse
import os

import pandas

from src.datasets import *
from src.models import *
from src.trainers import *
from src.utils.configuration import Config
from src.utils.import_related_ops import pandas_related_ops
from src.utils.logger import Logger
from src.utils.mapper import ConfigMapper
from src.utils.misc import seed

pandas_related_ops()


# Command line arguments
parser = argparse.ArgumentParser(description="Train or test the model")
parser.add_argument(
    "--config_path", type=str, action="store", help="Path to the config file"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Whether to use validation data or test data",
    default=False,
)
args = parser.parse_args()

# Config
config = Config(path=args.config_path)

if not args.test:  # Training
    # Seed
    seed(config.trainer.params.seed)

    # Load dataset
    train_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.train
    )
    val_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.val
    )

    # Model
    model = ConfigMapper.get_object("models", config.model.name)(
        **config.model.params.as_dict()
    )

    # Trainer
    trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
        config.trainer.params
    )

    # Train!
    trainer.train(model, train_data, val_data)
else:  # Test
    # Load dataset
    test_data = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.test
    )

    # Model
    model = ConfigMapper.get_object("models", config.model.name)(
        **config.model.params.as_dict()
    )

    # Trainer
    trainer = ConfigMapper.get_object("trainers", config.trainer.name)(
        config.trainer.params
    )

    # Test!
    trainer.test(model, test_data)
