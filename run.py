# Imports
import argparse
import os

from src.utils.configuation import Config
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.misc import seed


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

# Logger
logger = None
# logger = Logger(
    # log_path=os.path.join(
        # log_dir,
        # args.config_dir.strip("/").split("/")[-1]
        # + ("" if args.validation else "_orig"),
    # )
# )


if not args.test: # Training
    # Seed
    seed(config.seed)

    # Load dataset
    train_data = configmapper.get_object("datasets", config.dataset.name)(config.dataset.parameters.train)
    val_data = configmapper.get_object("datasets", config.dataset.name)(config.dataset.parameters.val)

    # Model
    model = configmapper.get_object("models", config.model.name)(config.model.parameters)

    # Trainer
    trainer = configmapper.get_object("trainers", config.trainer.name)(config.trainer.parameters)

    # Train!
    trainer.train(model, train_data, val_data, logger)
else: # Test
    # Load dataset
    test_data = configmapper.get_object("datasets", config.dataset.name)(config.dataset.parameters.test)

    # Model
    model = configmapper.get_object("models", config.model.name)(config.model.parameters)

    # Trainer
    trainer = configmapper.get_object("trainers", config.trainer.name)(config.trainer.parameters)

    # Train!
    trainer.eval(model, test_data, logger)
