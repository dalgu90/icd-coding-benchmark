import argparse

from src.modules.preprocessing_pipelines import *
from src.utils.configuration import Config
from src.utils.mapper import ConfigMapper

# Command line arguments
parser = argparse.ArgumentParser(description="Preprocessing datasets")
parser.add_argument(
    "--config_path", type=str, action="store", help="Path to the config file"
)
args = parser.parse_args()

# Config
config = Config(path=args.config_path)

# Preprocess!
preprocessing = ConfigMapper.get_object(
    "preprocessing_pipelines", config.preprocessing.name
)(config.preprocessing.params)
preprocessing.preprocess()
