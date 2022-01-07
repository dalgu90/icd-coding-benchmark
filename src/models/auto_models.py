from transformers import AutoModelForSequenceClassification
from src.utils.mapper import ConfigMapper

ConfigMapper.map("models", "automodelforsequenceclassification")(AutoModelForSequenceClassification)
