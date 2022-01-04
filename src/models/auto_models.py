from transformers import AutoModelForSequenceClassification
from src.utils.mapper import configmapper

configmapper.map("models", "automodelforsequenceclassification")(AutoModelForSequenceClassification)
