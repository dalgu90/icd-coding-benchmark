"""Metrics."""
from sklearn.metrics import (
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from src.utils.mapper import ConfigMapper

ConfigMapper.map("metrics", "sklearn_f1")(f1_score)
ConfigMapper.map("metrics", "sklearn_p")(precision_score)
ConfigMapper.map("metrics", "sklearn_r")(recall_score)
ConfigMapper.map("metrics", "sklearn_roc")(roc_auc_score)
ConfigMapper.map("metrics", "sklearn_acc")(accuracy_score)
ConfigMapper.map("metrics", "sklearn_mse")(mean_squared_error)
