"""Metrics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import roc_curve, auc

from src.utils.configuration import Config
from src.utils.mapper import ConfigMapper

ConfigMapper.map("metrics", "sklearn_f1")(f1_score)
ConfigMapper.map("metrics", "sklearn_p")(precision_score)
ConfigMapper.map("metrics", "sklearn_r")(recall_score)
ConfigMapper.map("metrics", "sklearn_roc")(roc_auc_score)
ConfigMapper.map("metrics", "sklearn_acc")(accuracy_score)
ConfigMapper.map("metrics", "sklearn_mse")(mean_squared_error)


def to_np_array(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    return array

class Metric:
    def __init__(self, config):
        if isinstance(config, dict):
            config = Config(dic=config)
        self.config = config

    def __call__(self, y_true, y_pred=None, p_pred=None):
        return self.forward(y_true, y_pred, p_pred)

    def forward(self, y_true, y_pred=None, p_pred=None):
        raise NotImplementedError("This is the base class for metrics")

def load_metric(config_dict):
    metric_params = config_dict.get('params', None)
    metric_class = config_dict.get('class', config_dict.get('name'))
    return ConfigMapper.get_object("metrics", metric_class)(metric_params)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

@ConfigMapper.map("metrics", "macro_f1")
class MacroF1(Metric):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, y_true, y_pred=None, p_pred=None):
        yhat = to_np_array(y_pred if y_pred is not None else p_pred.round())
        y = to_np_array(y_true)
        return macro_f1(yhat, y)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

@ConfigMapper.map("metrics", "micro_f1")
class MicroF1(Metric):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, y_true, y_pred=None, p_pred=None):
        yhatmic = to_np_array(y_pred if y_pred is not None else p_pred.round())
        ymic = to_np_array(y_true)
        yhatmic = yhatmic.ravel()
        ymic = ymic.ravel()
        return micro_f1(yhatmic, ymic)

@ConfigMapper.map("metrics", "macro_auc")
class MacroAUC(Metric):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, y_true, y_pred=None, p_pred=None):
        y, yhat_raw = y_true, p_pred

        # From auc_metrics()
        if yhat_raw.shape[0] <= 1:
            return
        fpr = {}
        tpr = {}
        roc_auc = {}
        #get AUC for each label individually
        relevant_labels = []
        auc_labels = {}
        for i in range(y.shape[1]):
            #only if there are true positives for this label
            if y[:,i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
                if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                    auc_score = auc(fpr[i], tpr[i])
                    if not np.isnan(auc_score):
                        auc_labels["auc_%d" % i] = auc_score
                        relevant_labels.append(i)

        #macro-AUC: just average the auc scores
        aucs = []
        for i in relevant_labels:
            aucs.append(auc_labels['auc_%d' % i])
        return np.mean(aucs)

@ConfigMapper.map("metrics", "micro_auc")
class MicroAUC(Metric):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, y_true, y_pred=None, p_pred=None):
        assert p_pred is not None
        yhat_raw_mic = to_np_array(p_pred).ravel()
        ymic = to_np_array(y_true).ravel()

        # From auc_metrics()
        fpr, tpr, _ = roc_curve(ymic, yhat_raw_mic)
        return auc(fpr, tpr)

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

##############
# AT-K
##############

@ConfigMapper.map("metrics", "recall_at_k")
class RecallAtK(Metric):
    def __init__(self, config):
        super().__init__(config)
        self.k = self.config.k

    def forward(self, y_true, y_pred=None, p_pred=None):
        assert p_pred is not None
        y_true = to_np_array(y_true)
        p_pred = to_np_array(p_pred)
        return recall_at_k(p_pred, y_true, 5)

@ConfigMapper.map("metrics", "prec_at_k")
class PrecAtK(Metric):
    def __init__(self, config):
        super().__init__(config)
        self.k = self.config.k

    def forward(self, y_true, y_pred=None, p_pred=None):
        assert p_pred is not None
        y_true = to_np_array(y_true)
        p_pred = to_np_array(p_pred)
        return precision_at_k(p_pred, y_true, 5)

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
