import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix


def precision_at_k(y_true, y_pred, k):
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    precisions = []

    for i in range(y_true.shape[0]):
        true_positives = np.sum(y_true[i, top_k_preds[i]])
        precisions.append(true_positives / k)

    return np.mean(precisions)


def average_precision_at_k(y_true, y_pred, k):
    res = 0
    for i in range(k):
        res += precision_at_k(y_true, y_pred, i + 1)
    res /= k
    return res


def _binary_tpr_fpr(y_true, y_pred, positive_label=1):
    """
    Compute TPR/FPR for binary one-page evaluation.

    Convention:
      positive = monitored page (label = positive_label)
      negative = non-monitored (other label, typically 0)

    Returns
    -------
    tpr, fpr, tp, fp, tn, fn
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_all = np.unique(np.concatenate([unique_true, unique_pred]))

    if len(unique_all) > 2:
        raise ValueError(
            f"TPR/FPR require binary labels (one-page datasets), got labels={unique_all}"
        )

    # For your one-page generator: monitored = 1, non-monitored = 0
    negative_label = 0 if positive_label != 0 else 1

    # Force confusion-matrix ordering:
    # labels=[0,1] => rows true, cols pred => [[TN, FP],[FN, TP]] when positive_label=1
    if positive_label == 1 and negative_label == 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    elif positive_label == 0 and negative_label == 1:
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Only binary labels {0,1} are supported for TPR/FPR.")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr, int(tp), int(fp), int(tn), int(fn)


def measurement(y_true, y_pred, eval_metrics, num_tabs=1, positive_label=1):
    """
    Calculate evaluation metrics for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    eval_metrics (list): List of evaluation metrics to calculate.
    num_tabs (int): Number of tabs (for multi-label scenarios).
    positive_label (int): Positive class label for binary TPR/FPR.
                          For one-page datasets, this should be the monitored class,
                          which your generator sets to 1.

    Returns:
    dict: Dictionary of calculated metrics.
    """
    results = {}

    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred), 4)

        elif eval_metric == "Precision":
            results[eval_metric] = round(
                precision_score(y_true, y_pred, average="macro", zero_division=0), 4
            )

        elif eval_metric == "Recall":
            results[eval_metric] = round(
                recall_score(y_true, y_pred, average="macro", zero_division=0), 4
            )

        elif eval_metric == "F1-score":
            results[eval_metric] = round(
                f1_score(y_true, y_pred, average="macro", zero_division=0), 4
            )

        elif eval_metric == "P@min":
            per_class_precision = precision_score(
                y_true, y_pred, average=None, zero_division=0
            )
            results[eval_metric] = round(np.min(per_class_precision), 4)

        elif eval_metric == "r-Precision":
            results[eval_metric] = round(cal_r_precision(y_true, y_pred), 4)

        elif eval_metric == "AUC":
            results[eval_metric] = round(roc_auc_score(y_true, y_pred, average="macro"), 4)

        elif eval_metric == "TPR":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["TPR"] = round(tpr, 4)

        elif eval_metric == "FPR":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["FPR"] = round(fpr, 4)

        elif eval_metric == "TP":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["TP"] = tp

        elif eval_metric == "FP":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["FP"] = fp

        elif eval_metric == "TN":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["TN"] = tn

        elif eval_metric == "FN":
            tpr, fpr, tp, fp, tn, fn = _binary_tpr_fpr(
                y_true, y_pred, positive_label=positive_label
            )
            results["FN"] = fn

        elif eval_metric.startswith("P@"):
            k = int(eval_metric[2:])
            results[eval_metric] = round(precision_at_k(y_true, y_pred, k), 4)

        elif eval_metric.startswith("AP@"):
            k = int(eval_metric[3:])
            results[eval_metric] = round(average_precision_at_k(y_true, y_pred, k), 4)

        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")

    return results


def cal_r_precision(y_true, y_pred, base_r=20):
    """
    Calculate r-Precision for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    base_r (int): Base value for r-Precision calculation.

    Returns:
    float: Calculated r-Precision value.
    """
    open_class = y_true.max()
    web2tp = {}
    web2fp = {}
    web2wp = {}

    for web in range(open_class + 1):
        web2tp[web] = 0
        web2fp[web] = 0
        web2wp[web] = 0

    for index in range(len(y_true)):
        cur_true = y_true[index]
        cur_pred = y_pred[index]
        if cur_true == cur_pred:
            web2tp[cur_pred] += 1
        else:
            if cur_true == open_class:
                web2fp[cur_pred] += 1
            else:
                web2wp[cur_pred] += 1

    res = 0
    for web in range(open_class):
        denominator = web2tp[web] + base_r * web2fp[web] + web2wp[web]
        if denominator == 0:
            continue
        res += web2tp[web] / denominator
    res /= open_class
    return res


def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad
