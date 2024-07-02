import pandas as pd
import numpy as np
from aequitas.group import Group    
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def evaluate(y_test, predicted):
    """Evaluate the model using the confusion matrix and some metrics.
    ------------------------------------------------------ 
    Args:
        targets (list): list of true values
        predicted (list): list of predicted values
    ------------------------------------------------------ 
    Returns:
        cm (np.array): confusion matrix
        accuracy (float): accuracy of the model
        precision (float): precision of the model
        recall (float): recall of the model
        fpr (float): false positive rate of the model
        f1_score (float): f1 score of the model
        auc (float): area under the curve of the model
    """
    y_pred = []
    y_true = []
    y_pred.extend(predicted)
    y_true.extend(y_test)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = 0 if (tp + tn + fp + fn)==0 else (tp + tn) / (tp + tn + fp + fn)
    precision = 0 if (tp + fp)==0 else tp / (tp + fp) 
    recall = 0 if (tp + fn)==0 else tp / (tp + fn)
    fpr = 0 if (fp + tn)==0 else fp / (fp + tn)
    tnr = 0 if (tn + fp)==0 else tn / (tn + fp)
    f1 = 0 if (precision + recall)==0 else 2 * (precision * recall) / (precision + recall) 
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    return {
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "recall": recall,
        "tnr": tnr,
        "accuracy": accuracy,
        "precision": precision,
        "f1_score": f1,
        "auc": auc
    }

def evaluate_aequitas(x_test, y_test, predictions):
    """Evaluate the model using the Aequitas library.
    ------------------------------------------------------
    Args:
        x_test (pd.DataFrame): dataframe with the test features
        y_test (pd.Series): series with the test labels
        predictions (np.array): array with the predictions
    ------------------------------------------------------
    Returns:
        threshold (float): threshold for the model
        fpr@5FPR (float): false positive rate of the model
        recall@5FPR (float): recall of the model
        tnr@5FPR (float): true negative rate of the model
        accuracy@5FPR (float): accuracy of the model
        precision@5FPR (float): precision of the model
        f1_score@5FPR (float): f1 score of the model
        fpr_ratio (float): false positive rate ratio of the model
        fnr_ratio (float): false negative rate ratio of the model
        recall_older (float): recall of the older group
        recall_younger (float): recall of the younger group
        fpr_older (float): false positive rate of the older group
        fpr_younger (float): false positive rate of the younger group
        fnr_older (float): false negative rate of the older group
        fnr_younger (float): false negative rate of the younger group
    """
    fprs, _, thresholds = roc_curve(y_test, predictions)    
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    preds_binary = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_binary).ravel()
    cm_recall = 0 if (tp + fn)==0 else tp / (tp + fn)
    cm_tnr = 0 if (tn + fp)==0 else tn / (tn + fp)
    cm_accuracy = 0 if (tp + tn + fp + fn)==0 else (tp + tn) / (tp + tn + fp + fn)
    cm_precision = 0 if (tp + fp)==0 else tp / (tp + fp) 
    cm_fpr = 0 if (fp + tn)==0 else fp / (fp + tn)
    cm_f1 = 0 if (cm_precision + cm_recall)==0 else 2 * (cm_precision * cm_recall) / (cm_precision + cm_recall)        
    aequitas_df = pd.DataFrame(
        {
            "age": (x_test["customer_age"]>=50).map({True: "Older", False: "Younger"}),
            "preds": preds_binary,
            "y": y_test.values if isinstance(y_test, pd.Series) else y_test
        }
    )
    g = Group()
    aequitas_df["score"] = aequitas_df["preds"]
    aequitas_df["label_value"] = aequitas_df["y"]
    aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=["age"])[0]
    recall_older = aequitas_results[aequitas_results["attribute_value"] == "Older"][["tpr"]].values[0][0]
    recall_younger = aequitas_results[aequitas_results["attribute_value"] == "Younger"][["tpr"]].values[0][0]
    fpr_older = aequitas_results[aequitas_results["attribute_value"] == "Older"][["fpr"]].values[0][0]
    fpr_younger = aequitas_results[aequitas_results["attribute_value"] == "Younger"][["fpr"]].values[0][0]
    fnr_older = 1 - recall_older
    fnr_younger = 1 - recall_younger
    if fpr_younger >= fpr_older:
        fpr_ratio = fpr_younger and fpr_older/fpr_younger or 0
    else:
        fpr_ratio = fpr_older and fpr_younger/fpr_older or 0
    if fnr_younger > fnr_older:
        fnr_ratio = fnr_older and fnr_older/fnr_younger or 0
    else:
        fnr_ratio = fnr_older and fnr_younger/fnr_older or 0
    return {
        "threshold": threshold,
        "fpr@5FPR": cm_fpr,
        "recall@5FPR": cm_recall,
        "tnr@5FPR": cm_tnr,
        "accuracy@5FPR": cm_accuracy,
        "precision@5FPR": cm_precision,
        "f1_score@5FPR": cm_f1,
        "fpr_ratio": fpr_ratio,
        "fnr_ratio": fnr_ratio,
        "recall_older": recall_older,
        "recall_younger": recall_younger,
        "fpr_older": fpr_older,
        "fpr_younger": fpr_younger,
        "fnr_older": fnr_older,
        "fnr_younger": fnr_younger
    }
    