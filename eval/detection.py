import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, confusion_matrix

def evaluate_detection(y_true: np.ndarray, y_pred_proba: np.ndarray, target_fpr: float = 0.01) -> dict:
    """
    Evaluates detection metrics enforcing a fixed false-positive budget.
    We threshold the probabilities such that False Positive Rate <= target_fpr on legitimate traffic.
    """
    if len(y_true) == 0:
         return {}
         
    # Sort predictions descending
    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    total_pos = tps[-1]
    total_neg = fps[-1]
    
    fprs = fps / total_neg if total_neg > 0 else np.zeros_like(fps)
    
    # Find the threshold index where FPR <= target_fpr
    valid_thresholds = np.where(fprs <= target_fpr)[0]
    best_idx = valid_thresholds[-1] if len(valid_thresholds) > 0 else 0
    
    # Threshold at the best FPR that matches the target budget
    best_threshold = y_pred_proba[desc_score_indices[best_idx]]
    y_pred_binary = (y_pred_proba >= best_threshold).astype(int)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recalls, precisions)
    
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
    
    return {
        "target_fpr": target_fpr,
        "actual_fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_pr": float(auc_pr),
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
    }
