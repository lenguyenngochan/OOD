import numpy as np
from sklearn.metrics import roc_curve

def select_threshold(id_scores, ood_scores=None, method='tpr', desired_tpr=0.95):
    if method == 'tpr':
        threshold = np.percentile(id_scores, 100 * (1 - desired_tpr))
        return threshold
    elif method == 'youden' and ood_scores is not None:
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        
        fpr, tpr, thresholds = roc_curve(all_labels, -all_scores)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        threshold = -thresholds[best_idx]
        return threshold
    else:
        raise ValueError("Invalid method or missing OOD scores")