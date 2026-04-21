import torch
import numpy as np
import tqdm as tqdm
from sklearn.metrics import roc_curve, auc
from .threshold_selector import select_threshold

class MahalanobisPipeline:
    def __init__(self, sptnet_wrapper, scorer, device='cuda'):
        self.sptnet = sptnet_wrapper
        self.scorer = scorer
        self.device = device
        self.threshold = None
        
    def compute_stats_from_loader(self, train_loader, num_classes):
        all_features = []
        all_labels = []
        for images, labels in tqdm(train_loader, desc="Extracting features for stats"):
            images = images.to(self.device)
            features = self.sptnet.extract_features(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
        features = torch.cat(all_features).to(self.device)
        labels = torch.cat(all_labels).to(self.device)
        self.scorer.fit(features, labels, num_classes)
        
    def compute_scores_for_loader(self, loader):
        all_scores = []
        for images, _ in tqdm(loader, desc="Computing Mahalanobis scores"):
            images = images.to(self.device)
            features = self.sptnet.extract_features(images)
            scores = self.scorer.compute_scores(features)
            all_scores.extend(scores)
        return np.array(all_scores)
    
    def select_threshold(self, val_loader, method='tpr', desired_tpr=0.95):
        id_scores = self.compute_scores_for_loader(val_loader)
        self.threshold = select_threshold(id_scores, method, desired_tpr)
        print(f"Threshold selected: {self.threshold}")
        
    def evaluate(self, test_id_loader, test_ood_loader):
        id_scores = self.compute_scores_for_loader(test_id_loader)
        ood_scores = self.compute_scores_for_loader(test_ood_loader)
        
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        
        fpr, tpr, _ = roc_curve(all_labels, -all_scores)
        roc_auc = auc(fpr, tpr)
        
        idx = np.argmax(tpr >= 0.95) if np.any(tpr >= 0.95) else len(tpr) - 1
        fpr95 = fpr[idx]
        return {'auc': roc_auc, 'fpr95': fpr95}
