import torch
import numpy as np

class MahalanobisScorer:
    def __init__(self, device='cuda', eps=1e-6):
        self.device = device
        self.eps = eps
        self.means = None
        self.inv_covs = None
        
    def fit(self, features, labels, num_classes):
        self.means = []
        self.inv_covs = []
        feat_dim = features.shappe[1]
        
        for c in range(num_classes):
            mask = labels == c
            class_feats = features[mask]
            if len(class_feats) < 2:
                continue
            mean = class_feats.mean(dim=0)
            centered = class_feats - mean
            cov = (centered.T() @ centered) / (len(class_feats) - 1)
            cov += self.eps * torch.eye(feat_dim, device=self.device)
            inv_cov = torch.linalg.inv(cov)
            self.means.append(mean)
            self.inv_covs.append(inv_cov)
    
    def compute_scores(self, features):
        batch_scores = []
        for feat in features:
            min_dist = float('inf')
            for mean, inv_cov in zip(self.means, self.inv_covs):
                diff = feat - mean
                dist = diff.T() @ inv_cov @ diff
                if dist < min_dist:
                    min_dist = dist
            batch_scores.append(min_dist.item())
        return np.array(batch_scores)
