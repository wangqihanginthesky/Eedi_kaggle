import torch
import torch.nn as nn
import torch.nn.functional as F

class MultipleNegativeRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultipleNegativeRankingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negatives):
        # アンカーとポジティブの距離
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        
        # アンカーと各ネガティブの距離
        neg_dists = torch.stack([F.pairwise_distance(anchor, neg, p=2) for neg in negatives], dim=1)
        
        # margin付きのランキングロス計算
        losses = F.relu(pos_dist.unsqueeze(1) - neg_dists + self.margin)
        return losses.mean()