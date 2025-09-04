import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_cosine_loss_with_negatives(logits, seen_attributes, labels):
    labels = labels.squeeze()  # shape: (64,)

    # Positive pairs
    positive_attributes = seen_attributes[labels]  # shape: (64, 85)
    positive_targets = torch.ones(logits.size(0), device=logits.device)  # shape: (64,)

    # Negative pairs
    neg_labels = torch.randint(low=0, high=seen_attributes.size(0), size=labels.size(), device=logits.device)
    while_mask = neg_labels == labels
    while while_mask.any():
        neg_labels[while_mask] = torch.randint(low=0, high=seen_attributes.size(0), size=while_mask.sum().size(), device=logits.device)
        while_mask = neg_labels == labels

    negative_attributes = seen_attributes[neg_labels]  # shape: (64, 85)
    negative_targets = -torch.ones(logits.size(0), device=logits.device)  # shape: (64,)

    # Combine positive and negative examples
    input_vectors = torch.cat([logits, logits], dim=0)  # shape: (128, 85)
    target_vectors = torch.cat([positive_attributes, negative_attributes], dim=0)  # shape: (128, 85)
    targets = torch.cat([positive_targets, negative_targets], dim=0)  # shape: (128,)

    # Compute cosine embedding loss
    cosine_loss = nn.CosineEmbeddingLoss()
    loss = cosine_loss(input_vectors, target_vectors, targets)
    return loss

class MetricLearningModel(nn.Module):
    def __init__(self, attr_dim=85, metric_dim=85, lambda_attr=1.0, margin=1.0):
        super().__init__()
        self.WA = nn.Linear(attr_dim, metric_dim, bias=False)  # Mahalanobis-like projection
        self.lambda_attr = lambda_attr
        self.margin = margin

    def forward(self, logits, seen_attributes, labels):
        # --- Project image and attribute embeddings ---
        F_proj = self.WA(logits)                      # (B, m)
        A_proj = self.WA(seen_attributes)              # (S, m)

        # --- Get correct attribute projections per sample ---
        Y_true = A_proj[labels]                  # (B, m)

        # --- Hinge loss (metric learning) ---
        d_pos = pairwise_distance(F_proj, Y_true)  # (B,)

        # All distances between projected images and class attributes
        dists = torch.cdist(F_proj, A_proj)      # (B, S)

        # Create mask to exclude the correct class
        B, S = logits.shape[0], seen_attributes.shape[0]
        labels_exp = labels.unsqueeze(1).expand(-1, S)
        class_indices = torch.arange(S).to(labels.device)
        neg_mask = labels_exp != class_indices  # (B, S)

        # Mask out correct class (fill with inf)
        d_neg = torch.where(neg_mask, dists, torch.full_like(dists, float('inf')))
        d_neg_min, _ = torch.min(d_neg, dim=1)  # hardest negative distance

        # Hinge loss: max(0, margin + d_pos - d_neg_min)
        hinge_loss = F.relu(self.margin + d_pos - d_neg_min).mean()

        # --- Attribute prediction loss ---
        Y_true_full = seen_attributes[labels]     # (B, 85)
        attr_loss = F.mse_loss(logits, Y_true_full)    # Attribute regression loss

        # --- Final loss ---
        total_loss = hinge_loss + self.lambda_attr * attr_loss 
        return total_loss

def pairwise_distance(x1, x2):
    return torch.norm(x1 - x2, dim=1)

