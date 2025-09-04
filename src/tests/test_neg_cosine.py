import torch
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

def main():
    # Simulate batch size of 64, attribute dim of 85, and 10 seen classes
    batch_size = 64
    attr_dim = 85
    num_seen_classes = 10

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Dummy model outputs
    logits = torch.randn(batch_size, attr_dim)

    # Dummy seen class attribute vectors
    seen_attributes = torch.randn(num_seen_classes, attr_dim)

    # Dummy labels (integers in [0, 9])
    labels = torch.randint(0, num_seen_classes, (batch_size, 1))

    # Compute loss with negatives
    loss = compute_cosine_loss_with_negatives(logits, seen_attributes, labels)

    print(f"CosineEmbeddingLoss with negatives: {loss.item():.4f}")

if __name__ == "__main__":
    main()