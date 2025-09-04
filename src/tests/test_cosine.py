import torch
import torch.nn as nn

def compute_cosine_loss(logits, seen_attributes, labels):
    labels = labels.squeeze()  # (64,)
    target_attributes = seen_attributes[labels]  # (64, 85)
    target = torch.ones(logits.size(0), device=logits.device)  # (64,)
    cosine_loss = nn.CosineEmbeddingLoss()
    loss = cosine_loss(logits, target_attributes, target)

    print(labels)
    print()
    print(seen_attributes)
    print()
    print(target_attributes)

    return loss

def main():
    # Simulate batch size of 64, attribute dim of 85, and 10 seen classes
    batch_size = 3
    attr_dim = 5
    num_seen_classes = 10

    # Random seed for reproducibility
    torch.manual_seed(42)

    # Dummy model outputs: (64, 85)
    logits = torch.randn(batch_size, attr_dim)

    # Dummy seen class attribute vectors: (10, 85)
    seen_attributes = torch.randn(num_seen_classes, attr_dim)

    # Dummy labels (integers in [0, 9]): (64, 1)
    labels = torch.randint(0, num_seen_classes, (batch_size, 1))

    # Compute cosine loss
    loss = compute_cosine_loss(logits, seen_attributes, labels)

    print(f"CosineEmbeddingLoss: {loss.item():.4f}")


if __name__ == "__main__":
    main()