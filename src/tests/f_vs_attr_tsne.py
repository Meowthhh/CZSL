import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_F_vs_attributes(F_pred, attr_matrix, y_true=None, seen_classes=None, unseen_classes=None):
    F_pred_np = F_pred.cpu().numpy()
    A_np = attr_matrix.cpu().numpy()
    
    all_data = np.concatenate([F_pred_np, A_np], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", random_state=42)
    embedded = tsne.fit_transform(all_data)

    n_preds = F_pred_np.shape[0]
    preds_emb = embedded[:n_preds]
    attrs_emb = embedded[n_preds:]

    plt.figure(figsize=(10, 8))
    plt.scatter(preds_emb[:, 0], preds_emb[:, 1], c='blue', label="Predicted F", alpha=0.6)

    for i, emb in enumerate(attrs_emb):
        if seen_classes and i in seen_classes:
            color = 'green'
        elif unseen_classes and i in unseen_classes:
            color = 'red'
        else:
            color = 'gray'
        plt.scatter(emb[0], emb[1], color=color, marker='X', s=100)
        plt.text(emb[0], emb[1]+0.1, str(i), fontsize=6, color=color)

    plt.title("t-SNE: Predicted Features and Class Attribute Vectors")
    plt.legend(["Predicted F", "Class Attributes"])
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    torch.manual_seed(0)

    attr_matrix = torch.randn(50, 85)
    F_pred = torch.randn(100, 85)
    y_true = torch.randint(0, 50, (100,))
    seen_classes = list(range(0, 10))
    unseen_classes = list(range(10, 50))

    plot_tsne_F_vs_attributes(F_pred, attr_matrix, y_true, seen_classes, unseen_classes)