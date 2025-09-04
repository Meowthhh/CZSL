import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random 
import os
import torch.nn as nn

def plot_tsne_F_vs_attributes(F_pred, attr_matrix, y_true=None, n_seen_classes=None, n_unseen_classes=None,
                              iteration=None, save=False, file_path="analysis\\tsne"):
    all_classes = list(range(0, 50))
    seen_classes = all_classes[:n_seen_classes]
    unseen_classes = all_classes[n_seen_classes:]

    F_pred_np = F_pred.cpu().numpy()
    A_np = attr_matrix.cpu().numpy()
    
    # Compute cosine similarity (B x C)
    F_norm = F_pred / F_pred.norm(dim=1, keepdim=True)
    A_norm = attr_matrix / attr_matrix.norm(dim=1, keepdim=True)
    sim = torch.matmul(F_norm, A_norm.T)
    pred_classes = torch.argmax(sim, dim=1).cpu().numpy()  # (B,)

    # Prepare data for t-SNE
    all_data = np.concatenate([F_pred_np, A_np], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", random_state=42)
    embedded = tsne.fit_transform(all_data)

    n_preds = F_pred_np.shape[0]
    preds_emb = embedded[:n_preds]
    attrs_emb = embedded[n_preds:]

    plt.figure(figsize=(12, 8))
    
    # Plot predicted features with predicted class labels
    for i, (x, y) in enumerate(preds_emb):
        plt.scatter(x, y, c='blue', alpha=0.6)
        plt.text(x, y, f"{pred_classes[i]}", fontsize=7, color='blue')

    # Plot attribute vectors
    for i, (x, y) in enumerate(attrs_emb):
        if seen_classes and i in seen_classes:
            color = 'green'
        elif unseen_classes and i in unseen_classes:
            color = 'red'
        else:
            color = 'gray'
        plt.scatter(x, y, color=color, s=100)
        plt.text(x, y, f"{i}", fontsize=6, color='black', fontweight='bold')

    plt.title("t-SNE: Predicted Features with Predicted Class Labels vs Class Attribute Vectors")
    plt.grid(True)

    if save:
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f"{file_path}\\tsne_{iteration}.png", dpi=300)
        plt.close()
    else:
        plt.show()


def compute_D_acc(discriminator, test_dataloader, seen_classes, novel_classes, task_no, batch_size=128, opt1='gzsl',
                  opt2='test_seen', psuedo_ft=None, psuedo_lb=None):
    """
    Compute the accuracy of the discriminator on the test set using cosine similarity and identifier projections
    :param discriminator: the discriminator model
    :param test_dataloader: the test dataloader
    :param seen_classes: the seen classes for each tasks
    :param novel_classes: the novel classes for each tasks
    :param task_no: current task number
    :param batch_size: batch size
    :param opt1: the type of the evaluation test space
    :param opt2: the type of the evaluation test set
    :param psuedo_ft: the pseudo features for the current task included (not used in our code)
    :param psuedo_lb: the pseudo labels for the current task included (not used in our code)
    """

    if psuedo_ft is not None:
        data = Data.TensorDataset(psuedo_ft, psuedo_lb)
        test_loader = Data.DataLoader(data, batch_size=batch_size)
    else:
        test_loader = test_dataloader.get_loader(opt2, batch_size=batch_size)
    att = test_dataloader.data['whole_attributes'].cuda()
    if opt1 == 'gzsl':
        search_space = np.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = test_dataloader.data['unseen_label']

    pred_label = []
    true_label = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
            features = F.normalize(features, p=2, dim=-1, eps=1e-12)
            if psuedo_ft is None:
                features = features.unsqueeze(1).repeat(1, search_space.shape[0], 1)
            else:
                features = features.squeeze(1).unsqueeze(1).repeat(1, search_space.shape[0], 1)
            semantic_embeddings = discriminator(att).cuda()
            semantic_embeddings = F.normalize(semantic_embeddings, p=2, dim=-1, eps=1e-12)
            cosine_sim = F.cosine_similarity(semantic_embeddings, features, dim=-1)
            predicted_label = torch.argmax(cosine_sim, dim=1)
            predicted_label = search_space[predicted_label.cpu()]
            pred_label = np.append(pred_label, predicted_label)
            true_label = np.append(true_label, labels.cpu().numpy())
    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0
    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]
    return acc


def compute_acc(model, test_dataloader, seen_classes, novel_classes, task_no, batch_size=128, opt1='gzsl',
                  opt2='test_seen', psuedo_ft=None, psuedo_lb=None, iteration=None):
    
    if psuedo_ft is not None:
        data = Data.TensorDataset(psuedo_ft, psuedo_lb)
        test_loader = Data.DataLoader(data, batch_size=batch_size)
    else:
        test_loader = test_dataloader.get_loader(opt2, batch_size=batch_size)
    att = test_dataloader.data['whole_attributes'].cuda()
    if opt1 == 'gzsl':
        search_space = np.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = test_dataloader.data['unseen_label']

    pred_label = []
    true_label = []
    model.eval()

    with torch.no_grad():

        # analysis = {} # For diagnosis (writing predictions to a file for each class)

        for features, labels in test_loader:
            iteration += 1

            features, labels = features.cuda(), labels.cuda()
            logits = model(features).cuda()
            f_pred = logits # For diagnosis (logits vs attributes tsne plot)
            logits = F.normalize(logits, p=2, dim=-1, eps=1e-12)
            if psuedo_ft is None:
                logits = logits.unsqueeze(1).repeat(1, search_space.shape[0], 1)
            else:
                logits = logits.squeeze(1).unsqueeze(1).repeat(1, search_space.shape[0], 1)
            semantic_embeddings = att
            semantic_embeddings = F.normalize(semantic_embeddings, p=2, dim=-1, eps=1e-12)
            cosine_sim = F.cosine_similarity(semantic_embeddings, logits, dim=-1)
            predicted_label = torch.argmax(cosine_sim, dim=1)
            predicted_label = search_space[predicted_label.cpu()]

            # if opt2 == 'test_unseen': # For diagnosis (writing predictions to a file for each class)
            #     print(predicted_label, labels)
            #     for t, p in zip(labels, predicted_label):
            #         if t.item() not in analysis:
            #             analysis[t.item()] = set()
            #         analysis[t.item()].add(p.item())

            # # For diagnosis (logits vs attributes tsne plot)
            # plot_tsne_F_vs_attributes(F_pred=f_pred, attr_matrix=att, y_true=labels,
            #                           n_seen_classes=seen_classes, n_unseen_classes=novel_classes,
            #                           iteration=iteration, save=True, file_path="analysis\\tsne")
            

            pred_label = np.append(pred_label, predicted_label)
            true_label = np.append(true_label, labels.cpu().numpy())
    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0

    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]

    # with open('analysis.txt', 'w') as f: # For diagnosis (writing predictions to a file for each class)
    #     for key, values in analysis.items():
    #         f.write(f"{key}: {', '.join(map(str, values))}\n\n") 

    return acc


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators.
    """
    # Seed Python's built-in random module
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)