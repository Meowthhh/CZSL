import json
import os
from omegaconf import OmegaConf
import statistics
import wandb
from args import get_args
from utils import compute_acc, seed_everything
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from data import DATA_LOADER as dataloader
from dk_classifier3 import DKClassifier3
from dk_classifier2 import DKClassifier2
from tqdm import tqdm
from losses import compute_cosine_loss_with_negatives, MetricLearningModel

cwd = os.path.dirname(os.getcwd())

seen_acc_history = []
unseen_acc_history = []
harmonic_mean_history = []
best_seen_acc_history = []
best_unseen_acc_history = []
best_harmonic_mean_history = []


class Train_Dataloader:
    def __init__(self, train_feat_seen, train_label_seen):
        self.data = {'train_': train_feat_seen, 'train_label': train_label_seen}

    def get_loader(self, opt='train_', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)
        return data_loader


class Test_Dataloader:
    def __init__(self, test_attr, test_seen_f, test_seen_l, test_unseen_f, test_unseen_l):

        self.data = {'test_seen': test_seen_f, 'test_seenlabel': test_seen_l,
                     'whole_attributes': test_attr,
                     'test_unseen': test_unseen_f, 'test_unseenlabel': test_unseen_l,
                     'seen_label': np.unique(test_seen_l),
                     'unseen_label': np.unique(test_unseen_l)}

    def get_loader(self, opt='test_seen', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt + 'label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size)
        return data_loader


def train(task_id, Neighbors, model, data, seen_classes, novel_classes, feature_size=2048, dlr=0.005, batch_size=64,
           epochs=50, lambda1=10.0, alpha=1.0, avg_feature=None , all_classes=None, num_tasks=None):
    
    # Prepare attribute
    total_seen_classes = seen_classes * task_id
    cur_train_feature_seen, cur_train_label_seen, cur_train_att_seen = data.task_train_data(task_id, seen_classes, all_classes,
                                                                                 novel_classes, num_tasks, Neighbors)
    print(f'Current seen label {sorted(list(set([i.item() for i in cur_train_label_seen])))},')
    
    train_label_seen_tensor = torch.tensor(cur_train_label_seen.clone().detach() - total_seen_classes + seen_classes).squeeze()
    accumulated_features = torch.zeros((seen_classes, feature_size))
    class_count = torch.zeros(seen_classes) 
    accumulated_features.index_add_(0, train_label_seen_tensor, cur_train_feature_seen) # Accumulate features for each class
    class_count.index_add_(0, train_label_seen_tensor, torch.ones_like(train_label_seen_tensor, dtype=torch.float)) # Count occurrences of each class
    class_count[class_count == 0] = 1 # Avoid division by zero for classes with no samples
    cur_avg_feature = accumulated_features / class_count.unsqueeze(1) # Shape: (10, 2048)

    if task_id == 1:
        avg_feature = cur_avg_feature.cuda()
    else:
        avg_feature = torch.cat([avg_feature, cur_avg_feature.cuda()], dim=0)

    # No. of examples and their labels for the current task
    train_feat_seen = cur_train_feature_seen
    train_label_seen = cur_train_label_seen
    
    train_loader = Train_Dataloader(train_feat_seen, train_label_seen)
    train_dataloader = train_loader.get_loader('train_', batch_size)

    # Prepare seen attributes for the current task
    all_attribute = data.attribute_mapping(seen_classes, novel_classes, task_id).cuda() # Shape: (50, 85)
    seen_attr = all_attribute[0:total_seen_classes, :] # Shape: (10, 85) increases as new classes arrive in new tasks

    optimizer = optim.Adam(model.parameters(), lr=dlr, weight_decay=0.00001)
    entropy_loss = nn.CrossEntropyLoss()
    metric_loss = MetricLearningModel(attr_dim=seen_attr.shape[1]).cuda()
    metric_optimizer = optim.Adam(metric_loss.parameters(), lr=dlr)

    epoch_seen_accuracy_history = []
    epoch_unseen_accuracy_history = []
    epoch_harmonic_accuracy_history = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        losses = {
            "d_loss": [],
        }

        # Mini batch loops
        for feature, label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            if feature.shape[0] == 1:
                continue

            feature = feature.cuda() # Shape: (128, 2048) assuming bs=128 
            label = label.cuda() # Shape: (128, 1) assuming bs=128
            optimizer.zero_grad()
            metric_optimizer.zero_grad()

            # Repeat and normalize seen attributes for faster processing 
            batch_repeated_seen_attr = seen_attr.unsqueeze(0).repeat([feature.size(0), 1, 1]) # Shape: (128, 10, 85)
            batch_repeated_seen_attr_norm = F.normalize(batch_repeated_seen_attr, p=2, dim=-1, eps=1e-12)
            
            logits = model(feature) # Shape: (128, 85)
            logits_norm = F.normalize(logits, p=2, dim=-1, eps=1e-12)
            class_repeated_logits_norm = logits_norm.unsqueeze(1).repeat([1, total_seen_classes, 1]) # Shape: (128, 10, 85)
            real_cosine_sim = lambda1 * F.cosine_similarity(batch_repeated_seen_attr_norm, class_repeated_logits_norm, dim=-1) # Shape: (128, 10)
            
            label = label.type(torch.LongTensor)  
            label = label.to('cuda') 
            
            # Cross-Entropy Loss for classification
            # classification_loss = entropy_loss(real_cosine_sim, label.squeeze()) # Squeezed label shape: (128)

            # Cosine contrastive loss with negatives
            # attr_cosine_loss_ = compute_cosine_loss_with_negatives(logits_norm, F.normalize(seen_attr, p=2, dim=-1, eps=1e-12), label) 

            # Metric learning loss
            attr_cosine_loss = metric_loss(logits_norm, F.normalize(seen_attr, p=2, dim=-1, eps=1e-12), label.squeeze()) 

            
            # Weighted sum of losses if multiple 
            alpha = 1
            # d_loss = alpha * attr_cosine_loss + (1 - alpha) * classification_loss
            d_loss = alpha * attr_cosine_loss
            
            d_loss.backward(retain_graph=True)
            losses["d_loss"].append(d_loss.item())
            optimizer.step()
            metric_optimizer.step()
                
        if epoch == epochs - 1:
            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(
                task_id, seen_classes, all_classes, novel_classes, num_tasks)
            print(f'current unseen label {sorted(list(set([i.item() for i in test_unseen_l])))}')
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, task_id).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_unseen_f,
                                              test_unseen_l)
            D_seen_acc = compute_acc(model, test_dataloader, seen_classes, novel_classes, task_id,
                                       batch_size=batch_size, opt1='gzsl', opt2='test_seen', iteration=0) # iteration=0 for seen (only for diagnosis)
            D_unseen_acc = compute_acc(model, test_dataloader, seen_classes, novel_classes, task_id,
                                         batch_size=batch_size, opt1='gzsl', opt2='test_unseen', iteration=1000) # iteration=1000 for unseen (only for diagnosis)
            if D_unseen_acc == 0 or D_seen_acc == 0:
                D_harmonic_mean = 0
            else:
                D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)

            epoch_seen_accuracy_history.append(D_seen_acc)
            if task_id != opt.num_tasks:
                epoch_unseen_accuracy_history.append(D_unseen_acc)
                epoch_harmonic_accuracy_history.append(D_harmonic_mean)
            print(
                f'\n\nBest accuracy at task {task_id} at epoch {epoch}: unseen : {D_unseen_acc:.4f}, seen : {D_seen_acc:.4f}, H : {D_harmonic_mean:.4f}\n\n')

        loss_metrics = {
            "epoch": epoch,
            "task_no": task_id,
        }
        for key, value in losses.items():
            if len(value) > 0:
                loss_metrics[f"losses/{key}"] = statistics.mean(value)
        with open(f"logs/{opt.run_name}/losses.json", "a+") as f:
            json.dump(loss_metrics, f)
            f.write("\n")

    seen_acc_history.append(D_seen_acc)
    unseen_acc_history.append(D_unseen_acc)
    harmonic_mean_history.append(D_harmonic_mean)

    final_model_acc_history = []
    forgetting_measure = 0
    if task_id == opt.num_tasks:
        for t in range(1, opt.num_tasks + 1):
            test_seen_f, test_seen_l, test_seen_a, test_unseen_f, test_unseen_l, test_unseen_a = data.task_test_data_(t,
                                                                                                                      seen_classes,
                                                                                                                      all_classes,
                                                                                                                      novel_classes,
                                                                                                                      num_tasks)
            att_per_task_ = data.attribute_mapping(seen_classes, novel_classes, t).cuda()
            test_dataloader = Test_Dataloader(att_per_task_, test_seen_f, test_seen_l, test_unseen_f,
                                              test_unseen_l)
            final_model_acc = compute_acc(model, test_dataloader, seen_classes, novel_classes, t,
                                            batch_size=batch_size, opt1='gzsl', opt2='test_seen')
            final_model_acc_history.append(final_model_acc)
        final_model_acc_difference = np.array(final_model_acc_history) - np.array(seen_acc_history)
        forgetting_measure = np.mean(final_model_acc_difference[:-1])

    checkpoint = {
        "task_no": task_id,
        "model": model.state_dict(),
        "optimizer_D": optimizer.state_dict(),
    }
    
    try:
        torch.save(checkpoint, f'checkpoints/{opt.run_name}/checkpoint_task_{task_id}.pth')
        with open(f"logs/{opt.run_name}/metrics.json", "w") as f:
            json.dump({
                "seen_acc_history": seen_acc_history,
                "unseen_acc_history": unseen_acc_history,
                "harmonic_mean_history": harmonic_mean_history,
                "mean_seen_acc": statistics.mean(seen_acc_history),
                "mean_unseen_acc": statistics.mean(unseen_acc_history),
                "mean_harmonic_mean": statistics.mean(harmonic_mean_history),
                "forgetting_measure": forgetting_measure,
                "final_model_acc_history": final_model_acc_history,
            }, f)
    except:
        print("Saving failed")

    return avg_feature



def main(opt):
    data = dataloader(opt)
    model = DKClassifier3().cuda()
    # model = DKClassifier2().cuda()
    avg_feature = None

    # Hyper-parameter validation with quarter of total tasks
    if opt.validation:
        iter_task = opt.num_tasks // 4
    else:
        iter_task = opt.num_tasks

    for task_id in range(1, iter_task):
        avg_feature = train(task_id, opt.Neighbors, model,
                            data, opt.seen_classes, opt.novel_classes,
                            feature_size=opt.feature_size, dlr=opt.d_lr,
                            batch_size=opt.batch_size, epochs=opt.epochs,
                            lambda1=opt.t, alpha=opt.alpha,
                            avg_feature=avg_feature, all_classes=opt.all_classes,
                            num_tasks=opt.num_tasks)
        
        exit() # Remove this line to run all tasks
          



if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    cwd = os.getcwd()
    opt = get_args(cwd)

    print("Script has started")
    print()
    seed_everything(opt.seed)

    # log
    checkpoint_dir = os.path.join(cwd, 'checkpoints', opt.run_name)
    logs_dir = os.path.join(cwd, 'logs', opt.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    with open(f"logs/{opt.run_name}/config.json", "w") as f:
        json.dump(OmegaConf.to_yaml(opt), f)
    if opt.wandb_log:
        wandb.init(project='Random_Walk_CGZSL', name=opt.run_name, config=vars(opt))
                
    print(opt)
    main(opt)

        