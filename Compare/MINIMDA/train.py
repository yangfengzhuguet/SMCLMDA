from minimda import MINIMDA
from torch import optim,nn
from tqdm import trange
from utils import k_matrix
import dgl
import networkx as nx
import copy
import torch
import numpy as np
import torch as th
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import matplotlib.pyplot as plt

def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc,aupr

def train(data,args):
    print('-----------------------train--------------------------')
    model = MINIMDA(args)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    cross_entropy = nn.BCELoss()
    epochs = trange(args.epochs, desc='train')
    miRNA=data['ms']
    disease=data['ds']
    for e in epochs:
        model.train()
        optimizer.zero_grad()

        mm_matrix = k_matrix(data['ms'], args.neighbor)
        dd_matrix = k_matrix(data['ds'], args.neighbor)
        mm_nx=nx.from_numpy_array(mm_matrix)
        dd_nx=nx.from_numpy_array(dd_matrix)
        mm_graph = dgl.from_networkx(mm_nx)
        dd_graph = dgl.from_networkx(dd_nx)

        md_copy = copy.deepcopy(data['md'])
        md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
        md_graph = dgl.graph(
            (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
            num_nodes=args.miRNA_number + args.disease_number)

        miRNA_th=th.Tensor(miRNA)
        disease_th=th.Tensor(disease)
        train_samples_th = th.Tensor(data['train_samples']).float()

        train_score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['train_samples'])

        train_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2])
        auc_, aupr = show_auc(train_score, train_samples_th[:,2], 'train')
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(e, train_loss, auc_, aupr))
        train_loss.backward()
        # print(train_loss)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        print('--------------------test-------------------------')
        mm_matrix = k_matrix(data['ms'], args.neighbor)
        dd_matrix = k_matrix(data['ds'], args.neighbor)
        mm_nx = nx.from_numpy_array(mm_matrix)
        dd_nx = nx.from_numpy_array(dd_matrix)
        mm_graph = dgl.from_networkx(mm_nx)
        dd_graph = dgl.from_networkx(dd_nx)

        md_copy = copy.deepcopy(data['md'])
        md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
        md_graph = dgl.graph(
            (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
            num_nodes=args.miRNA_number + args.disease_number)

        miRNA_th = th.Tensor(miRNA)
        disease_th = th.Tensor(disease)
        test_samples_th = th.Tensor(data['test_samples']).float()
        test_score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['test_samples'])
        test_loss = cross_entropy(th.flatten(test_score), test_samples_th[:, 2])
        auc_, aupr_ = show_auc(test_score, test_samples_th[:, 2], 'test')
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-------------'.format(test_loss, auc_, aupr_))
        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        y_true = np.array(test_samples_th[:, 2].detach().cpu())
        y_true = np.where(y_true == 1, True, False)
        y_scores = np.array(test_score.detach().cpu())

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_ = auc(fpr, tpr)
        roc_auc_ = round(roc_auc_, 3)

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        aupr = round(aupr, 3)

        best_threshold = 0.0
        best_f1 = 0.0
        best_metrics = {}

        sensitivities = []
        specificities = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            accuracy = (y_pred == y_true).mean()
            mcc = matthews_corrcoef(y_true, y_pred)
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            specificity = tn / (tn + fp)

            sensitivities.append(recall)
            specificities.append(specificity)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "specificity": specificity
                }

        plt.figure(1)
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_}")
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        plt.title('ROC Curve', font1)
        plt.legend(prop=font1)

        plt.figure(2)
        plt.plot(recall, precision, label=f"AUPR={aupr}", color='purple')
        plt.xlabel('Recall', font1)
        plt.ylabel('Precision', font1)
        plt.title('Precision-Recall Curve', font1)
        plt.legend(prop=font1)

        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        plt.text(0.6, 0.2, best_metrics_str, bbox=dict(facecolor='white', alpha=0.5), fontsize=9)
        # plt.savefig("./Result_causal_for_ROC_10_fold_mean.tiff", dpi=600)
        # plt.show()
        # plt.close()

        print(f"Best Threshold: {best_threshold}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"Specificity: {best_metrics['specificity']:.4f}")
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"AUROC: {roc_auc_:.4f}")
        print(f"AUPR: {aupr:.4f}")
        model.train()
    return best_metrics['accuracy'], best_metrics['precision'], best_metrics['recall'], best_metrics['specificity'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_
