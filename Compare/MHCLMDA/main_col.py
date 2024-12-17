import numpy as np
import copy
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import random
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from model import Model
from hypergraph_utils import *
import matplotlib.pyplot as plt
import os
import seaborn as sns
from kl_loss import kl_loss
from function import create_resultmatrix
from utils import f1_score_binary,precision_binary,recall_binary
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc,aupr
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def contrastive_loss(h1, h2, tau=0.7):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

# MD = np.loadtxt("data/md_delete.txt")
# MM = np.loadtxt("data/mm_delete.txt")
# DD = np.loadtxt("data/dd_delete.txt")
# DG = np.loadtxt("data/dg_delete.txt")
# MG = np.loadtxt("data/mg_delete.txt")

def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def train(epochs, train_data):
    set_seed(42)
    train_data['train'] = train_data['train']
    train_edges = train_data['train'][:, :2]
    train_edges = torch.from_numpy(train_edges).int().to(device)
    train_labels = train_data['train'][:, 2]
    train_labels = torch.from_numpy(train_labels).float().to(device)
    test_edges = train_data['test'][:, :2]
    test_edges = torch.from_numpy(test_edges).to(device)
    test_labels = train_data['test'][:, 2]
    test_labels = torch.from_numpy(test_labels).float().to(device)

    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yanzheng.csv', sep=',', header=None)
    # yanzheng = yanzheng.values
    # test_edges = yanzheng[:, :2]
    # test_edges = torch.from_numpy(test_edges).int().to(device)
    # test_labels = yanzheng[:, 2]
    # test_labels = torch.from_numpy(test_labels).float().to(device)
    print('----------------------------train---------------------')
    for epoch in range(0, epochs):
        model.train()
        optimizer2.zero_grad()
        (reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, recover, result_h,
         mir_feature_1,
         mir_feature_2, mir_feature_3, dis_feature_1, dis_feature_2, dis_feature_3) = model(AT, A, HMG, HDG, mir_feat,
                                                                                            dis_feat, HMD, HDM, HMM,
                                                                                            HDD, train_edges)
        train_score = recover
        # MA = torch.masked_select(A, train_mask_tensor)
        # reG = torch.masked_select(reconstructionG.t(), train_mask_tensor)
        # reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)
        # reMMDD = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)
        # ret = torch.masked_select(result.t(), train_mask_tensor)
        # rec = torch.masked_select(recover.t(), train_mask_tensor)
        # re1 = torch.masked_select(reconstruction1.t(), train_mask_tensor)
        loss_c_m = contrastive_loss(mir_feature_2, mir_feature_1) + contrastive_loss(mir_feature_2, mir_feature_3)
        loss_c_d = contrastive_loss(dis_feature_2, dis_feature_1) + contrastive_loss(dis_feature_2, dis_feature_3)
        loss_c = loss_c_m + loss_c_d

        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)
        loss_v = loss_k + F.binary_cross_entropy_with_logits(reconstruction1, train_labels,
                                                             pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            result, train_labels, pos_weight=pos_weight)

        loss_r_h = F.binary_cross_entropy_with_logits(reconstructionG, train_labels,
                                                      pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            reconstructionMD, train_labels, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            reconstructionMMDD.t(), train_labels, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(recover,
                                                                                                              train_labels,
                                                                                                              pos_weight=pos_weight)
        loss = loss_r_h + 0.7 * loss_v + 0.3 * loss_c
        auc_, aupr = show_auc(train_score, train_labels, 'train')
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(epoch, loss, auc_, aupr))
        loss.backward()
        optimizer2.step()
    model.eval()
    with torch.no_grad():
        print('---------------------test-----------------------')
        (reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, recover, result_h,
         mir_feature_1, mir_feature_2, mir_feature_3, dis_feature_1, dis_feature_2, dis_feature_3) = model(AT_, A_, HMG,
                                                                                                           HDG,
                                                                                                           mir_feat,
                                                                                                           dis_feat,
                                                                                                           HMD, HDM,
                                                                                                           HMM, HDD,
                                                                                                           test_edges)
        test_score = recover

        loss_c_m = contrastive_loss(mir_feature_2, mir_feature_1) + contrastive_loss(mir_feature_2, mir_feature_3)
        loss_c_d = contrastive_loss(dis_feature_2, dis_feature_1) + contrastive_loss(dis_feature_2, dis_feature_3)
        loss_c = loss_c_m + loss_c_d

        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)
        loss_v = loss_k + F.binary_cross_entropy_with_logits(reconstruction1, test_labels,
                                                             pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            result, test_labels, pos_weight=pos_weight)

        loss_r_h = F.binary_cross_entropy_with_logits(reconstructionG, test_labels,
                                                      pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            reconstructionMD, test_labels, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            reconstructionMMDD.t(), test_labels, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(recover,
                                                                                                             test_labels,
                                                                                                             pos_weight=pos_weight)
        loss1 = loss_r_h + 0.7 * loss_v + 0.3 * loss_c
        auc_, aupr_ = show_auc(test_score, test_labels, 'train')
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-------------'.format(loss1, auc_, aupr_))

        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        y_true = np.array(test_labels.detach().cpu())
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




parent_dir = 'data/HMDD v_4/predata'
parent_dir_ = 'data/HMDD v_4/fenceng'
simData = load_pkl(parent_dir, 'sim_set.pkl')
Acc = []
Pre = []
Rec = []
Spe = []
Mcc = []
F1 = []
AUROC = []
AUPR = []
for i in range(1):
    print(f'----------------------------{i+1}-------------------------')
    train_data = load_pkl(parent_dir_, 'pos_neg_pair_fengceng.pkl')
    # train_data['train'] = train_data['train'].values

    MD = pd.read_csv('data/HMDD v_4/hmdd4_MDA_.csv', sep=',', header=None)
    MD = MD.values
    MM_gua = simData['miRNA_mut']['mi_gua']
    MM_gua = MM_gua.cpu().numpy()
    MM_cos = simData['miRNA_mut']['mi_cos']
    MM_cos = MM_cos.cpu().numpy()
    MM_fun = simData['miRNA_mut']['mi_fun']
    MM_fun = MM_fun.cpu().numpy()
    DD_gua = simData['disease_mut']['di_gua']
    DD_gua = DD_gua.cpu().numpy()
    DD_cos = simData['disease_mut']['di_cos']
    DD_cos = DD_cos.cpu().numpy()
    DD_sem = simData['disease_mut']['di_sem']
    DD_sem = DD_sem.cpu().numpy()

    train_index = train_data['train'][:, :3]
    train_MD = np.zeros((MD.shape[0], MD.shape[1]))
    for row in train_index:
        i, j, value = row
        train_MD[int(i), int(j)] = value

    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yanzheng.csv', sep=',', header=None)
    # yanzheng = yanzheng.values
    # test_index = yanzheng[:,:3]
    test_index = train_data['test'][:, :3]
    test_MD = np.zeros((MD.shape[0], MD.shape[1]))
    for row in test_index:
        i, j, value = row
        test_MD[int(i), int(j)] = value

    alpha = 0.7

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs to train.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    X = copy.deepcopy(MD)
    W = copy.deepcopy(X)
    Xn = copy.deepcopy(train_MD)
    Xn_ = copy.deepcopy(test_MD)

    HHMG = construct_H_with_KNN(MM_gua)
    HMG = generate_G_from_H(HHMG)
    HMG = HMG.double()
    HHDG = construct_H_with_KNN(DD_gua)
    HDG = generate_G_from_H(HHDG)
    HDG = HDG.double()
    mir_feat = torch.eye(MD.shape[0])
    dis_feat = torch.eye(MD.shape[1])
    parameters = [MD.shape[1], MD.shape[0]]

    A = copy.deepcopy(Xn)
    AT = A.T
    A_ = copy.deepcopy(Xn_)
    AT_ = A_.T

    HHMD = construct_H_with_KNN(A)
    HMD = generate_G_from_H(HHMD)
    HMD = HMD.double()

    HHDM = construct_H_with_KNN(AT)
    HDM = generate_G_from_H(HHDM)
    HDM = HDM.double()

    HHMM = construct_H_with_KNN(MM_fun)
    HMM = generate_G_from_H(HHMM)
    HMM = HMM.double()

    HHDD = construct_H_with_KNN(DD_sem)
    HDD = generate_G_from_H(HHDD)
    HDD = HDD.double()

    model = Model()
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    A = torch.from_numpy(A)
    AT = torch.from_numpy(AT)
    A_ = torch.from_numpy(A_)
    AT_ = torch.from_numpy(AT_)
    MD = torch.tensor(MD)
    MDT = MD.T
    MDT = torch.tensor(MDT)
    XX = copy.deepcopy(Xn)
    XXA = A
    pos_weight = float(XX.shape[0] * XX.shape[1] - XX.sum()) / XX.sum()
    pos_weight = np.array(pos_weight)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[1] / float((A.shape[0] * A.shape[1] - A.sum()) * 2)

    loss_kl = kl_loss(435, 757)
    mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)

    XX = torch.from_numpy(XX)
    if args.cuda:
        model.cuda()

        XX = XX.cuda()
        MD = MD.cuda()
        MDT = MDT.cuda()

        A = A.cuda()
        AT = AT.cuda()
        A_ = A_.cuda()
        AT_ = AT_.cuda()

        HMG = HMG.cuda()
        HDG = HDG.cuda()

        HMD = HMD.cuda()
        HDM = HDM.cuda()

        HMM = HMM.cuda()
        HDD = HDD.cuda()
        mir_feat = mir_feat.cuda()
        dis_feat = dis_feat.cuda()

    accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train(7, train_data)
    Acc.append(accuracy)
    Pre.append(precision)
    Rec.append(recall)
    Spe.append(specificity)
    Mcc.append(mcc)
    F1.append(f1)
    AUROC.append(auc_)
    AUPR.append(aupr_)
print('-----------------------------------------------------------------------------')
for acc in Acc:
    print(acc)
print(f'avg ACC:{sum(Acc) / len(Acc)}')
for pre in Pre:
    print(pre)
print(f'avg Pre:{sum(Pre) / len(Pre)}')
for rec in Rec:
    print(rec)
print(f'avg Rec:{sum(Rec) / len(Rec)}')
for spe in Spe:
    print(spe)
print(f'avg Spe:{sum(Spe) / len(Spe)}')
for mcc in Mcc:
    print(mcc)
print(f'avg Mcc:{sum(Mcc) / len(Mcc)}')
for f1 in F1:
    print(f1)
print(f'avg F1:{sum(F1) / len(F1)}')
for auc in AUROC:
    print(auc)
print(f'avg AUROC:{sum(AUROC) / len(AUROC)}')
for aupr in AUPR:
    print(aupr)
print(f'avg AUPR:{sum(AUPR) / len(AUPR)}')



