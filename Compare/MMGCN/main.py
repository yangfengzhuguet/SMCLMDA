from param import parameter_parser
from MMGCN import MMGCN
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import pickle
import torch
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc,aupr

def train(model, simData, train_data, optimizer, opt):
    train_data['train'] = train_data['train']
    train_edges = train_data['train'][:, :2]
    train_edges = torch.from_numpy(train_edges).int().to(device)
    train_labels = train_data['train'][:, 2]
    train_labels = torch.from_numpy(train_labels).float().to(device)
    test_edges = train_data['test'][:,:2]
    test_edges = torch.from_numpy(test_edges).to(device)
    test_labels = train_data['test'][:,2]
    test_labels = torch.from_numpy(test_labels).float().to(device)





    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yinguo.csv', sep=',', header=None)
    # yanzheng = yanzheng.values
    # test_edges = yanzheng[:, :2]
    # test_edges = torch.from_numpy(test_edges).int().to(device)
    # test_labels = yanzheng[:, 2]
    # test_labels = torch.from_numpy(test_labels).float().to(device)
    #
    # mda = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/hmdd2_MDA_.csv',sep=',',header=None)
    # mda = mda.values
    # mda = torch.Tensor(mda).to(device)
    print('------------trian-----------')
    for epoch in range(0, opt.epoch):
        model.train()
        train_score = model(simData, train_edges)
        loss = torch.nn.BCELoss()
        loss = loss(train_score, train_labels)
        auc_, aupr = show_auc(train_score, train_labels, 'train')
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(epoch, loss, auc_, aupr))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        print('------------test---------------')
        test_score = model(simData, test_edges)
        test_loss  =torch.nn.BCELoss()
        test_loss = test_loss(test_score, test_labels)
        auc_, aupr_ = show_auc(test_score, test_labels, 'test')
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(test_loss, auc_, aupr_))

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parameter_parser()
    set_seed(42)
    parent_dir = 'data/HMDD v_4/predata'
    parent_dir_ = 'data/HMDD v_4/fenceng'
    simData = load_pkl(parent_dir, 'sim_set.pkl')
    train_data = load_pkl(parent_dir_, 'pos_neg_pair_fengceng.pkl')
    model = MMGCN(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    is_tenfold = 'true'
    Acc = []
    Pre = []
    Rec = []
    Spe = []
    Mcc = []
    F1 = []
    AUROC = []
    AUPR = []
    if is_tenfold == 'no':
        for i in range(10):
            print(f'----------------------------{i+1}----------------------')
            train_data = load_pkl(parent_dir_, f'pos_neg_pair_10_{i + 1}.pkl')
            accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train(model, simData, train_data, optimizer, args)
            Acc.append(accuracy)
            Pre.append(precision)
            Rec.append(recall)
            Spe.append(specificity)
            Mcc.append(mcc)
            F1.append(f1)
            AUROC.append(auc_)
            AUPR.append(aupr_)
    else:
        train_data = load_pkl( parent_dir_, 'pos_neg_pair_fengceng.pkl')
        accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train(model, simData, train_data, optimizer,args)
        Acc.append(accuracy)
        Pre.append(precision)
        Rec.append(recall)
        Spe.append(specificity)
        Mcc.append(mcc)
        F1.append(f1)
        AUROC.append(auc_)
        AUPR.append(aupr_)
    print('-------------------------------------------------------------------')
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


if __name__ == "__main__":
    main()
