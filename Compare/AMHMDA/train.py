import torch
import random
import torch.nn
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import pandas as pd
from model import AMHMDA, EmbeddingM, EmbeddingD,MDI##*
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):##*
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc,aupr

def train_test(simData, train_data, param,state):
    epo_metric = []
    valid_metric = []
    train_data['train'] = train_data['train']
    train_edges = train_data['train'][:,:2]
    train_edges = torch.from_numpy(train_edges).int().to(device)
    train_labels = train_data['train'][:,2]
    train_labels = torch.from_numpy(train_labels).float().to(device)
    test_edges = train_data['test'][:,:2]
    test_edges = torch.from_numpy(test_edges).to(device)
    test_labels = train_data['test'][:,2]
    test_labels = torch.from_numpy(test_labels).float().to(device)

    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yinguo.csv',sep=',',header=None)
    # yanzheng = yanzheng.values
    # test_edges = yanzheng[:,:2]
    # test_edges = torch.from_numpy(test_edges).int().to(device)
    # test_labels = yanzheng[:,2]
    # test_labels = torch.from_numpy(test_labels).float().to(device)
    #
    #
    #
    # mda = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/hmdd2_MDA_.csv',sep=',',header=None)
    # mda = mda.values
    # mda = torch.Tensor(mda).to(device)

    torch.manual_seed(42)
    model = AMHMDA(EmbeddingM(param), EmbeddingD(param), MDI(param))##*
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)  ###
    print("------------------train---------------------")
    for e in range(param.epoch):
        model.train()
        # train_score = model(simData, train_edges)##*
        train_score = model(simData, train_edges)##*
        train_loss = torch.nn.BCELoss()
        loss = train_loss(train_score, train_labels)
        auc_, aupr = show_auc(train_score, train_labels, 'train')
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(e, loss, auc_, aupr))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        print('---------------------test-----------------------')
        test_score = model(simData, test_edges)
        test_loss = torch.nn.BCELoss()
        test_loss = test_loss(test_score, test_labels)
        auc_, aupr_ = show_auc(test_score, test_labels, 'test')
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(test_loss, auc_, aupr_))
        # 设置图像分辨率和字体
        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        # 示例数据
        y_true = np.array(test_labels.detach().cpu())  # 因果性标签
        y_true = np.where(y_true == 1, True, False)  # 将1设为True，2设为False
        y_scores = np.array(test_score.detach().cpu())  # 预测分数
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_ = auc(fpr, tpr)
        roc_auc_ = round(roc_auc_, 3)

        # 计算Precision-Recall曲线
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        aupr = round(aupr, 3)
        # 计算不同阈值下的性能指标，并找到最佳阈值
        best_threshold = 0.0
        best_f1 = 0.0
        best_metrics = {}

        # 保存每个阈值下的灵敏度和特异性
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

        # 绘制ROC曲线
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_}")
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        plt.title('ROC Curve', font1)
        plt.legend(prop=font1)

        # 绘制Precision-Recall曲线
        plt.figure(2)
        plt.plot(recall, precision, label=f"AUPR={aupr}", color='purple')
        plt.xlabel('Recall', font1)
        plt.ylabel('Precision', font1)
        plt.title('Precision-Recall Curve', font1)
        plt.legend(prop=font1)

        # 显示最佳阈值下的性能指标
        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        plt.text(0.6, 0.2, best_metrics_str, bbox=dict(facecolor='white', alpha=0.5), fontsize=9)

        # 显示并保存图像
        # plt.savefig("./Result_causal_for_ROC_10_fold_mean.tiff", dpi=600)
        # plt.show()
        # plt.close()
        # 打印最佳阈值和性能指标
        print(f"Best Threshold: {best_threshold}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")  # 灵敏度
        print(f"Specificity: {best_metrics['specificity']:.4f}")  # 特异性
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"AUROC: {roc_auc_:.4f}")
        print(f"AUPR: {aupr:.4f}")
        model.train()
    return best_metrics['accuracy'], best_metrics['precision'], best_metrics['recall'], best_metrics['specificity'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_

