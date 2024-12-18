import numpy as np
import torch.nn
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import torch.nn.functional as F
from numpy import interp
import tqdm
import time


def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    # if flag == 'test':
    #     # Plot the roc curve and save it
    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc='lower right')
    #
    #     # Save the curve to the specified directory
    #     # output_dir = 'result/HMDD v3-2-wuguangdui/10-fold_'
    #     output_dir = 'result/HMDD V4/fencneg'
    #     os.makedirs(output_dir, exist_ok=True)
    #     # plt.savefig(os.path.join(output_dir, '10-fold_9_test_auc.png'))
    #     plt.savefig(os.path.join(output_dir, 'fenceng.png'))
    return auroc,aupr

def train_SMCLMDA(arges, model, sim_set, meta_set, emb, pos_miRNA, pos_disease,optimizer, pair_pos_neg_fengceng, device):
    model = model.to(device)
    train_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,0], dtype=torch.long).to(device) 
    train_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,1],dtype=torch.long).to(device) 
    train_label_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,2]).to(device).float() 

    test_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:, 0],dtype=torch.long).to(device)  
    test_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,1],dtype=torch.long).to(device)  
    test_label_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,2]).to(device).float()  

    # # Loading the causal dataset yanzheng
    # train_miRNA_index_fc = pair_pos_neg_fengceng['train'][0].values
    # train_miRNA_index_fc = torch.tensor(train_miRNA_index_fc, dtype=torch.long).to(device)
    # train_disease_index_fc = pair_pos_neg_fengceng['train'][1].values
    # train_disease_index_fc = torch.tensor(train_disease_index_fc, dtype=torch.long).to(device)
    # train_label_fc = pair_pos_neg_fengceng['train'][2].values
    # train_label_fc = torch.tensor(train_label_fc).to(device).float()
    # #
    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yinguo.csv',sep=',',header=None)
    # test_miRNA_index_fc = yanzheng[0].values
    # test_miRNA_index_fc = torch.tensor(test_miRNA_index_fc, dtype=torch.long).to(device)
    # test_disease_index_fc = yanzheng[1].values
    # test_disease_index_fc = torch.tensor(test_disease_index_fc, dtype=torch.long).to(device)
    # test_label_fc = yanzheng[2].values
    # test_label_fc = torch.tensor(test_label_fc).to(device).float()
    # #
    # mda = pd.read_csv('data/mir2disease+lunwen/-1-0-+1/mirBase_lunwen_101_MDA_.csv',sep=',',header=None)
    # mda = mda.values
    # mda_ = torch.Tensor(mda).to(device)

    loss_min = float('inf')
    best_auc = 0
    best_aupr = 0
    tsne = TSNE(n_components=2, random_state=42)
    m = 1
    n = 1
    pca = PCA(n_components=2)
    print('######################### Start training #############################')
    for epoch_ in tqdm.tqdm(range(arges.epoch), desc='Training Epochs'):
        time_start = time.time()
        model.train()
        train_score, loss_cl_train = model(sim_set, meta_set, emb, pos_miRNA, pos_disease, train_miRNA_index_fc, train_disease_index_fc)
        loss1 = torch.nn.BCELoss()
        loss_train = loss1(train_score, train_label_fc)
        # loss_train = F.binary_cross_entropy(train_score, mda_)
        loss = arges.lambda_2 * loss_cl_train + (1 - arges.lambda_2) * loss_train
        auc_, aupr = show_auc(train_score, train_label_fc, 'train')
        if loss_train < loss_min:
            loss_min = loss_train
        if auc_ > best_auc:
            best_auc = auc_
        if aupr > best_aupr:
            best_aupr = aupr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end = time.time()
        time_epoch = time_end - time_start
        # if epoch_ == 10:
        #     pred_epoch_10 = train_score
        #     pred_epoch_10 = pred_epoch_10.detach().cpu().numpy()
        #     pred_epoch_10 = pd.DataFrame(pred_epoch_10)
        #     pred_epoch_10.to_csv('pre_epoch_10.csv',header=False,index=False)
        # elif epoch_ == 50:
        #     pred_epoch_50 = train_score
        #     pred_epoch_50 = pred_epoch_50.detach().cpu().numpy()
        #     pred_epoch_50 = pd.DataFrame(pred_epoch_50)
        #     pred_epoch_50.to_csv('pre_epoch_50.csv', header=False, index=False)
        # elif epoch_ ==100:
        #     pred_epoch_100 = train_score
        #     pred_epoch_100 = pred_epoch_100.detach().cpu().numpy()
        #     pred_epoch_100 = pd.DataFrame(pred_epoch_100)
        #     pred_epoch_100.to_csv('pre_epoch_100.csv', header=False, index=False)
        # elif epoch_ ==200:
        #     pred_epoch_200 = train_score
        #     pred_epoch_200 = pred_epoch_200.detach().cpu().numpy()
        #     pred_epoch_200 = pd.DataFrame(pred_epoch_200)
        #     pred_epoch_200.to_csv('pre_epoch_200.csv', header=False, index=False)
        # elif epoch_ ==250:
        #     pred_epoch_250 = train_score
        #     pred_epoch_250 = pred_epoch_250.detach().cpu().numpy()
        #     pred_epoch_250 = pd.DataFrame(pred_epoch_250)
        #     pred_epoch_250.to_csv('pre_epoch_250.csv', header=False, index=False)

        # if epoch_ in [0, 49, 99, 299]:
        #     mi_emb = mi_emb_[train_miRNA_index_fc]
        #     di_emb = di_emb_[train_disease_index_fc]
        #     mi_emb = mi_emb.detach().cpu().numpy()
        #     di_emb = di_emb.detach().cpu().numpy()
        #     all_features = np.concatenate([mi_emb, di_emb], axis=1)
        #     # t-SNE dimensionality reduction
        #     low_dim_features = tsne.fit_transform(all_features)
        #     # # umap dimensionality reduction
        #     # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        #     # low_dim_features = reducer.fit_transform(all_features)
        #     # Plotting t-SNE projections
        #     plt.figure(figsize=(8, 6))
        #     plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=train_label_fc.detach().cpu().numpy(), cmap='coolwarm', s=5)
        #     plt.title(f'epoch: {epoch_+1}')
        #     plt.colorbar(label='Label (0: Negative, 1: Positive)')
        #     plt.show()
        #     plt.savefig(os.path.join('result/HMDD V4', f'{epoch_}.png'))
        print('-------Time when the epoch runs：{} seconds ----------'.format(time_epoch))
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(epoch_, loss, auc_, aupr))
    # label = train_label_fc.detach().cpu().numpy().tolist()
    # pred_epoch_10 = pred_epoch_10.detach().cpu().numpy().tolist()
    # pred_epoch_50 = pred_epoch_50.detach().cpu().numpy().tolist()
    # pred_epoch_100 = pred_epoch_100.detach().cpu().numpy().tolist()
    # pred_epoch_200 = pred_epoch_200.detach().cpu().numpy().tolist()
    # pred_epoch_250 = pred_epoch_250.detach().cpu().numpy().tolist()
    # data = {
    #     'Label': label * 5,
    #     'Score': pred_epoch_10 + pred_epoch_50 + pred_epoch_100 + pred_epoch_200 + pred_epoch_250,
    #     'Epoch': ['Epoch 10'] * len(label) + ['Epoch 50'] * len(label) + ['Epoch 100'] * len(label)+ ['Epoch 200'] * len(label)+ ['Epoch 400'] * len(label)
    # }
    #
    # df = pd.DataFrame(data)
    #
    #
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='Epoch', y='Score', hue='Label', data=df, flierprops=dict(marker='o', markersize=5, linestyle='none', alpha=0.5))
    # plt.title('Prediction Score Distribution Across Epochs')
    # plt.ylabel('Prediction Score')
    # plt.xlabel('Epoch')
    # plt.savefig(os.path.join('result/HMDD V4', f'train_10_balance.png'))
    # plt.show()
    print('The loss_min:{}, best auc{}, best aupr{}'.format(loss_min, best_auc, best_aupr))
    print('######################### Model training is complete, and we're moving on to testing. #############################')
    model.eval()
    with torch.no_grad():
        test_score, loss_cl_test = model(sim_set, meta_set, emb, pos_miRNA, pos_disease, test_miRNA_index_fc, test_disease_index_fc)
        loss2 = torch.nn.BCELoss()
        loss_test = loss2(test_score, test_label_fc)
        loss_ = arges.lambda_2 * loss_cl_test + (1 - arges.lambda_2) * loss_test
        auc_, aupr_ = show_auc(test_score, test_label_fc, 'test')
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(loss_, auc_, aupr_))
        data = test_score.cpu().numpy()
        df = pd.DataFrame(data)
        df.to_csv('score.csv', header=False, index=False)
        pred1 = test_score.detach().cpu().numpy()
        label1 = test_label_fc.detach().cpu().numpy()

        # data1 = {
        #     'Label1': label1,
        #     'Score1': pred1,
        #     'Epoch1': ['test'] * len(label1)
        # }
        #
        # df1 = pd.DataFrame(data1)
        #
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(x='Epoch1', y='Score1', hue='Label1', data=df1,
        #             flierprops=dict(marker='o', markersize=5, linestyle='none', alpha=0.5))
        # plt.title('Prediction Score Distribution Across Epochs')
        # plt.ylabel('Prediction Score')
        # plt.xlabel('Epoch')
        # # plt.savefig(os.path.join('result/HMDD V4', f'test_10_imbalance_xianxiang.png'))
        # plt.show()


        # # Suppose y_true is the actual label and y_pred is the model's predicted label
        # pred1 = np.where(pred1 >= 0.4207,1,0)
        # conf_matrix = confusion_matrix(label1, pred1)
        #
        # # Visualization with Seaborn
        # plt.figure(figsize=(6, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
        # plt.title("Confusion Matrix", fontsize=24)
        # plt.ylabel('Actual Label', fontsize=22)
        # plt.xlabel('Predicted Label', fontsize=22)
        # plt.xticks(fontsize=16)  
        # plt.yticks(fontsize=16)  
        # plt.savefig(os.path.join('result/HMDD V4', f'test_10_balance_matrix_.png'))
        # plt.show()

        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        y_true = np.array(test_label_fc.detach().cpu())  
        y_true = np.where(y_true == 1, True, False)  
        y_scores = np.array(test_score.detach().cpu())  
        # Plotting the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_ = auc(fpr, tpr)
        roc_auc_ = round(roc_auc_, 3)
        # np.save('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_fp10.npy', fpr)
        # np.save('result/HMDD V4/10-fold-balance/fp_and_tp_nocl/hmdd4_balance_tp10.npy', tpr)


        # Plotting Precision-Recall Curve Lines
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        aupr = round(aupr, 3)
        # Calculate the performance metrics at different thresholds and find the optimal thresholds
        best_threshold = 0.0
        best_f1 = 0.0
        best_metrics = {}

        # Sensitivity and specificity at each threshold are preserved
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

        # Plotting the ROC curve
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_}")
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        plt.title('ROC Curve', font1)
        plt.legend(prop=font1)

        # Plotting the Precision-Recal curvel
        plt.figure(2)
        plt.plot(recall, precision, label=f"AUPR={aupr}", color='purple')
        plt.xlabel('Recall', font1)
        plt.ylabel('Precision', font1)
        plt.title('Precision-Recall Curve', font1)
        plt.legend(prop=font1)

        # Displays performance metrics at optimal thresholds
        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        plt.text(0.6, 0.2, best_metrics_str, bbox=dict(facecolor='white', alpha=0.5), fontsize=9)

        # Display and save images
        # plt.savefig("./Result_causal_for_ROC_10_fold_mean.tiff", dpi=600)
        # plt.show()
        # plt.close()
        # Printing Optimal Thresholds and Performance Metrics
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
