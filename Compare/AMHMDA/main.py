import torch
import random
import numpy as np
from datapro import Simdata_pro,loading_data
import os
import pickle
from train import train_test
# import warnings
#
# warnings.filterwarnings("ignore")


class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch =12
        self.gcn_layers = 2
        self.view = 3
        self.fm = 128
        self.fd = 128
        self.inSize = 128
        self.outSize = 128
        self.nodeNum = 64
        self.hdnDropout = 0.5
        self.fcDropout = 0.5
        self.maskMDI = False
        self.device = torch.device('cuda')

def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data



def main():
    param = Config()
    parent_dir = 'data/HMDD V3_2_res+yangzheng(yinguo)/predata'
    parent_dir_ = 'data/HMDD V3_2_res+yangzheng(yinguo)/10-fold'
    simData = load_pkl(parent_dir, 'sim_set.pkl')
    Acc = []
    Pre = []
    Rec = []
    Spe = []
    Mcc = []
    F1 = []
    AUROC = []
    AUPR = []
    is_tenfold = 'true'
    if is_tenfold == 'true':
        for i in range(10):
            print(f'-------------------the{i + 1}-------------------------------')
            train_data = load_pkl(parent_dir_, f'pos_neg_pair_10_{i+1}.pkl')
            accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train_test(simData, train_data, param,state='valid')
            Acc.append(accuracy)
            Pre.append(precision)
            Rec.append(recall)
            Spe.append(specificity)
            Mcc.append(mcc)
            F1.append(f1)
            AUROC.append(auc_)
            AUPR.append(aupr_)
    else:
        train_data = load_pkl(parent_dir_, 'pos_neg_pair_fengceng.pkl')
        accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train_test(simData, train_data, param,state='valid')
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
