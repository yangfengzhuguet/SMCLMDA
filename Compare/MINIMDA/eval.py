import argparse
from utils import get_data,data_processing
from train import train
import numpy as np
import pandas as pd
import os
import torch
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def result(args):
    # data = get_data(args)
    # data_processing(data,args)
    data = dict()
    parent_dir = 'data/HMDD v_4/predata'
    parent_dir_ = 'data/HMDD v_4/fenceng'
    # parent_dir_1 = 'data/HMDD V3_2_res+yangzheng(yinguo)/10-fold-balance'
    simData = load_pkl(parent_dir, 'sim_set.pkl')
    train_data = load_pkl(parent_dir_, 'pos_neg_pair_fengceng.pkl')
    # train_data = load_pkl(parent_dir_, f'pos_neg_pair_10_{i+1}.pkl')
    ms = (simData['miRNA_mut']['mi_gua']+simData['miRNA_mut']['mi_cos']+simData['miRNA_mut']['mi_fun']) / 3
    ds = (simData['disease_mut']['di_gua']+simData['disease_mut']['di_cos']+simData['disease_mut']['di_sem']) / 3
    ms = ms.cpu().numpy()
    ds = ds.cpu().numpy()
    train_data['train'] = train_data['train']
    train_edges = train_data['train'][:, :3]
    train_md = train_edges[train_edges[:,2]!=0]
    train_md = train_md[:,:2]

    test_edges = train_data['test'][:, :3]
    test_md = test_edges[test_edges[:, 2] != 0]
    test_md = test_md[:,:2]

    # yanzheng = pd.read_csv('data/HMDD V3_2_res+yangzheng(yinguo)/test_yanzheng.csv', sep=',', header=None)
    # yanzheng = yanzheng.values
    # test_edges = yanzheng[:, :3]
    # test_md = test_edges[test_edges[:,2]!=0]
    # test_md = test_md[:,:2]


    train_samples = train_edges
    test_samples  =test_edges

    result = np.concatenate((train_md, test_md), axis=0)
    data['ms'] = ms
    data['ds'] = ds
    data['train_md'] = train_md
    data['test_md'] = test_md
    data['train_samples'] = train_samples
    data['test_samples'] = test_samples
    data['md'] = result

    args.miRNA_number = 728
    args.disease_number = 884
    accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train(data,args)
    return accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=23, metavar='N', help='number of epochs to train')
parser.add_argument('--fm', type=int, default=64, help='length of miRNA feature')
parser.add_argument('--fd', type=int, default=64, help='length of dataset feature')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--in_feats", type=int, default=64, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=64, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--gcn_bias", type=bool, default=True, help='gcn bias')
parser.add_argument("--gcn_batchnorm", type=bool, default=True, help='gcn batchnorm')
parser.add_argument("--gcn_activation", default='relu', help='gcn activation')
parser.add_argument("--num_layers", type=int, default=3, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 1], help='mlp layers')

parser.add_argument('--neighbor', type=int, default=30, help='neighbor')
parser.add_argument('--dataset', default='HMDD v2.0', help='dataset')
parser.add_argument('--save_score', default='True', help='save_score')
parser.add_argument('--negative_rate', type=float,default=1.0, help='negative_rate')


args = parser.parse_args()
args.dd2=True
# args.data_dir = 'data/' + args.dataset + '/'
# args.result_dir = 'result/' + args.dataset + '/'
# args.save_score = True if str(args.save_score) == 'True' else False
Acc = []
Pre = []
Rec = []
Spe = []
Mcc = []
F1 = []
AUROC = []
AUPR = []
for i in range(1):
    print(f'---------------------------------------{i+1}---------------------------------')
    accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = result(args)
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
