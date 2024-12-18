import torch
import pandas as pd
import os
import pickle
from torch.optim import Adam, RMSprop
from args import get_args
from utile import set_seed, construct_meta_pos, get_syn_sim, get_emb
from model import SMCLMDA
from train import train_SMCLMDA

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)
    # Load dictionary from file
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def main():
    args = get_args()
    set_seed(args.seed)
    is_tenfold = 'true'
    if args.isuse == 'yes':
        mi_final, di_final = get_syn_sim(78, 37)  # The similarity matrix of each data source is fused using a nonlinear method and saved in its own directory.
    emb = get_emb() # Node2Vec acquires localized features of miRNAs and diseases
    sim_set = load_pkl(args.parent_dir, 'sim_set.pkl') # Used to obtain a fused similarity matrix or a multi-view similarity matrix
    meta_set = load_pkl(args.parent_dir, 'meta_set.pkl') # Get the similarity matrix of the meta-path
    mdm = meta_set['meta']['mdm']
    dmd = meta_set['meta']['dmd']
    pos_miRNA_mask, pos_disease_mask = construct_meta_pos(mdm, dmd, args.pos_sum) # Get masks for positive and negative sample pairs
    print('--------------------------------Data organized and ready to build the model-------------------')
    model = SMCLMDA(args)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.we_decay)
    Acc = []
    Pre = []
    Rec = []
    Spe = []
    Mcc = []
    F1 = []
    AUROC = []
    AUPR = []
    if is_tenfold == 'true':
        for i in range(10):
            print(f'-------------------the{i+1}fold-------------------------------')
            pair_pos_neg_fengceng = load_pkl(args.parent_dir_, f'pos_neg_pair_10_{i+1}.pkl')  # Load positive and negative samples for ten-fold (five-fold) training and testing
            accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train_SMCLMDA(args, model, sim_set, meta_set, emb, pos_miRNA_mask, pos_disease_mask, optimizer, pair_pos_neg_fengceng, device)
            Acc.append(accuracy)
            Pre.append(precision)
            Rec.append(recall)
            Spe.append(specificity)
            Mcc.append(mcc)
            F1.append(f1)
            AUROC.append(auc_)
            AUPR.append(aupr_)
    else:
        pair_pos_neg_fengceng = load_pkl(args.parent_dir_, 'pos_neg_pair_fengceng.pkl') # Load positive and negative samples for independent test training and testing
        accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ =  train_SMCLMDA(args, model, sim_set, meta_set, emb, pos_miRNA_mask, pos_disease_mask, optimizer,pair_pos_neg_fengceng, device)
        Acc.append(accuracy)
        Pre.append(precision)
        Rec.append(recall)
        Spe.append(specificity)
        Mcc.append(mcc)
        F1.append(f1)
        AUROC.append(auc_)
        AUPR.append(aupr_)
    print('---------------------------------Print metrics----------------------------------')
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
    print('----------------------------------- ending ----------------------------------')
if __name__ == "__main__":
    main()
