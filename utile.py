import numpy as np
import csv
import random
import scipy.sparse as sp
import pandas as pd
import torch
import os
import pickle
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############SNF进行相似度矩阵非线性融合##############
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        # md_data = np.array(md_data)
        return torch.tensor(md_data)
        # return md_data

# W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn


# updataing rules
def MiRNA_updating (S1,S2,S3,P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1,(P2+P3)/2),S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot(np.dot(S2,(P1+P3)/2),S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization(P333)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def disease_updating(S1,S2,S3, P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif> 0.0000001:
        it = it + 1
        P111 =np.dot(np.dot(S1,(P2+P3)/2), S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot(np.dot(S2,(P1+P3)/2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization(P333)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P


# 采用SNF进行多源特征融合
def get_syn_sim (k1, k2):#k1=78，k2=37
    ### 两个miRAN相似性矩阵缺失值使用0进行填充

    disease_semantic_sim = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disSSim.csv') # 疾病语义相似性
    disease_GIP_sim = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disGIPSim.csv') # 疾病的高斯核相似性
    disease_cos_sim = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disCosSim.csv') # 疾病的余弦相似性

    miRNA_GIP_sim = read_csv("data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miGIPSim.csv") # 基于GIP的miRNA相似性
    miRNA_cos_sim = read_csv("data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miCosSim.csv") # 基于cosine的miRNA相似性
    miRNA_func_sim = read_csv("data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miGIPSim.csv") # 基于func的miRNA相似性


# 对miRNA相似性矩阵进行归一化
    mi_GIP_sim_norm = new_normalization(miRNA_GIP_sim)
    mi_cos_sim_norm = new_normalization(miRNA_cos_sim)
    mi_func_sim_norm = new_normalization(miRNA_func_sim)

# 对miRNA相似性矩阵求取knn
    mi_GIP_knn = KNN_kernel(miRNA_GIP_sim, k1)
    mi_cos_knn = KNN_kernel(miRNA_cos_sim, k1)
    mi_func_knn = KNN_kernel(miRNA_func_sim, k1)

# 迭代更新每个相似性网络
    Pmi= MiRNA_updating(mi_GIP_knn, mi_cos_knn, mi_func_knn, mi_GIP_sim_norm, mi_cos_sim_norm, mi_func_sim_norm)
    Pmi_final = (Pmi + Pmi.T)/2
# 对疾病相似性矩阵进行归一化
    dis_sem_norm = new_normalization(disease_semantic_sim)
    dis_GIP_norm = new_normalization(disease_GIP_sim)
    dis_cos_norm = new_normalization(disease_cos_sim)

# 对疾病相似性矩阵进行求取knn
    dis_sem_knn = KNN_kernel(disease_semantic_sim, k2)
    dis_GIP_knn = KNN_kernel(disease_GIP_sim, k2)
    dis_cos_knn = KNN_kernel(disease_cos_sim, k2)


    Pdiease = disease_updating(dis_sem_knn, dis_GIP_knn, dis_cos_knn, dis_sem_norm, dis_GIP_norm, dis_cos_norm)
    Pdiease_final = (Pdiease+Pdiease.T)/2
# 获得最终的miRNA，疾病相似性矩阵
    return Pmi_final, Pdiease_final

# 以下代码用于获取融合的miRNA、disease特征
# mi_final, di_final = get_syn_sim(78,37)
# mi_final = pd.DataFrame(mi_final)
# di_final = pd.DataFrame(di_final)
# mi_final.to_csv('mi_final.csv', header=False, index=False)
# di_final.to_csv('di_final.csv', header=False, index=False)


# 添加自环
def self_hoop(matrix):
    for i in matrix.shape[0]:
        matrix[i][i] += 1
    return matrix

# 度归一化
def normalized(wmat):
    deg = torch.diag(torch.sum(wmat, dim=0))
    degpow = torch.pow(deg, -0.5)
    degpow[torch.isinf(degpow)] = 0
    degpow[torch.isnan(degpow)] = 0
    W = torch.mm(torch.mm(degpow, wmat), degpow)
    # deg = np.diag(np.sum(wmat,axis=0))
    # degpow = np.power(deg,-0.5)
    # degpow[np.isinf(degpow)] = 0
    # W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def get_emb():
    emb = dict()
    emb = read_csv('feature_embedding/embadding-hmdd4/emb_node2vec_all.csv')
    emb_dis = emb[:884, :].to(device)
    emb_mi = emb[884:, :].to(device)
    emb = {'miRNA': emb_mi, 'disease': emb_dis}
    return emb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置了Pytorch库的随机种子

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def get_data():
    sim_set = dict()

    # 获取使用SNF融合的miRNA、disease相似度矩阵、边集合
    mi_final = read_csv('data/mir2disease+lunwen/-1-0-+1/miRNA_sim/mi_final.csv').to(device)
    mi_final_edges = get_edge_index(mi_final).to(device)

    di_final = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/di_final.csv').to(device)
    di_final_edges = get_edge_index(di_final).to(device)
    # 保存使用SNF融合的特征
    sim_set['miRNA_snf'] = {'mi_final': mi_final, 'mi_final_edges': mi_final_edges}
    sim_set['disease_snf'] = {'di_final': di_final, 'di_final_edges': di_final_edges}

    # 加载没有融合的多视图特征
    mi_gua = read_csv('data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miGIPSim.csv').to(device)
    # mi_gua = normalized(mi_gua).to(device)
    mi_gua_edges = get_edge_index(mi_gua).to(device)
    mi_cos = read_csv('data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miCosSim.csv').to(device)
    # mi_cos = normalized(mi_cos).to(device)
    mi_cos_edges = get_edge_index(mi_cos).to(device)
    mi_fun = read_csv('data/mir2disease+lunwen/-1-0-+1/miRNA_sim/miFunSim_norm.csv').to(device)
    # mi_fun = normalized(mi_fun).to(device)
    mi_fun_edges = get_edge_index(mi_fun).to(device)

    di_gua = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disGIPSim.csv').to(device)
    # di_gua = normalized(di_gua).to(device)
    di_gua_edges = get_edge_index(di_gua).to(device)
    di_cos = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disCosSim.csv').to(device)
    # di_cos = normalized(di_cos).to(device)
    di_cos_edges = get_edge_index(di_cos).to(device)
    di_sem = read_csv('data/mir2disease+lunwen/-1-0-+1/disease_sim/disSSim.csv').to(device)
    # di_sem = normalized(di_sem).to(device)
    di_sem_edges = get_edge_index(di_sem).to(device)

    sim_set['miRNA_mut'] = {'mi_gua': mi_gua, 'mi_gua_edges': mi_gua_edges, 'mi_cos': mi_cos, 'mi_cos_edges': mi_cos_edges, 'mi_fun':mi_fun, 'mi_fun_edges': mi_fun_edges}
    sim_set['disease_mut'] = {'di_gua': di_gua, 'di_gua_edges': di_gua_edges, 'di_cos': di_cos, 'di_cos_edges': di_cos_edges, 'di_sem': di_sem, 'di_sem_edges': di_sem_edges}
    # 定义保存数据的目录和文件名
    parent_dir = "data/mir2disease+lunwen/-1-0-+1/predata"
    filename = "sim_set.pkl"
    file_path = os.path.join(parent_dir, filename)

    # 确保目录存在
    os.makedirs(parent_dir, exist_ok=True)

    # 保存字典到文件
    with open(file_path, 'wb') as file:
        pickle.dump(sim_set, file)

    print(f"sim_set saved to {file_path}")
    return sim_set
# 将中间预处理的结果保存到pkl文件中
# get_data()

# 加载正负样本用于训练和测试
def load_pos_neg():
    pos_neg_pair_fengceng = dict()
    all_train_pos = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/train/all_train_pos.csv', header=None).to_numpy() # 分层抽样已经划分好的训练集正样本：3436
    all_test_pos = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/test/all_test_pos.csv', header=None).to_numpy() # 分层抽样已经划分好的测试集正样本：792
    all_neg = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/all_neg.csv', header=None).to_numpy() # 所有负样本：157342
    #yanzheng_test_pos_neg = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/test/yanzheng_test_pos_neg.csv', header=None).to_numpy() # 所有因果关联（1）和非因果关联（0）的预测
    # 以下是分层抽样好的十倍交叉测试
    # all_train_pos = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/10-fold_/train/train_10_pos.csv', header=None).to_numpy()  # 加载所有10折中正样本
    # all_test_pos = pd.read_csv('data/HMDD V3_2+yanzheng(yinguo)/10-fold_/test/test_10_pos.csv', header=None).to_numpy()  # 加载所有10折中正样本
    # 打乱正负样本
    number_train_pos = list(range(all_train_pos.shape[0]))
    random.shuffle(number_train_pos)
    all_train_pos = all_train_pos[number_train_pos]

    number_test_pos = list(range(all_test_pos.shape[0]))
    random.shuffle(number_test_pos)
    all_test_pos = all_test_pos[number_test_pos]

    number_all_neg = list(range(all_neg.shape[0]))
    random.shuffle(number_all_neg)
    all_neg = all_neg[number_all_neg]
    # 训练集抽取相同数量的负样本进行训练
    all_train_neg = all_neg[:all_train_pos.shape[0], :]

    # 从 all_neg 中删除所有在 all_train_neg 中的行
    train_neg_set = set(map(tuple, all_train_neg))  # 以行的前三列作为唯一标识
    all_neg_filtered = np.array([row for row in all_neg if tuple(row) not in train_neg_set])

    train_pos_neg = np.concatenate((all_train_pos, all_train_neg), axis=0)
    #train_pos_neg_ = np.concatenate((all_train_pos, all_neg), axis=0)
    test_pos_neg = np.concatenate((all_test_pos, all_neg_filtered), axis=0)
    # 将结果再次打乱然后保存在pkl文件中
    number_all_train = list(range(train_pos_neg.shape[0]))
    random.shuffle(number_all_train)
    train_pos_neg = train_pos_neg[number_all_train]

    # number_all_train_ = list(range(train_pos_neg_.shape[0]))
    # random.shuffle(number_all_train_)
    # train_pos_neg_ = train_pos_neg_[number_all_train_]

    number_all_test = list(range(test_pos_neg.shape[0]))
    random.shuffle(number_all_test)
    test_pos_neg = test_pos_neg[number_all_test]

    # number_yanzheng = list(range(yanzheng_test_pos_neg.shape[0]))
    # random.shuffle(number_yanzheng)
    # yanzheng_test_pos_neg = yanzheng_test_pos_neg[number_yanzheng]

    #pos_neg_pair_fengceng = {'train': train_pos_neg, 'train_': train_pos_neg_, 'test': yanzheng_test_pos_neg}
    pos_neg_pair_fengceng = {'train': train_pos_neg, 'test': test_pos_neg}
    # 定义保存数据的目录和文件名
    parent_dir = "data/HMDD V3_2+yanzheng(yinguo)/predata"
    filename = "pos_neg_pair_fengceng.pkl"
    file_path = os.path.join(parent_dir, filename)
    # 确保目录存在
    os.makedirs(parent_dir, exist_ok=True)
    # 保存字典到文件
    with open(file_path, 'wb') as file:
        pickle.dump(pos_neg_pair_fengceng, file)
    print(f"pos_neg_pair_fengceng saved to {file_path}")
    return pos_neg_pair_fengceng

# load_pos_neg()
# 构造十折或者五折交叉
def get_fold():
    path = "data/mir2disease+lunwen/-1-0-+1/mirBase_lunwen_101_MDA_.csv"

    Rowid = []
    Cloumnid = []
    Labels = []
    Divide = []
    Rowid_neg = []
    Cloumnid_neg = []
    Labels_neg = []
    Divide_neg = []
    pos_neg_pair_fengceng = dict()

    # 读取csv并保存正负样本
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tt = 0
        for line in reader:
            for i in range(len(line)):
                if float(line[i]) == 1:
                    Divide.append("222")  # 正样本
                    Rowid.append(tt)
                    Cloumnid.append(i)
                    Labels.append(int(float(line[i])))
                elif float(line[i])==-1:
                    Divide_neg.append("111")  # 负样本
                    Rowid_neg.append(tt)
                    Cloumnid_neg.append(i)
                    Labels_neg.append(int(float(line[i])))
                else:
                    pass
            tt = tt + 1

    # print(len(Rowid), len(Cloumnid), len(Labels), len(Divide))
    # print(Rowid[0], Cloumnid[0], Labels[0], Divide[0])

    #   整合正负样本的4列，并进行打乱
    Data = [Rowid, Cloumnid, Labels, Divide]
    Data_neg = [Rowid_neg, Cloumnid_neg, Labels_neg, Divide_neg]

    Data = np.array(Data).T
    Data_neg = np.array(Data_neg).T
    print(Data.shape, Data_neg.shape)

    row = list(range(Data.shape[0]))
    random.shuffle(row)
    Data = Data[row]

    # Ten fold cross-validation
    num_cross_val = 5
    for fold in range(num_cross_val):
        train_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val != fold])
        test_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val == fold])

        clo = list(range(Data_neg.shape[0]))
        random.shuffle(clo)
        num = Data.shape[0] / 5 * 4  # 取和9折正样本相同数量的负样本
        train_neg = Data_neg[clo][:int(num), :]

        #   重新设置训练和测试标签
        for i in range(train_neg.shape[0]):
            train_neg[i][3] = "222"
        for i in range(test_pos.shape[0]):
            test_pos[i][3] = "111"

        # 删除 Data_neg 中包含在 train_neg 中的所有行
        train_neg_set = set(map(tuple, train_neg[:, :3]))  # 以前三列作为唯一标识
        Data_neg_filtered = np.array([row for row in Data_neg if tuple(row[:3]) not in train_neg_set])

        # 从 Data_neg_filtered 中随机选择与 test_pos 相同数量的样本
        # if Data_neg_filtered.shape[0] >= test_pos.shape[0]:
        #     sampled_neg = Data_neg_filtered[
        #         np.random.choice(Data_neg_filtered.shape[0], test_pos.shape[0], replace=False)]
        # else:
        #     raise ValueError("Data_neg_filtered 中的样本数量少于 test_pos 样本数量，无法选择相同数量的样本")

        # 最终组合训练和测试集
        #   最终   组合训练和测试集
        train = np.concatenate((train_pos, train_neg), axis=0)
        test = np.concatenate((test_pos, Data_neg_filtered), axis=0)
        # 将 train 中第三列的 -1 替换为 0
        train[:, 2][train[:, 2] == '-1'] = 0

        # 将 test 中第三列的 -1 替换为 0
        test[:, 2][test[:, 2] == '-1'] = 0

        # 再次打乱数据
        li = list(range(train.shape[0]))
        random.shuffle(li)
        train = train[li].astype(int)
        li = list(range(test.shape[0]))
        random.shuffle(li)
        test = test[li].astype(int)
        pos_neg_pair_fengceng = {'train': train, 'test': test}
        # 定义保存数据的目录和文件名
        parent_dir = "data/mir2disease+lunwen/-1-0-+1/5-fold"
        filename = f"pos_neg_pair_5_{fold+1}.pkl"
        file_path = os.path.join(parent_dir, filename)
        # 确保目录存在
        os.makedirs(parent_dir, exist_ok=True)
        # 保存字典到文件
        with open(file_path, 'wb') as file:
            pickle.dump(pos_neg_pair_fengceng, file)
        print(f"pos_neg_pair_5_{fold+1} saved to {file_path}")
        # 保存的文件中0：miRNA索引，1：disease索引，2：标签，3：没用
    return pos_neg_pair_fengceng
# get_fold()


# 获取经过分层抽样已经划分好的关联
def get_fengceng():
    path = "data/HMDD v_4/hmdd4_MDA_.csv"
    path1 = "data/HMDD v_4/fenceng/h4_fenceng_train_pos_mda.csv"
    path2 = "data/HMDD v_4/fenceng/h4_fenceng_test_pos_mda.csv"


    Rowid = []
    Cloumnid = []
    Labels = []
    Divide = []
    Rowid_neg = []
    Cloumnid_neg = []
    Labels_neg = []
    Divide_neg = []

    Rowid1 = []
    Cloumnid1 = []
    Labels1 = []
    Divide1 = []
    Rowid_neg1 = []
    Cloumnid_neg1 = []
    Labels_neg1 = []
    Divide_neg1 = []

    Rowid2 = []
    Cloumnid2 = []
    Labels2 = []
    Divide2 = []
    Rowid_neg2 = []
    Cloumnid_neg2 = []
    Labels_neg2 = []
    Divide_neg2 = []
    pos_neg_pair_fengceng = dict()

    # 读取csv并保存正负样本
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tt = 0
        for line in reader:
            for i in range(len(line)):
                if float(line[i]) == 1:
                    Divide.append("222")  # 正样本
                    Rowid.append(tt)
                    Cloumnid.append(i)
                    Labels.append(int(float(line[i])))
                else:
                    Divide_neg.append("111")  # 负样本
                    Rowid_neg.append(tt)
                    Cloumnid_neg.append(i)
                    Labels_neg.append(int(float(line[i])))

            tt = tt + 1

    with open(path1, 'r', newline='') as csv_file1:
        reader1 = csv.reader(csv_file1)
        tt1 = 0
        for line1 in reader1:
            for i in range(len(line1)):
                if float(line1[i]) == 1:
                    Divide1.append("222")  # 正样本
                    Rowid1.append(tt1)
                    Cloumnid1.append(i)
                    Labels1.append(int(float(line1[i])))
                else:
                    Divide_neg1.append("111")  # 负样本
                    Rowid_neg1.append(tt1)
                    Cloumnid_neg1.append(i)
                    Labels_neg1.append(int(float(line1[i])))

            tt1 = tt1 + 1

    with open(path2, 'r', newline='') as csv_file2:
        reader2 = csv.reader(csv_file2)
        tt2 = 0
        for line2 in reader2:
            for i in range(len(line2)):
                if float(line2[i]) == 1:
                    Divide2.append("222")  # 正样本
                    Rowid2.append(tt2)
                    Cloumnid2.append(i)
                    Labels2.append(int(float(line2[i])))
                else:
                    Divide_neg2.append("111")  # 负样本
                    Rowid_neg2.append(tt2)
                    Cloumnid_neg2.append(i)
                    Labels_neg2.append(int(float(line2[i])))

            tt2 = tt2 + 1

    # print(len(Rowid), len(Cloumnid), len(Labels), len(Divide))
    # print(Rowid[0], Cloumnid[0], Labels[0], Divide[0])

    #   整合负样本的4列，并进行打乱
    Data_neg = [Rowid_neg, Cloumnid_neg, Labels_neg, Divide_neg]
    Data_neg = np.array(Data_neg).T   # 所有的负样本




    train_pos = [Rowid1, Cloumnid1, Labels1, Divide1]
    train_pos = np.array(train_pos).T
    test_pos = [Rowid2, Cloumnid2, Labels2, Divide2]
    test_pos = np.array(test_pos).T


    clo = list(range(Data_neg.shape[0]))
    random.shuffle(clo)
    num = train_pos.shape[0]  # 训练集中取正样本相同数量的负样本
    train_neg = Data_neg[clo][:int(num), :]


    # 删除 Data_neg 中包含在 train_neg 中的所有行
    train_neg_set = set(map(tuple, train_neg[:, :3]))  # 以前三列作为唯一标识
    Data_neg_filtered = np.array([row for row in Data_neg if tuple(row[:3]) not in train_neg_set]) # 这些是不包含训练集的所有负样本

    # 从 Data_neg_filtered 中随机选择与 test_pos 相同数量的样本
    # if Data_neg_filtered.shape[0] >= test_pos.shape[0]:
    #     sampled_neg = Data_neg_filtered[
    #         np.random.choice(Data_neg_filtered.shape[0], test_pos.shape[0], replace=False)]
    # else:
    #     raise ValueError("Data_neg_filtered 中的样本数量少于 test_pos 样本数量，无法选择相同数量的样本")

    # 最终组合训练和测试集
    #   最终   组合训练和测试集
    train = np.concatenate((train_pos, train_neg), axis=0)
    test = np.concatenate((test_pos, Data_neg_filtered), axis=0)
    # 再次打乱数据
    li = list(range(train.shape[0]))
    random.shuffle(li)
    train = train[li].astype(int)
    li = list(range(test.shape[0]))
    random.shuffle(li)
    test = test[li].astype(int)

    pos_neg_pair_fengceng = {'train': train, 'test': test}
    # 定义保存数据的目录和文件名
    parent_dir = "data/HMDD v_4/fenceng/"
    filename = f"pos_neg_pair_fengceng.pkl"
    file_path = os.path.join(parent_dir, filename)
    # 确保目录存在
    os.makedirs(parent_dir, exist_ok=True)
    # 保存字典到文件
    with open(file_path, 'wb') as file:
        pickle.dump(pos_neg_pair_fengceng, file)
    print(f"pos_neg_pair_fengceng saved to {file_path}")
        # 保存的文件中0：miRNA索引，1：disease索引，2：标签，3：没用
    return pos_neg_pair_fengceng
# get_fengceng()

# 构造元路径用于对比学习
def construct_meta_path():
    """
    这里我们构造了四条元路径：
    M-D-M：miRNA-disease-miRNA   A * A.t()
    M-D-M-D-M：A * A.t() * A * A.t()
    D-M-D: A.t() * A
    D-M-D-M-D: A.t() * A * A.t() * A
    :return:
    """
    print('--------------------------------------start to construct meta-path-------------------------')
    meta_set = dict()
    mda = read_csv('data/mir2disease+lunwen/-1-0-+1/mirBase_lunwen_101_MDA_.csv').to(device)

    mdm = torch.mm(mda, mda.t())
    mdmdm = torch.mm(torch.mm(mdm, mda), mda.t())
    dmd = torch.mm(mda.t(), mda)
    dmdmd = torch.mm(torch.mm(dmd, mda.t()), mda)
    # 对不同元路径得出的相似性进行归一化
    mdm_norm = normalized(mdm).to(device)
    mdmdm_norm = normalized(mdmdm).to(device)
    dmd_norm = normalized(dmd).to(device)
    dmdmd_norm = normalized(dmdmd).to(device)

    mdm_edges = get_edge_index(mdm_norm).to(device)
    mdmdm_edges = get_edge_index(mdmdm_norm).to(device)
    dmd_edges = get_edge_index(dmd_norm).to(device)
    dmdmd_edges = get_edge_index(dmdmd_norm).to(device)
    meta_set['miRNA'] = {'mdm': mdm_norm, 'mdm_edges': mdm_edges, 'mdmdm': mdmdm_norm, 'mdmdm_edges': mdmdm_edges}
    meta_set['disease'] = {'dmd': dmd_norm, 'dmd_edges': dmd_edges, 'dmdmd': dmdmd_norm, 'dmdmd_edges': dmdmd_edges}
    meta_set['meta'] = {'mdm': mdm, 'dmd': dmd}
    meta_set['mda_whole'] = {'mda': mda}
    # 定义保存数据的目录和文件名
    parent_dir = "data/mir2disease+lunwen/-1-0-+1/predata"
    filename = "meta_set.pkl"
    file_path = os.path.join(parent_dir, filename)

    # 确保目录存在
    os.makedirs(parent_dir, exist_ok=True)

    # 保存字典到文件
    with open(file_path, 'wb') as file:
        pickle.dump(meta_set, file)

    print(f"meta_set saved to {file_path}")
    return meta_set, mdm, dmd
# 将中间预处理的结果保存到pkl文件中
# construct_meta_path()

# 构建元路径上的正样本对
def construct_meta_pos(mdm, dmd, pos_sum):
    print('----------------------------------start to construct postive sample pairs----------------------')
    mdm = mdm.detach().cpu().numpy()
    dmd = dmd.detach().cpu().numpy()
    dia_miRNA = sp.dia_matrix((np.ones(mdm.shape[0]), 0), shape=(mdm.shape[0], mdm.shape[1])).toarray()
    m_info = np.ones((mdm.shape[0], mdm.shape[1])) - dia_miRNA
    mdm = mdm * m_info  # Hadamard乘积也为元素乘积
    mdm = torch.tensor(mdm)

    pos_miRNA = np.zeros((mdm.shape[0], mdm.shape[1]))
    k_miRNA = 0
    for i in range(mdm.shape[0]):
        """
        选取自身为正样本对，然后选取元路径中的pos_num - 1个作为正样本
        """
        pos_miRNA[i, i] = 1
        rownon_index_miRNA = mdm[i].nonzero().view(-1)
        if len(rownon_index_miRNA) > pos_sum - 1:
            sort_miRNA = np.argsort(-mdm[i, rownon_index_miRNA])
            select_miRNA = rownon_index_miRNA[sort_miRNA[:pos_sum - 1]]
            pos_miRNA[i, select_miRNA] = 1
            k_miRNA += 1
        else:
            pos_miRNA[i, rownon_index_miRNA] = 1

    # 构建disease的正样本对
    dia_disease = sp.dia_matrix((np.ones(dmd.shape[0]), 0), shape=(dmd.shape[0], dmd.shape[1])).toarray()
    d_info = np.ones((dmd.shape[0], dmd.shape[1])) - dia_disease
    dmd = dmd * d_info
    dmd = torch.tensor(dmd)

    pos_disease = np.zeros((dmd.shape[0], dmd.shape[1]))
    k_disease = 0
    for j in range(dmd.shape[0]):
        pos_disease[j, j] = 1
        rownon_index_disease = dmd[j].nonzero().view(-1)
        if len(rownon_index_disease) > pos_sum - 1:
            sort_disease = np.argsort(-dmd[j, rownon_index_disease])
            select_disease = rownon_index_disease[sort_disease[:pos_sum - 1]]
            pos_disease[j, select_disease] = 1
            k_disease += 1
        else:
            pos_disease[j, rownon_index_disease] = 1

    # 归一化
    # pos_miRNA = normalize_sys(pos_miRNA)
    # pos_disease = normalize_sys(pos_disease)
    # 返回的是一个掩码矩阵
    return torch.tensor(pos_miRNA).to(device), torch.tensor(pos_disease).to(device)





