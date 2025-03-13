import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
import os
from sklearn.decomposition import PCA
# import umap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the Normalization Layer
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
        """
        assert mode in ['PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        col_mean = x.mean(dim=0)
        if self.mode == 'PN-SI':
            x = x - col_mean  # center
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual  # scale

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


# Define semantic layer attention SLA for fusion of view information under different original paths.
class SLAttention(nn.Module):
    def __init__(self, hidden, dropout):
        super(SLAttention, self).__init__()
        self.fc = nn.Linear(hidden, hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.act_tan = nn.Tanh()
        self.a = nn.Parameter(torch.empty(size=(1, hidden)), requires_grad=True)
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        #self.act_sof = torch.nn.functional.softmax()

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x 

    def forward(self, list_): 
        beta = []
        a = self.a
        for view_ in list_:
            view_ = self.dropout(view_)
            feat = self.act_tan(self.fc(view_)).mean(dim=0)
            beta.append(a.matmul(feat.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = torch.nn.functional.softmax(beta, dim=-1)

        fin_metra = 0
        for i in range(len(list_)):
            fin_metra += list_[i] * beta[i]
        return fin_metra

# Define the meta-path map convolution
class GConv_meta(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.8, bias=True):
        super(GConv_meta, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        self.drop = nn.Dropout(drop)
        self.acti_fun = nn.PReLU() # PRelu

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            self.bias.data.fill_(0.001)
        else:
            self.register_parameter('bias', None)
        for model in self.modules():
            self.weight_init(model)
    def weight_init(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_normal_(model.weight, gain=1.414)
            if model.bias is not None:
                model.bias.data.fill_(0.0)

    def forward(self, emb, meta):
        emb = F.dropout(emb, 0.3)
        emb_feat = self.fc(emb)
        out = torch.spmm(meta, emb_feat)
        if self.bias is not None:
            out += self.bias
        out = self.drop(out)
        out = self.acti_fun(out)
        return out

# Define similarity map convolution
class GCN_sim(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCN_sim, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.PReLU()
        self.feat_drop = 0.3
        self.lin = nn.Linear(in_channels,64)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(64))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for model in self.modules():
            self.weights_init(model)

    def weights_init(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_normal_(model.weight, gain=1.414)
            if model.bias is not None:
                model.bias.data.fill_(0.0)

    def forward(self, emb, sim, flag):
        if flag == 'yes':
            emb = self.drop_out(emb)
        emb_feat = self.fc(emb)
        emb_feat = self.lin(emb_feat)
        out = torch.mm(sim, emb_feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# ablation experiment
class SimpleLinearModel(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)  # linear layer
        self.kan = nn.Linear(in_channels, 64)  
        self.drop_out = nn.Dropout(0.3)  # dropout
        self.act = nn.PReLU() 
        self.lin = nn.Linear(in_channels, 64) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(64))
            self.bias.data.fill_(0.0)  
        else:
            self.register_parameter('bias', None)

        for model in self.modules():
            self.weights_init(model) 

    def weights_init(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_normal_(model.weight, gain=1.414)  
            if model.bias is not None:
                model.bias.data.fill_(0.0)

    def forward(self, emb, flag):
        # The flag variable determines whether to use dropout
        if flag == 'yes':
            emb = self.drop_out(emb)  # dropout

        emb_feat = self.fc(emb)  
        if self.bias is not None:
            emb_feat += self.bias 

        out = self.act(emb_feat) 

# Defining Comparative Learning
class contrast_learning(nn.Module):
    def __init__(self, hidden, temperature, lambda_1):
        super(contrast_learning, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden)
        )
        self.temperature = temperature
        self.lambda_1 = lambda_1

        for fc in self.project:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    # The similarity between the two views is calculated for the subsequent loss function
    def similarity(self, meta_view, sim_view):
        meta_view_norm = torch.norm(meta_view, dim=-1, keepdim=True)
        sim_view_norm = torch.norm(sim_view, dim=-1, keepdim=True)
        view_dot_fenzi = torch.mm(meta_view, sim_view.t())
        view_dot_fenmu = torch.mm(meta_view_norm, sim_view_norm.t())
        sim_matrix = torch.exp(view_dot_fenzi / view_dot_fenmu / self.temperature)
        return sim_matrix

    def forward(self, meta_, sim_, posSamplePairs):
        # Projection of features through a linear layer
        meta_project = self.project(meta_)
        sim_project = self.project(sim_)
        view_sim = self.similarity(meta_project, sim_project)
        view_sim_T = view_sim.t()

        view_sim = view_sim / (torch.sum(view_sim, dim=1).view(-1, 1) + 1e-8)
        loss_meta = -torch.log(view_sim.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        view_sim_T = view_sim_T / (torch.sum(view_sim_T, dim=1).view(-1, 1) + 1e-8)
        loss_sim = -torch.log(view_sim_T.mul(posSamplePairs.to(device)).sum(dim=-1)).mean()

        return self.lambda_1 * loss_meta + (1 - self.lambda_1) * loss_sim

# Defining miRNA embedding in graph convolutional layers
class mi_embadding(nn.Module):
    def __init__(self, args):
        super(mi_embadding, self).__init__()
        self.args = args
       
        self.gcn = GCN_sim(self.args.fm, self.args.fm)

        # For ablation experiments
        self.lin1 = SimpleLinearModel(self.args.fm, self.args.fm)
        self.lin2 = SimpleLinearModel(self.args.fm, self.args.fm)

        # Pipeline Attention
        self.mi_fc1 = nn.Linear(in_features=6,
                                out_features=5 * 6)
        self.mi_fc2 = nn.Linear(in_features=5 * 6,
                                out_features=6)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_mi = nn.Conv2d(in_channels=6, out_channels=1,
                                kernel_size=(1, 1), stride=1, bias=True)
        # PCA
        self.pca = PCA(n_components=64)
        # self.umap = umap.UMAP(n_components=64) # 主成分变体
        self.lle = LocallyLinearEmbedding(n_neighbors=8, n_components=64, method='standard')


        # self.mi_fc1 = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
        #                        out_features=5 * self.args.view * self.args.gcn_layers)
        # self.mi_fc2 = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
        #                        out_features=self.args.view * self.args.gcn_layers)
        # self.sigmoidx = nn.Sigmoid()
        # self.cnn_mi = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1,
        #                        kernel_size=(1, 1), stride=1, bias=True)


    def forward(self, sim_set, emb, flag):
        # Using randomly initialized features without Node2Vec
        x_m = torch.randn(728, 64).to(device)
        # Characterization of miRNAs using SNF fusions
        # mi_final_1 =  self.gcn(emb['miRNA'], sim_set['miRNA']['mi_final_mat'].to(device), flag)
        # mi_final_2 = self.gcn(mi_final_1, sim_set['miRNA']['mi_final_mat'].to(device), flag)

        # Characterization using unfused miRNAs
        # For newly organized datasets Multiple views
        # mi_gua_1 = self.gcn(emb, sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        # # mi_gua_2 = self.gcn(mi_gua_1, sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        # mi_cos_1 = self.gcn(emb, sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        # # mi_cos_2 = self.gcn(mi_cos_1, sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        # mi_fun_1 = self.gcn(emb, sim_set['miRNA_mut']['mi_fun'].to(device), flag)
        # # mi_fun_2 = self.gcn(mi_fun_1, sim_set['miRNA_mut']['mi_fun'].to(device), flag)

        mi_gua_1 = self.gcn(emb['miRNA'], sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        mi_gua_2 = self.gcn(mi_gua_1, sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        mi_cos_1 = self.gcn(emb['miRNA'], sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        mi_cos_2 = self.gcn(mi_cos_1, sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        mi_fun_1 = self.gcn(emb['miRNA'], sim_set['miRNA_mut']['mi_fun'].to(device), flag)
        mi_fun_2 = self.gcn(mi_fun_1, sim_set['miRNA_mut']['mi_fun'].to(device), flag)

        # Using random initial features
        # mi_gua_1 = self.gcn(x_m, sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        # mi_gua_2 = self.gcn(mi_gua_1, sim_set['miRNA_mut']['mi_gua'].to(device), flag)
        # mi_cos_1 = self.gcn(x_m, sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        # mi_cos_2 = self.gcn(mi_cos_1, sim_set['miRNA_mut']['mi_cos'].to(device), flag)
        # mi_fun_1 = self.gcn(x_m, sim_set['miRNA_mut']['mi_fun'].to(device), flag)
        # mi_fun_2 = self.gcn(mi_fun_1, sim_set['miRNA_mut']['mi_fun'].to(device), flag)

        # Graph Convolutional Replacement of Linear Layers for Ablation Experiments
        # mi_gua_1 = self.lin1(emb['miRNA'], flag)
        # mi_gua_2 = self.lin2(mi_gua_1, flag)
        # mi_cos_1 = self.lin1(emb['miRNA'], flag)
        # mi_cos_2 = self.lin2(mi_cos_1, flag)
        # mi_fun_1 = self.lin1(emb['miRNA'], flag)
        # mi_fun_2 = self.lin2(mi_fun_1, flag)


        # SNF fusion view for newly organized datasets
        # mi_final_1 = self.gcn(emb['miRNA'], sim_set['miRNA_snf']['mi_final'].to(device), flag)
        # mi_final_2 = self.gcn(mi_final_1, sim_set['miRNA_snf']['mi_final'].to(device), flag)

        # Pipeline Attention
        # XM = torch.cat((mi_gene_1,mi_gene_2,mi_gua_1,mi_gua_2,mi_seq_1,mi_seq_2,mi_tf_1,mi_tf_2,mi_ex_1,mi_ex_2,mi_pa_1,mi_pa_2), 1).t()
        # XM = torch.cat((mi_gua_1, mi_gua_2, mi_cos_1, mi_cos_2, mi_fun_1, mi_fun_2), 1)
        # XM = XM.view(1, 6, 64, -1)
        # # XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
        #
        # globalAvgPool_x = nn.AvgPool2d((self.args.fm, 728), (1, 1))
        # x_channel_attention = globalAvgPool_x(XM)
        #
        # x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
        # x_channel_attention = self.mi_fc1(x_channel_attention)
        # x_channel_attention = torch.relu(x_channel_attention)
        # x_channel_attention = self.mi_fc2(x_channel_attention)
        # x_channel_attention = self.sigmoidx(x_channel_attention)
        # x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
        # XM_channel_attention = x_channel_attention * XM
        # XM_channel_attention = torch.relu(XM_channel_attention)
        #
        # x = self.cnn_mi(XM_channel_attention)
        # mi_emb = x.view(self.args.fm, 728).t()
        # PCA
        # XM for newly organized datasets
        XM = torch.cat((mi_gua_1,mi_gua_2,mi_cos_1,mi_cos_2,mi_fun_1,mi_fun_2), 1)
        # XM = torch.cat((mi_gua_1, mi_gua_2, mi_cos_1,mi_cos_2), 1)
        # XM = torch.cat((mi_gua_1,mi_cos_1,mi_fun_1), 1)
        # XM = torch.cat((mi_final_1, mi_final_2), 1)


        XM = self.pca.fit_transform(XM.cpu().detach().numpy())
        XM = torch.Tensor.cpu(torch.from_numpy(XM)).to(device)
        mi_emb = XM

        # Feature extraction using weighted averaging
        # mi_emb = (mi_gua_1+mi_gua_2+mi_cos_1+mi_cos_2+mi_fun_1+mi_fun_2)/6

        return mi_emb.float()

# Define the embedding of disase in the graph convolutional layer
class di_embadding(nn.Module):
    def __init__(self, args):
        super(di_embadding, self).__init__()
        self.args = args

        self.gcn = GCN_sim(self.args.fd, self.args.fd)

        #ablation experiment
        self.lin1 = SimpleLinearModel(self.args.fd, self.args.fd)
        self.lin2 = SimpleLinearModel(self.args.fd, self.args.fd)
        # Pipeline Attention
        self.di_fc1 = nn.Linear(in_features=6,
                                out_features=5 * 6)
        self.di_fc2 = nn.Linear(in_features=5* 6,
                                out_features=6)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_di = nn.Conv2d(in_channels=6, out_channels=1,
                                kernel_size=(1, 1), stride=1, bias=True)
        # PCA
        self.pca = PCA(n_components=64)
        # self.umap = umap.UMAP(n_components=64)
        self.lle = LocallyLinearEmbedding(n_neighbors=8, n_components=64, method='standard')
        # self.lin = nn.Linear(12,64)  

        # self.di_fc1 = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
        #                        out_features=5 * self.args.view * self.args.gcn_layers)
        # self.di_fc2 = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
        #                        out_features=self.args.view * self.args.gcn_layers)
        # self.sigmoidy = nn.Sigmoid()
        # self.cnn_di = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1,
        #                        kernel_size=(1, 1), stride=1, bias=True)

    def forward(self,sim_set, emb, flag):
        # Using randomly initialized features without Node2Vec
        d_m = torch.randn(884, 64).to(device)
        # Disease characterization using SNF fusion
        # di_final_1 =  self.gcn(emb['disease'], sim_set['disease']['di_final_mat'].to(device), flag)
        # di_final_2 = self.gcn(di_final_1, sim_set['disease']['di_final_mat'].to(device), flag)

        # Use of unfused disease characteristics
        # For newly organized data sets

        # di_gua_1 = self.gcn(emb, sim_set['disease_mut']['di_gua'].to(device), flag) # 
        # # di_gua_2 = self.gcn(di_gua_1, sim_set['disease_mut']['di_gua'].to(device), flag)
        # di_cos_1 = self.gcn(emb, sim_set['disease_mut']['di_cos'].to(device), flag)
        # # di_cos_2 = self.gcn(di_cos_1, sim_set['disease_mut']['di_cos'].to(device), flag)
        # di_sem_1 = self.gcn(emb, sim_set['disease_mut']['di_sem'].to(device), flag)
        # # di_sem_2 = self.gcn(di_sem_1, sim_set['disease_mut']['di_sem'].to(device), flag)

        di_gua_1 = self.gcn(emb['disease'], sim_set['disease_mut']['di_gua'].to(device), flag)
        di_gua_2 = self.gcn(di_gua_1, sim_set['disease_mut']['di_gua'].to(device), flag)
        di_cos_1 = self.gcn(emb['disease'], sim_set['disease_mut']['di_cos'].to(device), flag)
        di_cos_2 = self.gcn(di_cos_1, sim_set['disease_mut']['di_cos'].to(device), flag)
        di_sem_1 = self.gcn(emb['disease'], sim_set['disease_mut']['di_sem'].to(device), flag)
        di_sem_2 = self.gcn(di_sem_1, sim_set['disease_mut']['di_sem'].to(device), flag)

        # Using random initial features
        # di_gua_1 = self.gcn(d_m, sim_set['disease_mut']['di_gua'].to(device), flag)
        # di_gua_2 = self.gcn(di_gua_1, sim_set['disease_mut']['di_gua'].to(device), flag)
        # di_cos_1 = self.gcn(d_m, sim_set['disease_mut']['di_cos'].to(device), flag)
        # di_cos_2 = self.gcn(di_cos_1, sim_set['disease_mut']['di_cos'].to(device), flag)
        # di_sem_1 = self.gcn(d_m, sim_set['disease_mut']['di_sem'].to(device), flag)
        # di_sem_2 = self.gcn(di_sem_1, sim_set['disease_mut']['di_sem'].to(device), flag)

        # Graph Convolutional Replacement of Linear Layers for Ablation Experiments
        # di_gua_1 = self.lin1(emb['disease'], flag)
        # di_gua_2 = self.lin2(di_gua_1, flag)
        # di_cos_1 = self.lin1(emb['disease'], flag)
        # di_cos_2 = self.lin2(di_cos_1, flag)
        # di_sem_1 = self.lin1(emb['disease'], flag)
        # di_sem_2 = self.lin2(di_sem_1, flag)

        # SNF fusion view for newly organized datasets
        # di_final_1 = self.gcn(emb['disease'], sim_set['disease_snf']['di_final'].to(device), flag)
        # di_final_2 = self.gcn(di_final_1, sim_set['disease_snf']['di_final'].to(device), flag)

        # Pipeline Attention
        # XM = torch.cat((di_sem_1,di_gua_2,di_gua_1,di_gua_2), 1).t()
        # XM = torch.cat((di_gua_1, di_gua_2, di_cos_1, di_cos_2, di_sem_1, di_sem_2), 1)
        # XM = XM.view(1, 6, self.args.fd, -1)
        # # XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
        #
        # globalAvgPool_x = nn.AvgPool2d((self.args.fd, 884), (1, 1))
        # x_channel_attention = globalAvgPool_x(XM)
        #
        # x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
        # x_channel_attention = self.di_fc1(x_channel_attention)
        # x_channel_attention = torch.relu(x_channel_attention)
        # x_channel_attention = self.di_fc2(x_channel_attention)
        # x_channel_attention = self.sigmoidy(x_channel_attention)
        # x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
        # XM_channel_attention = x_channel_attention * XM
        # XM_channel_attention = torch.relu(XM_channel_attention)
        #
        # x = self.cnn_di(XM_channel_attention)
        # di_emb = x.view(self.args.fd, 884).t()
        # PCA
        # XM of the newly organized dataset
        XM = torch.cat((di_gua_1,di_gua_2,di_cos_1,di_cos_2,di_sem_1,di_sem_2), 1)
        # XM = torch.cat((di_gua_1, di_gua_2, di_cos_1,di_cos_2), 1)
        # XM = torch.cat((di_gua_1,di_cos_1,di_sem_1), 1)
        # XM = torch.cat((di_final_1, di_final_2), 1)

        XM = self.pca.fit_transform(XM.cpu().detach().numpy())
        XM = torch.Tensor.cpu(torch.from_numpy(XM)).to(device)
        # XM = self.lin(XM)
        di_emb = XM
        # Feature extraction using weighted averaging
        # di_emb = (di_gua_1+di_gua_2+di_cos_1+di_cos_2+di_sem_1+di_sem_2)/6
        return di_emb.float()

# Defining the extraction of meta-path features
class meta_emb(nn.Module):
    def __init__(self, args):
        super(meta_emb, self).__init__()
        self.mdm = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.mdmdm = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.dmd = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.dmdmd = GConv_meta(args.meta_inchannels, args.meta_outchannels)
        self.SLA = SLAttention(args.sla_hidden, args.sla_dropout)

    def forward(self, emb, meta):
        mdm = self.mdm(emb['miRNA'], meta['miRNA']['mdm'].to(device))
        mdmdm = self.mdmdm(emb['miRNA'], meta['miRNA']['mdmdm'].to(device))
        dmd = self.dmd(emb['disease'], meta['disease']['dmd'].to(device))
        dmdmd = self.dmdmd(emb['disease'], meta['disease']['dmdmd'].to(device))

        list_view_mi = [mdm, mdmdm]
        list_view_di = [dmd, dmdmd]
        meta_mi_emb = self.SLA(list_view_mi)
        meta_di_emb = self.SLA(list_view_di)
        return meta_mi_emb, meta_di_emb

# MLP for predicting
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.acti_func = torch.sigmoid
        self.mi_fc = nn.Linear(64, 64, bias=True)
        self.di_fc = nn.Linear(64, 64, bias=True)
        self.linear = nn.Linear(64, 1, bias=True)
        self.decode = nn.Linear(64, 64, bias=True)
    def forward(self, mi_emb, di_emb, mi_index, di_index):
        mi_feat = mi_emb[mi_index]
        di_feat = di_emb[di_index]
        mi_feat = self.mi_fc(mi_feat)
        di_feat = self.di_fc(di_feat)
        pair_feat = mi_feat * di_feat
        pair_feat = F.relu(self.decode(pair_feat))
        pair_feat = F.dropout(pair_feat, 0.2)
        out = self.linear(pair_feat)
        return torch.sigmoid(out)


# SMCLMDA
class SMCLMDA(nn.Module):
    def __init__(self, args):
        super(SMCLMDA, self).__init__()
        self.args = args
        self.mi_emb = mi_embadding(args)
        self.di_emb = di_embadding(args)
        self.meta_emb = meta_emb(args)
        # Building Contrastive Learning Modules
        self.CL_mi = contrast_learning(args.cl_hidden, args.temperature, args.lambda_1)
        self.CL_di = contrast_learning(args.cl_hidden, args.temperature, args.lambda_1)
        self.LayerNorm = torch.nn.LayerNorm(64)
        self.linear = nn.Linear(64, 64)
        self.mlp = MLP()
    # def forward(self, sim_set, meta_set, emb, pos_miRNA, pos_disease, miRNA_index, disease_index, mda):
    def forward(self, sim_set, meta_set, emb, pos_miRNA, pos_disease, miRNA_index, disease_index):
        # mi_emb = self.mi_emb(sim_set, mda, 'yes')
        # di_emb = self.di_emb(sim_set, mda.T, 'yes')
        mi_emb = self.mi_emb(sim_set, emb, 'yes')
        di_emb = self.di_emb(sim_set, emb, 'yes')


        meta_mi_emb, meta_di_emb = self.meta_emb(emb, meta_set) # Getting a characterization of meta-paths
        # constrastive learning
        loss_cl = self.CL_mi(meta_mi_emb, mi_emb, pos_miRNA) + self.CL_di(meta_di_emb, di_emb, pos_disease)
        # mi_emb_1 = self.mi_emb(sim_set, mda, 'no')
        # di_emb_1 = self.di_emb(sim_set, mda.T, 'no')
        mi_emb_1 = self.mi_emb(sim_set, emb, 'no')
        di_emb_1 = self.di_emb(sim_set, emb, 'no')

        mi_emb_ = self.LayerNorm(mi_emb_1)
        mi_emb_ = F.relu(self.linear(mi_emb_))
        di_emb_ = self.LayerNorm(di_emb_1)
        di_emb_ = F.relu(self.linear(di_emb_))


        train_score = self.mlp(mi_emb_, di_emb_, miRNA_index, disease_index)

        # meta_emb_ = (meta_mi_emb.mm(meta_di_emb.t())).to(device)
        # meta_score = self.mlp(meta_emb_)
        # score = 0.5 * meta_score + 0.5 * train_score
        #return train_score.view(-1 ), loss_cl
        # return train_score.view(-1), loss_cl
        return train_score.view(-1), loss_cl
        # return train_score.view(-1)



