from layer import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module

class GCN_sim(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCN_sim, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.PReLU()
        self.feat_drop = 0.3
        self.lin = nn.Linear(in_channels,128)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(128))
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

    def forward(self, emb, sim):
        emb_feat = self.fc(emb)
        emb_feat = self.lin(emb_feat)
        out = torch.mm(sim, emb_feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Model(nn.Module):
    def __init__(self, num_in_node = 884, num_in_edge = 728, num_hidden1 = 512, num_out=128):  # 435, 757, 512, 128
        super(Model, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, 512, 0.3)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, 512, 0.3)

        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)


        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)

        self.dropout = nn.Dropout(0.8)
        self.act = torch.sigmoid
        self.li1 = nn.Linear(128, 1)
        self.gcn1 = GCN_sim(246, 246)
        self.gcn2 = GCN_sim(270, 270)

    def sample_latent(self, z_node, z_hyperedge):
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)  # sigma
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_  # Reparameterization trick
        else:
            return self.z_node_mean, self.z_edge_mean


    def forward(self,AT, A , HMG, HDG, mir_feat, dis_feat, HMD, HDM , HMM , HDD, train_edges):
        train_edges = train_edges.T
        m_index = train_edges[0]
        d_index = train_edges[1]

        z_node_encoder = self.node_encoders1(AT)
        z_hyperedge_encoder = self.hyperedge_encoders1(A)
        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        ed_auto = torch.index_select(self.z_node_mean, 0, d_index).float()
        em_auto = torch.index_select(self.z_edge_mean, 0, m_index).float()
        pair_feat_auto = (em_auto * ed_auto).float()
        pair_feat_auto = torch.sigmoid(self.li1(pair_feat_auto))


        # mir_feature_1 = self.gcn1(MD, HMG).float()
        # dis_feature_1 = self.gcn2(MD.T, HDG).float()
        mir_feature_1 = self.hgnn_hyperedge2(mir_feat, HMG).float()
        dis_feature_1 = self.hgnn_node2(dis_feat, HDG).float()
        em_1 =torch.index_select(mir_feature_1, 0, m_index)
        ed_1 =torch.index_select(dis_feature_1, 0, d_index)
        pair_feat_1 = (em_1 * ed_1).float()
        pair_feat_1 = torch.sigmoid(self.li1(pair_feat_1.cuda()))

        # mir_feature_2 = self.gcn1(MD, HMM).float()
        # dis_feature_2 = self.gcn2(MD.T, HDM).float()
        mir_feature_2 = self.hgnn_hyperedge2(mir_feat, HMM).float()
        dis_feature_2 = self.hgnn_node2(dis_feat, HDM).float().float()
        em_2 = torch.index_select(mir_feature_2, 0, m_index)
        ed_2 = torch.index_select(dis_feature_2, 0, d_index)
        pair_feat_2 = (em_2 * ed_2).float()
        pair_feat_2 = torch.sigmoid(self.li1(pair_feat_2.cuda()))

        # mir_feature_3 = self.gcn1(MD, HMD).float()
        # dis_feature_3 = self.gcn2(MD.T, HDD).float()
        mir_feature_3 = self.hgnn_hyperedge2(mir_feat, HMD).float()
        dis_feature_3 = self.hgnn_node2(dis_feat, HDD).float().float()
        em_3 = torch.index_select(mir_feature_3, 0, m_index)
        ed_3 = torch.index_select(dis_feature_3, 0, d_index)
        pair_feat_3 = (em_3 * ed_3).float()
        pair_feat_3 = torch.sigmoid(self.li1(pair_feat_3.cuda()))

        result_si = (pair_feat_1 + pair_feat_2 + pair_feat_3) / 3
        result_con = 0.1*pair_feat_auto + 0.9*result_si

        # reconstructionMMDD = self.decoder1(dis_feature_3, mir_feature_3)
        # reconstructionMD = self.decoder1(dis_feature_2, mir_feature_2)
        # reconstructionG = self.decoder1(dis_feature_1, mir_feature_1)
        reconstruction_en = self.decoder2(self.z_node_mean, self.z_edge_mean)
        reconstruction_en_ = reconstruction_en.T[m_index, d_index]


        # result = self.z_node_mean.mm(self.z_edge_mean.t())
        # result_h = (reconstructionG + reconstructionMD + reconstructionMMDD)/3

        # recover = 0.1*result + 0.9*result_h

        return reconstruction_en_.view(-1), pair_feat_auto.view(-1), pair_feat_1.view(-1), pair_feat_2.view(-1), pair_feat_3.view(-1),  result_con.view(-1),result_si.view(-1),mir_feature_1 , mir_feature_2 , mir_feature_3 , dis_feature_1 ,dis_feature_2 ,dis_feature_3


