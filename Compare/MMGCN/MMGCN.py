import torch
from torch import nn
from torch_geometric.nn import GCNConv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

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


class MMGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MMGCN, self).__init__()
        self.args = args
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)

        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn1 = GCN_sim(293, 293)
        self.gcn2 = GCN_sim(533, 533)

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))

        self.fc1_x = nn.Linear(in_features=6, out_features=5*6)
        self.fc2_x = nn.Linear(in_features=5*6, out_features=6)

        self.fc1_y = nn.Linear(in_features=6, out_features=5 * 6)
        self.fc2_y = nn.Linear(in_features=5 * 6, out_features=6)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
        self.cnn_y = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
        self.linear = nn.Linear(128, 1, bias=True)
    def forward(self, data, train_edges):
    # def forward(self, sim_set, train_edges, emb):
        torch.manual_seed(42)
        x_m = torch.randn(728, self.args.fm)
        x_d = torch.randn(884, self.args.fd)

        # mi_gua_1 = self.gcn1(emb, sim_set['miRNA_mut']['mi_gua'].to(device))
        # mi_cos_1 = self.gcn1(emb, sim_set['miRNA_mut']['mi_cos'].to(device))
        # mi_fun_1 = self.gcn1(emb, sim_set['miRNA_mut']['mi_fun'].to(device))
        #
        # di_gua_1 = self.gcn2(emb.T, sim_set['disease_mut']['di_gua'].to(device))
        # di_cos_1 = self.gcn2(emb.T, sim_set['disease_mut']['di_cos'].to(device))
        # di_sem_1 = self.gcn2(emb.T, sim_set['disease_mut']['di_sem'].to(device))

        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.to(device), data['miRNA_mut']['mi_gua_edges'].to(device), data['miRNA_mut']['mi_gua'][data['miRNA_mut']['mi_gua_edges'][0], data['miRNA_mut']['mi_gua_edges'][1]].to(device)))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['miRNA_mut']['mi_gua_edges'].to(device), data['miRNA_mut']['mi_gua'][data['miRNA_mut']['mi_gua_edges'][0], data['miRNA_mut']['mi_gua_edges'][1]].to(device)))

        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.to(device), data['miRNA_mut']['mi_cos_edges'].to(device), data['miRNA_mut']['mi_cos'][data['miRNA_mut']['mi_cos_edges'][0], data['miRNA_mut']['mi_cos_edges'][1]].to(device)))
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['miRNA_mut']['mi_cos_edges'].to(device), data['miRNA_mut']['mi_cos'][data['miRNA_mut']['mi_cos_edges'][0], data['miRNA_mut']['mi_cos_edges'][1]].to(device)))

        x_m_g1 = torch.relu(self.gcn_x1_s(x_m.to(device), data['miRNA_mut']['mi_fun_edges'].to(device),data['miRNA_mut']['mi_fun'][data['miRNA_mut']['mi_fun_edges'][0], data['miRNA_mut']['mi_fun_edges'][1]].to(device)))
        x_m_g2 = torch.relu(self.gcn_x2_s(x_m_s1, data['miRNA_mut']['mi_fun_edges'].to(device), data['miRNA_mut']['mi_fun'][data['miRNA_mut']['mi_fun_edges'][0], data['miRNA_mut']['mi_fun_edges'][1]].to(device)))

        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.to(device), data['disease_mut']['di_gua_edges'].to(device), data['disease_mut']['di_gua'][data['disease_mut']['di_gua_edges'][0], data['disease_mut']['di_gua_edges'][1]].to(device)))
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['disease_mut']['di_gua_edges'].to(device), data['disease_mut']['di_gua'][data['disease_mut']['di_gua_edges'][0], data['disease_mut']['di_gua_edges'][1]].to(device)))

        y_d_s1 = torch.relu(self.gcn_y1_s(x_d.to(device), data['disease_mut']['di_cos_edges'].to(device), data['disease_mut']['di_cos'][data['disease_mut']['di_cos_edges'][0], data['disease_mut']['di_cos_edges'][1]].to(device)))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['disease_mut']['di_cos_edges'].to(device), data['disease_mut']['di_cos'][data['disease_mut']['di_cos_edges'][0], data['disease_mut']['di_cos_edges'][1]].to(device)))

        y_d_g1 = torch.relu(self.gcn_y1_s(x_d.to(device), data['disease_mut']['di_sem_edges'].to(device),data['disease_mut']['di_sem'][data['disease_mut']['di_sem_edges'][0],data['disease_mut']['di_sem_edges'][1]].to(device)))
        y_d_g2 = torch.relu(self.gcn_y2_s(y_d_s1, data['disease_mut']['di_sem_edges'].to(device),data['disease_mut']['di_sem'][data['disease_mut']['di_sem_edges'][0],data['disease_mut']['di_sem_edges'][1]].to(device)))

        XM = torch.cat((x_m_f1, x_m_f2, x_m_s1, x_m_s2, x_m_g1, x_m_g2), 1).t()
        # XM = torch.cat((mi_gua_1, mi_cos_1, mi_fun_1), 1).t()
        # XM = XM.view(1, 6, self.args.fm, -1)
        XM = XM.view(1, 6, self.args.fm, -1)

        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)

        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)

        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM

        XM_channel_attention = torch.relu(XM_channel_attention)

        YD = torch.cat((y_d_f1, y_d_f2, y_d_s1, y_d_s2, y_d_g1, y_d_g2), 1).t()
        # YD = torch.cat((di_gua_1, di_cos_1, di_sem_1), 1).t()

        # YD = YD.view(1, 6, self.args.fd, -1)
        YD = YD.view(1, 6, self.args.fd, -1)

        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD

        YD_channel_attention = torch.relu(YD_channel_attention)



        x = self.cnn_x(XM_channel_attention)
        x = x.view(128, self.args.miRNA_number).t()

        y = self.cnn_y(YD_channel_attention)
        y = y.view(128, self.args.disease_number).t()

        train_edges = train_edges.T
        m_index = train_edges[0]
        d_index = train_edges[1]
        Em = torch.index_select(x, 0, m_index)
        Ed = torch.index_select(y, 0, d_index)
        pair_feat = Em * Ed
        out = self.linear(pair_feat)
        return torch.sigmoid(out).view(-1)











