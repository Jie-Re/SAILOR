import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from utils import _act_dict
from torch.autograd import Variable

MAX_LOGSTD = 10

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, nhid, out_channels, cached, device):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, nhid, cached=cached).to(device) # cached only for transductive learning
        self.conv_mu = GCNConv(nhid, out_channels, cached=cached).to(device)
        self.conv_logstd = GCNConv(nhid, out_channels, cached=cached).to(device)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_weight=edge_attr).relu()
        return self.conv_mu(x, edge_index, edge_weight=edge_attr), self.conv_logstd(x, edge_index, edge_weight=edge_attr)


class Augmentor(nn.Module):
    def __init__(self, config, nnodes, nfeat, nclass):

        super(Augmentor, self).__init__()

        self.device = config['device']
        self.temperature = config['temperature']

        self.vgae = VGAE(VariationalGCNEncoder(nfeat, config['augmentor']['hidden'], nclass, True, self.device))
        self.epsilon = torch.nn.Parameter(torch.Tensor(nnodes))
        self.reset_parameters()


    def forward(self, x, edge_index, target_edge_index):
        p = self.vgae.encode(x, edge_index)
        s = torch.sigmoid((p[target_edge_index[0]] * p[target_edge_index[1]]).sum(dim=1))
        return p, s

    def fuse_encode(self, x, edge_index):
        p_g = self.vgae.encode(x, edge_index)

        b, w = self.vgae.encoder.conv1.parameters()
        b_mu, w_mu = self.vgae.encoder.conv_mu.parameters()
        b_sigma, w_sigma = self.vgae.encoder.conv_logstd.parameters()
    
        p_l = x@w.T+b
        mu = p_l@w_mu.T+b_mu
        sigma = p_l@w_sigma.T+b_sigma
        sigma = sigma.clamp(max=MAX_LOGSTD)
        p_l = self.vgae.reparametrize(mu, sigma)

        p = (1-self.epsilon.view(-1,1))*p_g + self.epsilon.view(-1,1)*p_l
        return p

    def reparametrize_n(self, mu, std):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    
    def augment(self, x, edge_index, tail_nodes):
        batch_size = 4096
        p = self.fuse_encode(x, edge_index)
        i = 0
        while i < tail_nodes.shape[0]:
            tail_batch_nodes = tail_nodes[i:i+batch_size]
            i = i+batch_size
            transP = self.compute_transP(p, tail_batch_nodes)
            transP_dis = torch.distributions.bernoulli.Bernoulli(probs=transP).sample()
            add_edges = torch.nonzero(transP_dis, as_tuple=True)
            row = torch.tensor([tail_batch_nodes[j].item() for j in add_edges[0]], dtype=torch.long).to(add_edges[0].device)
            col = add_edges[1]
            edge_index_to_add = torch.stack([row, col],dim=0)#.to(x.device)
            edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
        return edge_index
    

    def reset_parameters(self):
        self.vgae.reset_parameters()
        torch.nn.init.constant_(self.epsilon, 1.0)
        # self.epsilon.fill_(1.0)

    def compute_transP(self, cd, tail):
        """

        :param cd: class distribution [N, D]
        :param tail: tail node index
        :return: transition probability of tail to all the other nodes in the graph (transP) [N]
        """

        p_i = cd[tail]
        p_j = cd
        # Transition Probability
        pipj = (p_i @ p_j.T)  # [N]
        transP = F.softmax(pipj, dim=-1)
        return transP
