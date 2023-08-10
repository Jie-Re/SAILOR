import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, VGAE

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, with_bias, device):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, dropout=dropout, with_bias=with_bias).to(device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VariationalGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, with_bias, device):
        super(VariationalGATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, 1, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_mu = GATConv(2 * out_channels, out_channels, 1, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_logstd = GATConv(2 * out_channels, out_channels, 1, dropout=dropout, with_bias=with_bias).to(device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VariationalSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, with_bias, device):
        super(VariationalSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels, dropout=dropout, with_bias=with_bias).to(device)
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels, dropout=dropout, with_bias=with_bias).to(device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class MYVGAE(nn.Module):
    def __init__(self, config, nnodes, nfeat,  nclass, device=None):

        super(MYVGAE, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nnodes = nnodes

        if config['arch'] == 1:
            self.vgae = VGAE(VariationalGCNEncoder(self.nfeat, self.nclass, config['dropout'], config['with_bias'], self.device))
        if config['arch'] == 2:
          self.vgae = VGAE(VariationalGATEncoder(self.nfeat, self.nclass, config['dropout'], config['with_bias'], self.device))
        if config['arch'] == 3:
            self.vgae = VGAE(VariationalSAGEEncoder(self.nfeat, self.nclass, config['dropout'], config['with_bias'], self.device))

    def forward(self, x, edge_index):
        z = self.vgae.encode(x, edge_index)
        loss = self.vgae.recon_loss(z, edge_index)
        loss = loss + (1 / self.nnodes) * self.vgae.kl_loss()
        return z, loss
        
  
    def reset_parameters(self):
        self.vgae.reset_parameters()