import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge, BatchNorm, LayerNorm
from utils import _act_dict
from model.norm_tricks import mean_norm, node_norm, pair_norm

class GNN(nn.Module):
    def __init__(self, config, nfeat, nhids, acts, nheads, nclass):

        super(GNN, self).__init__()

        self.arch = config['arch']
        self.device = config['device']
        self.nfeat = nfeat
        self.nclass = nclass
        with_bias = config['with_bias']
        dropout = config['dropout']

        self.layers = torch.nn.ModuleList()
        if self.arch == 1:
            for nhid, act in zip(nhids, acts):
                self.layers.append(GCNConv(nfeat, nhid, cached=False, 
                    add_self_loops=True, normalize=True, bias=with_bias).to(self.device))
                if config['gnn']['norm_type'] == 'batchNorm':
                    self.layers.append(BatchNorm(nhid))
                if config['gnn']['norm_type'] ==  'meanNorm':
                    self.layers.append(mean_norm())
                if config['gnn']['norm_type'] == 'nodeNorm':
                    self.layers.append(node_norm())
                if config['gnn']['norm_type'] == 'layerNorm':
                    self.layers.append(LayerNorm(nhid))
                if config['gnn']['norm_type'] == 'pairNorm':
                    self.layers.append(pair_norm())
                self.layers.append(getattr(torch.nn, _act_dict.get(act, None))().to(self.device))
                self.layers.append(torch.nn.Dropout(dropout).to(self.device))
                nfeat = nhid
            self.layers.append(GCNConv(nfeat, nclass, cached=False, add_self_loops=True, normalize=True, bias=with_bias).to(self.device))
        elif self.arch == 2:
            # input layer
            self.layers.append(GATConv(nfeat, nhids[0], nheads[0], dropout=dropout, bias=with_bias).to(self.device))
            if config['gnn']['norm_type'] == 'batchNorm':
                self.layers.append(BatchNorm(nhids[0]*nheads[0]))
            if config['gnn']['norm_type'] ==  'meanNorm':
                self.layers.append(mean_norm())
            if config['gnn']['norm_type'] == 'nodeNorm':
                self.layers.append(node_norm())
            if config['gnn']['norm_type'] == 'layerNorm':
                self.layers.append(LayerNorm(nhids[0]*nheads[0]))
            if config['gnn']['norm_type'] == 'pairNorm':
                self.layers.append(pair_norm())
            self.layers.append(getattr(torch.nn, _act_dict.get(acts[0], None))().to(self.device))
            self.layers.append(torch.nn.Dropout(dropout).to(self.device))
            # hidden layers
            for i in range(1, len(nhids)-1):
                self.layers.append(GATConv(nhids[i-1]*nheads[i-1], nhids[i], nheads[i], dropout=dropout, with_bias=with_bias).to(self.device))
                if config['gnn']['norm_type'] == 'batchNorm':
                    self.layers.append(BatchNorm(nhids[i]))
                if config['gnn']['norm_type'] ==  'meanNorm':
                    self.layers.append(mean_norm())
                if config['gnn']['norm_type'] == 'nodeNorm':
                    self.layers.append(node_norm())
                if config['gnn']['norm_type'] == 'layerNorm':
                    self.layers.append(LayerNorm(nhids[i]))
                if config['gnn']['norm_type'] == 'pairNorm':
                    self.layers.append(pair_norm())
                self.layers.append(getattr(torch.nn, _act_dict.get(acts[i], None))().to(self.device))
                self.layers.append(torch.nn.Dropout(dropout).to(self.device))
            # output layer
            self.layers.append(GATConv(nhids[-1]*nheads[-2], nclass, nheads[-1], dropout=dropout, bias=with_bias).to(self.device))
        elif self.arch == 3:
            self.layers = torch.nn.ModuleList()
            for nhid, act in zip(nhids, acts):
                self.layers.append(SAGEConv(nfeat, nhid, bias=with_bias).to(self.device))
                if config['gnn']['norm_type'] == 'batchNorm':
                    self.layers.append(BatchNorm(nhid))
                if config['gnn']['norm_type'] ==  'meanNorm':
                    self.layers.append(mean_norm())
                if config['gnn']['norm_type'] == 'nodeNorm':
                    self.layers.append(node_norm())
                if config['gnn']['norm_type'] == 'layerNorm':
                    self.layers.append(LayerNorm(nhid))
                if config['gnn']['norm_type'] == 'pairNorm':
                    self.layers.append(pair_norm())
                self.layers.append(getattr(torch.nn, _act_dict.get(act, None))().to(self.device))
                self.layers.append(torch.nn.Dropout(dropout).to(self.device))
                nfeat = nhid
            self.layers.append(SAGEConv(nfeat, nclass, bias=with_bias).to(self.device))
        elif self.arch == 4:
            self.layers = torch.nn.ModuleList()
            for nhid, act in zip(nhids, acts):
                self.layers.append(GCNConv(nfeat, nhid, cached=False, 
                    add_self_loops=True, normalize=True, bias=with_bias).to(self.device))
                if config['gnn']['norm_type'] == 'batchNorm':
                    self.layers.append(BatchNorm(nhid))
                if config['gnn']['norm_type'] ==  'meanNorm':
                    self.layers.append(mean_norm())
                if config['gnn']['norm_type'] == 'nodeNorm':
                    self.layers.append(node_norm())
                if config['gnn']['norm_type'] == 'layerNorm':
                    self.layers.append(LayerNorm(nhid))
                if config['gnn']['norm_type'] == 'pairNorm':
                    self.layers.append(pair_norm())
                self.layers.append(getattr(torch.nn, _act_dict.get(act, None))().to(self.device))
                self.layers.append(torch.nn.Dropout(dropout).to(self.device))
                nfeat = nhid
            self.layers.append(JumpingKnowledge(mode='max').to(self.device))
            self.layers.append(torch.nn.Linear(nfeat, nclass).to(self.device)) # mode='max'

    def forward(self, x, edge_index, edge_attr=None):
        if self.arch == 4: # jknet
            layer_out = []
        for layer in self.layers:
            if isinstance(layer, GCNConv) or isinstance(layer, GATConv) or isinstance(layer, SAGEConv):
                x = layer(x, edge_index, edge_attr)
            elif self.arch == 4 and isinstance(layer, torch.nn.Dropout):
                x = layer(x)
                layer_out.append(x)
            elif self.arch == 4 and isinstance(layer, JumpingKnowledge):
                x = layer(layer_out)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()