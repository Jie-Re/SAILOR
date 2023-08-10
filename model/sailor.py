import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gnnmodels import GNN
from model.augmentor import Augmentor


class SAILOR(nn.Module):
    def __init__(self, config, nnodes, nfeat, nclass) -> None:
        super(SAILOR, self).__init__()
        self.device = config['device']
        self.arch = config['arch']

        # Define GNN Model
        self.gnn_n_layers = config['gnn']['n_layers']
        nhids, acts=[], []
        for _ in range(config['gnn']['n_layers']):
            nhids.append(config['gnn']['hidden'])
            acts.append(config['activation'])
        nheads = None
        if config['arch'] == 2:
            nheads = [8] * config['gnn']['n_layers'] + [1]
        self.gnn = GNN(config, nfeat, nhids, acts, nheads, nclass)
        
        self.augmentor = Augmentor(config, nnodes, nfeat, nclass)

        self.pth_to_models = f"{config['outpth']}/saved_models"
        if not os.path.exists(self.pth_to_models):
            os.makedirs(self.pth_to_models)


    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.augmentor.reset_parameters()
