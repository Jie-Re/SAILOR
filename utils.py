import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
import random
import os
import yaml
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
from torch_sparse import SparseTensor

_arch_dict = {1:'gcn', 2:'gat', 3:'graphSage'}

_act_dict = dict(relu="ReLU",
                relu6="ReLU6",
                sigmoid="Sigmoid",
                celu="CELU",
                elu="ELU",
                gelu="GELU",
                leakyrelu="LeakyReLU",
                prelu="PReLU",
                selu="SELU",
                silu="SiLU",
                softmax="Softmax",
                tanh="Tanh")


def get_config(args):
    with open(args.config, 'r') as c:
        config = yaml.safe_load(c)
    argsDict = args.__dict__
    for k, v in argsDict.items():
        if v != None:
            if 'gnn' in k:
                config['gnn'][k[4:]] = v
            elif 'augmentor' in k:
                config['augmentor'][k[9:]] = v
            else:
                config[k] = v
    config['iscuda'] = torch.cuda.is_available() and config['with_gpu']
    return config


def setup_seed(seed, is_cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def bestGPU(gpu_verbose=False, **w):
    import GPUtil
    import numpy as np

    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil * 100
        load = gpu.load * 100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')

    return int(best)

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def adj2edgeIndex(adj, not_sparseTensor=False):
    if isinstance(adj, sp.csr_matrix):
        adj = adj.tocoo()
    if isinstance(adj, sp.coo_matrix):
        return torch.LongTensor(np.array([adj.row,adj.col]))
    if not adj.is_sparse:
        adj = adj.to_sparse()
    if not_sparseTensor:
        return adj._indices()
    return SparseTensor.from_edge_index(adj._indices(), sparse_sizes=(adj.shape[0], adj.shape[1])).to(adj.device)
    # return adj._indices()