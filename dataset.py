import numpy as np
import torch
from torch_geometric.utils import degree, to_scipy_sparse_matrix, to_undirected, index_to_mask, remove_self_loops, remove_isolated_nodes
from sklearn import model_selection
import scipy.sparse as sp
import random
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from utils import adj2edgeIndex

DATA_PATH = '/home/jliao/public_data/pyg_data'

def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A - sp.diags(A.diagonal(), format='csr')
    A.eliminate_zeros()
    return A

def largest_connected_components(A):
    _, component_indices = sp.csgraph.connected_components(A)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[-1]
    nodes_to_keep = np.where(component_indices == components_to_keep)[0]
    return nodes_to_keep

def get_drop_graph(edge_index, head_nodes, num_nodes, drop_prt):
    row = torch.LongTensor([]).to(edge_index.device)
    col = torch.LongTensor([]).to(edge_index.device)
    adj = to_scipy_sparse_matrix(edge_index)
    adj = torch.sparse_coo_tensor(edge_index, adj.data, adj.shape, 
                                    dtype=torch.long, device=edge_index.device)
    for u in range(num_nodes):
        N = adj[u]._indices()[0].shape[0]
        if u in head_nodes:
            new_N = int(N*drop_prt)
            rand_idx = torch.LongTensor(random.sample(range(N), new_N)).to(edge_index.device)
            row = torch.cat([row, torch.LongTensor([u]*new_N).to(edge_index.device)])
            col = torch.cat([col, torch.index_select(adj[u]._indices()[0],0,rand_idx)])
        else:
            row = torch.cat([row, torch.LongTensor([u]*N).to(edge_index.device)])
            col = torch.cat([col, adj[u]._indices()[0]])
    drop_edge_index = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
    return drop_edge_index


def get_dataset(name: str, use_lcc: bool, use_undirected: bool, seed: int) -> InMemoryDataset:
    # path = os.path.join(DATA_PATH, name)
    path = DATA_PATH
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif name == 'CoauthorPhy':
        dataset = Coauthor(path, 'Physics')
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name, path)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(path, name.lower())
    elif name == 'Actor':
        dataset = Actor(path+'/'+name)
    else:
        raise Exception('Unknown dataset.')

    if name in ['Cora', 'Citeseer', 'Pubmed']:
        train_mask = dataset.data.train_mask
        val_mask = dataset.data.val_mask
        test_mask = dataset.data.test_mask
    elif name in ['ogbn-arxiv']:
        split_idx = dataset.get_idx_split()
        train_mask = index_to_mask(split_idx['train'], size=dataset.data.y.squeeze().shape[0])
        val_mask = index_to_mask(split_idx['valid'], size=dataset.data.y.squeeze().shape[0])
        test_mask = index_to_mask(split_idx['test'], size=dataset.data.y.squeeze().shape[0])
    elif name in ['CoauthorCS', 'CoauthorPhy', 'Computers', 'Photo']:
        # train_mask = torch.zeros(dataset.data.y.squeeze().shape[0], dtype=torch.bool)
        # val_mask = torch.zeros(dataset.data.y.squeeze().shape[0], dtype=torch.bool)
        # test_mask = torch.zeros(dataset.data.y.squeeze().shape[0], dtype=torch.bool)
        train_mask ,val_mask, test_mask = set_standard_train_val_test_split(seed, dataset.data.y.squeeze())
    else:
        train_mask ,val_mask, test_mask = set_standard_train_val_test_split2(seed, dataset.data.y.squeeze())

    if use_undirected:
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    if use_lcc:
        adj = to_scipy_sparse_matrix(dataset.data.edge_index)
        adj = eliminate_self_loops(adj)
        nodes_to_keep = largest_connected_components(adj)
        adj = adj[nodes_to_keep][:, nodes_to_keep]
        edge_index_new = adj2edgeIndex(adj)
        x_new = dataset.data.x[nodes_to_keep]
        y_new = dataset.data.y[nodes_to_keep]

        train_mask = train_mask[nodes_to_keep]
        val_mask = val_mask[nodes_to_keep]
        test_mask = test_mask[nodes_to_keep]

        data = Data(
            x=x_new,
            # edge_index=torch.LongTensor(edges),
            edge_index=edge_index_new,
            y=y_new.squeeze(),
            num_nodes=y_new.shape[0],
            num_feats=x_new.shape[1],
            num_classes=y_new.max().item()-y_new.min().item()+1,
            train_mask=train_mask,
            test_mask=val_mask,
            val_mask=test_mask
        )
        dataset.data = data
    else:
        data = Data(
            x=dataset.data.x,
            edge_index=dataset.data.edge_index,
            y=dataset.data.y.squeeze(),
            num_nodes=dataset.data.y.squeeze().shape[0],
            num_feats=dataset.data.x.shape[1],
            num_classes=dataset.data.y.squeeze().max().item()-dataset.data.y.squeeze().min().item()+1,
            train_mask=train_mask,
            test_mask=val_mask,
            val_mask=test_mask
        )
        dataset.data = data

    return dataset.data


class TailDataset(InMemoryDataset):
    def __init__(self, name, seed, degthres, use_lcc, use_undirected, verbose):
        self.data = get_dataset(name, use_lcc, use_undirected, seed)
        degrees, mindeg, maxdeg, degthres, tail_mask, head_mask  = self.set_head_tail_split(self.data, degthres=degthres, verbose=verbose)
        self.data.degrees = degrees
        self.data.mindeg = mindeg
        self.data.maxdeg = maxdeg
        self.data.degthres = degthres
        self.data.tail_mask = tail_mask
        self.data.head_mask = head_mask
        self.seed = seed

    def set_head_tail_split(self, data: Data, degthres: int=0, verbose: bool=True):
        degrees = degree(to_undirected(data.edge_index)[0])
        mindeg = int(degrees.min().item())
        maxdeg = int(degrees.max().item())
        # imbalance_rt = degbin[maxdeg] / degbin[mindeg]
        if degthres==0:
            # binning nodes according to their degree
            degbin = {}
            for i in degrees:
                if i.item() not in degbin.keys():
                    degbin[i.item()] = 0
                degbin[i.item()] += 1

            # count degbin ascendingly
            cntdeg = []
            degthres = 0
            tmp = 0
            for i in range(mindeg, maxdeg+1):
                if i in degbin.keys():
                    tmp += degbin[i]
                cntdeg.append(tmp)

                if tmp >= 0.8 * data.y.shape[0] and degthres == 0:
                    degthres = i
            assert degthres != 0

        # get tail nodes and head nodes
        tail_idx = torch.nonzero(degrees <= degthres, as_tuple=True)[0]
        head_idx = torch.nonzero(degrees > degthres, as_tuple=True)[0]

        # get mask
        def get_mask(idx):
            mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
            mask[idx] = 1
            return mask

        tail_mask = get_mask(tail_idx)
        head_mask = get_mask(head_idx)

        if verbose:
            print(f'minimum degree is {mindeg}')
            print(f'maximum degree is {maxdeg}')
            print(f"degree theshold is {degthres}")
            # print(f"imbalance ratio is {imbalance_rt}")
            print(f'tail nodes amount: {tail_idx.shape[0]}')
            print(f'head nodes amount: {head_idx.shape[0]}')
        return degrees, mindeg, maxdeg, degthres, tail_mask, head_mask


def set_standard_train_val_test_split(
    seed: int,
    labels: torch.tensor):
    num_nodes = labels.shape[0]
    train_idx, not_train_idx = model_selection.train_test_split(
                                np.arange(num_nodes), 
                                test_size = 0.9, 
                                random_state=seed, 
                                stratify=labels)
    val_idx, test_idx = model_selection.train_test_split(not_train_idx, 
                                train_size=1/9,  
                                random_state=seed, 
                                stratify=labels[not_train_idx])
    
    def get_mask(idx):
        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        mask[idx] = 1
        return mask

    train_mask = get_mask(train_idx)
    val_mask = get_mask(val_idx)
    test_mask =get_mask(test_idx)

    return train_mask, val_mask, test_mask

def set_standard_train_val_test_split2(
    seed: int,
    labels: torch.tensor):
    num_nodes = labels.shape[0]
    train_idx, not_train_idx = model_selection.train_test_split(
                                np.arange(num_nodes), 
                                train_size = 0.6,
                                test_size = 0.4, 
                                random_state=seed, 
                                stratify=labels)
    val_idx, test_idx = model_selection.train_test_split(not_train_idx, 
                                train_size=0.5, 
                                test_size=0.5, 
                                random_state=seed, 
                                stratify=labels[not_train_idx])
    
    def get_mask(idx):
        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        mask[idx] = 1
        return mask

    train_mask = get_mask(train_idx)
    val_mask = get_mask(val_idx)
    test_mask =get_mask(test_idx)

    return train_mask, val_mask, test_mask


def set_tail_train_val_test_split(
    seed: int,
    data: Data) -> Data:
    
    num_nodes = data.y.shape[0]
    train_idx = np.arange(num_nodes)[data.head_mask]
    val_idx, test_idx = model_selection.train_test_split(
                                            np.arange(num_nodes)[data.tail_mask], 
                                            train_size=0.2, 
                                            test_size=0.8, 
                                            random_state=seed, 
                                            stratify=data.y[data.tail_mask])
    
    # get mask
    def get_mask(idx):
        mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask =get_mask(test_idx)

    return data