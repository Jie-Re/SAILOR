import numpy as np
import torch
import torch.nn.functional as F
import os
from model.sailor import SAILOR
from dataset import TailDataset, get_drop_graph, set_tail_train_val_test_split
from utils import setup_seed, bestGPU, get_config, accuracy
from args import get_args
from torch_geometric.data import Data
from torch.optim import Optimizer
from copy import deepcopy
from tqdm import tqdm
import scipy as sp
from torch_scatter import scatter
from sklearn.metrics import f1_score

EPS = 1e-15
MAX_LOGSTD = 10

bce_loss = torch.nn.BCEWithLogitsLoss()
kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)

import matplotlib.pyplot as plt
def plot(config, loss_gnn, loss_vgae, loss_lp, loss_kl):
    iters = range(len(loss_gnn))
    plt.plot(iters, loss_gnn, color='red', label='loss_gnn')
    plt.plot(iters, loss_vgae, color='blue',label='loss_vgae')
    plt.plot(iters, loss_lp, color='green', label='loss_lp')
    plt.plot(iters, loss_kl, color='magenta', label='loss_kl')
    plt.legend()
    plt.savefig(f"{config['outpth']}/train_loss.png")
    plt.close()

def propagate(edge_index, x, edge_attr=None, k=1):
    for _ in range(k):
        if edge_attr is None:
            msg = x[edge_index[1]]
        else:
            msg = x[edge_index[1]]*edge_attr.view(-1,1)
        x = scatter(msg, edge_index[0], dim=-2, reduce='sum')
    return x

def train(model: torch.nn.Module, 
          gnn_optimizer: Optimizer, 
          augmentor_optimizer: Optimizer,
          data: Data):

    # update gnn
    model.augmentor.eval()
    model.gnn.train()
    aug_edge_index = model.augmentor.augment(data.x, data.edge_index, torch.nonzero(data.tail_mask, as_tuple=True)[0])
    z1 = model.gnn(data.x, aug_edge_index)
    loss_aug = F.cross_entropy(z1[data.train_mask], data.y[data.train_mask])
    loss_gnn = config['alpha']*loss_aug 
    loss_gnn.backward()
    gnn_optimizer.step()
    gnn_optimizer.zero_grad()

    # update augmentor
    model.augmentor.train()
    model.gnn.eval()
    p1, s = model.augmentor(data.x, data.drop_edge_index, target_edge_index = data.edge_index)
    loss_vgae_pos = -torch.log(s+EPS).mean()
    loss_vgae = loss_vgae_pos + (1 / data.x.shape[0]) * model.augmentor.vgae.kl_loss()
    
    p = model.augmentor.fuse_encode(data.x, data.edge_index)
    loss_lp = F.cross_entropy(p[data.train_mask], data.y[data.train_mask])

    z2 = model.augmentor.vgae.encode(data.x, aug_edge_index)
    z2 = F.softmax(z2, dim=-1)
    z1 = F.softmax(z1.detach(),dim=-1)
    loss_kl =  kl_loss(z2, z1)

    loss_augmentor = config['beta']*loss_vgae+config['eta']*loss_lp + config['theta']*loss_kl
    loss_augmentor.backward()
    augmentor_optimizer.step()
    augmentor_optimizer.zero_grad()

    if config['verbose']:
        print(f"loss_gnn {loss_gnn}, loss_vgae {loss_vgae}, loss_lp {loss_lp}")
        acc_gnn = accuracy(z1[data.train_mask], data.y[data.train_mask]).item()
        print(f"acc_gnn {acc_gnn}")
    return loss_gnn.item(), loss_vgae.item(), loss_lp.item(), loss_kl.item()
    

@torch.no_grad()
def evaluate(model: torch.nn.Module, data: Data, config):
    model.gnn.eval()
    eval_dict = {}
    output = model.gnn(data.x, data.edge_index)
    acc_gnn = accuracy(output[data.val_mask], data.y[data.val_mask])
    eval_dict['val_acc_gnn'] = acc_gnn.item()

    if config['evaluate_mode'] == 0:
        aug_edge_index = model.augmentor.augment(data.x, data.edge_index, torch.nonzero(data.tail_mask, as_tuple=True)[0])
        z1 = model.gnn(data.x, aug_edge_index)
        loss_aug = F.cross_entropy(z1[data.train_mask], data.y[data.train_mask])
        loss_gnn = config['alpha']*loss_aug 

        p1, s = model.augmentor(data.x, data.drop_edge_index, target_edge_index = data.edge_index)
        loss_vgae_pos = -torch.log(s+EPS).mean() # P.S. 不计算neg_loss
        loss_vgae = loss_vgae_pos + (1 / data.x.shape[0]) * model.augmentor.vgae.kl_loss()
        
        p = model.augmentor.fuse_encode(data.x, data.edge_index)
        loss_lp = F.cross_entropy(p[data.train_mask], data.y[data.train_mask])

        z2 = model.augmentor.vgae.encode(data.x, aug_edge_index)
        loss_kl = kl_loss(z1.detach(), z2) + kl_loss(z2, z1.detach())

        loss_augmentor = config['beta']*loss_vgae+config['eta']*loss_lp + config['theta']*loss_kl
                    
        eval_dict['val_loss_augmentor'] = loss_augmentor.item()
        eval_dict['val_loss_gnn'] = loss_gnn.item()

    return eval_dict


def test1(model: torch.nn.Module, data: Data):
    model.gnn.eval()
    model.gnn.load_state_dict(torch.load(f"{config['outpth']}/saved_models/gnn.pt"))
    
    output = model.gnn(data.x, data.edge_index)
    result = {}
    keys = ['train', 'val', 'test']
    for k in keys:
        mask = data.__getattr__(f'{k}_mask')
        result[f'loss_{k}'] = F.cross_entropy(output[mask], data.y[mask]).item()
        result[f'acc_{k}'] = accuracy(output[mask], data.y[mask]).item()
        result[f'weighted_f1_{k}'] = f1_score(y_true=data.y[mask].cpu().numpy(), 
                                              y_pred=output[mask].argmax(dim=-1).cpu().numpy(),
                                              average='weighted')
    return result
    

def test2(model: torch.nn.Module, data: Data):
    model.gnn.eval()
    model.gnn.load_state_dict(torch.load(f"{config['outpth']}/saved_models/gnn.pt"))
    
    output = model.gnn(data.x, data.edge_index)
    result = {}
    keys = ['test']
    keys2 = ['head', 'tail']
    for k in keys:
        mask = data.__getattr__(f'{k}_mask')
        result[f'loss_{k}'] = F.cross_entropy(output[mask], data.y[mask]).item()
        result[f'acc_{k}'] = accuracy(output[mask], data.y[mask]).item()
        result[f'weighted_f1_{k}'] = f1_score(y_true=data.y[mask].cpu().numpy(), 
                                              y_pred=output[mask].argmax(dim=-1).cpu().numpy(),
                                              average='weighted')
        for k2 in keys2:
            mask = data.__getattr__(f'{k}_mask')*data.__getattr__(f'{k2}_mask')
            result[f'loss_{k}:{k2}'] = F.cross_entropy(output[mask], data.y[mask]).item()
            result[f'acc_{k}:{k2}'] = accuracy(output[mask], data.y[mask]).item()
            result[f'weighted_f1_{k}:{k2}'] = f1_score(y_true=data.y[mask].cpu().numpy(), 
                                              y_pred=output[mask].argmax(dim=-1).cpu().numpy(),
                                              average='weighted')
    return result


def run(data: Data, config):
    model = define_model(dataset.data, config)
    model.to(config['device']).reset_parameters()

    if config['gnn']['optimizer'] == 'Adam':
        gnn_optimizer = torch.optim.Adam(
            model.gnn.parameters(), lr=config['gnn']['lr'], weight_decay=config['gnn']['weight_decay'])
    if config['gnn']['optimizer'] == 'AdamW':
        gnn_optimizer = torch.optim.AdamW(model.gnn.parameters(), lr=config['gnn']['lr'])
    if config['gnn']['optimizer'] == 'NAdam':
        gnn_optimizer = torch.optim.NAdam(model.gnn.parameters(), lr=config['gnn']['lr'])
    if config['gnn']['optimizer'] == 'RAdam':
        gnn_optimizer = torch.optim.RAdam(model.gnn.parameters(), lr=config['gnn']['lr'])
    if config['gnn']['optimizer'] == 'AMSGrad':
        gnn_optimizer = torch.optim.Adam(
            model.gnn.parameters(), lr=config['gnn']['lr'], weight_decay=config['gnn']['weight_decay'], amsgrad=True)
    
    if config['augmentor']['optimizer'] == 'Adam':
        augmentor_optimizer = torch.optim.Adam(
            model.augmentor.parameters(), lr=config['augmentor']['lr'], weight_decay=config['augmentor']['weight_decay'])
    if config['augmentor']['optimizer'] == 'AdamW':
        augmentor_optimizer = torch.optim.AdamW(model.augmentor.parameters(), lr=config['augmentor']['lr'])
    if config['augmentor']['optimizer'] == 'NAdam':
        augmentor_optimizer = torch.optim.NAdam(model.augmentor.parameters(), lr=config['augmentor']['lr'])
    if config['augmentor']['optimizer'] == 'RAdam':
        augmentor_optimizer = torch.optim.RAdam(model.augmentor.parameters(), lr=config['augmentor']['lr'])
    if config['augmentor']['optimizer'] == 'AMSGrad':
        augmentor_optimizer = torch.optim.Adam(
            model.augmentor.parameters(), lr=config['augmentor']['lr'], weight_decay=config['augmentor']['weight_decay'], 
            amsgrad=True)
    
    patience_counter = 0
    if config['evaluate_mode']:
        tmp_dict = {'best_acc_gnn': 0}
    else:
        tmp_dict = {'val_loss_augmentor': 10000, 'val_loss_gnn':10000}

    if config['visualize']:
        loss_gnn_list, loss_vgae_list, loss_lp_list, loss_kl_list = [],[],[],[]
    for epoch in tqdm(range(config['epoch']), desc='Training'):
        if patience_counter == config['patience']:
            break
        loss_gnn, loss_vgae, loss_lp, loss_kl = train(model, gnn_optimizer, augmentor_optimizer, data)
        if config['visualize']:
            loss_gnn_list.append(loss_gnn)
            loss_vgae_list.append(loss_vgae)
            loss_lp_list.append(loss_lp)
            loss_kl_list.append(loss_kl)
            
        eval_dict = evaluate(model, data, config)
        if config['evaluate_mode']:
            condition = (eval_dict['val_acc_gnn'] <= tmp_dict['best_acc_gnn'])
        else:
            condition =  (eval_dict['val_loss_gnn'] >= tmp_dict['val_loss_gnn'])
        if condition:
            patience_counter += 1
        else:
            if config['evaluate_mode'] == 0:
                tmp_dict['val_loss_gnn'] = eval_dict['val_loss_gnn']
                tmp_dict['val_loss_augmentor'] = eval_dict['val_loss_augmentor']
            tmp_dict['best_acc_gnn'] = eval_dict['val_acc_gnn']
            torch.save(model.augmentor.state_dict(),
                    f"{config['outpth']}/saved_models/augmentor.pt")
            torch.save(model.gnn.state_dict(),
                    f"{config['outpth']}/saved_models/gnn.pt")
            if config['verbose']:
                print(f"GNN updated! val_acc_gnn {eval_dict['val_acc_gnn']}")
                print('Augmentor updated!')
            patience_counter = 0
            tmp_dict['epoch'] = epoch
    
    print(tmp_dict)
    if config['visualize']:
        plot(config, loss_gnn_list, loss_vgae_list, loss_lp_list, loss_kl_list)
    if 'tail' in config['split_type']:
        result = test1(model, data)
    else:
        result = test2(model, data)
    return result


def write_result(result, config, args):
    argsDict = args.__dict__
    if args.params != None:
        if 'gnn' in args.params:
            k = args.params[4:]
            filename  = f"{config['outpth']}/sailor_result_{args.params}{config['gnn'][k]}.csv"
        elif 'augmentor' in args.params:
            k = args.params[9:]
            filename  = f"{config['outpth']}/sailor_result_{args.params}{config['augmentor'][k]}.csv"
        else:
            filename  = f"{config['outpth']}/sailor_result_{args.params}{config[f'{args.params}']}.csv"
    else:
        filename = f"{config['outpth']}/sailor_result.csv"
    # write out interested args
    if not os.path.exists(filename):
        fout = open(filename, 'w')
        for eachArg, value in argsDict.items():
            if value != None and eachArg != 'outpth' and eachArg != 'config' and eachArg != 'is_feat_preprocess':
                fout.write(f"{eachArg},")
        if 'tail' in config['split_type']:
            fout.write("acc_test,weightf1_test\n")
        else:
            fout.write('acc_test,weightf1_test,')
            fout.write('acc_test:head,weightf1_test:head,')
            fout.write('acc_test:tail,weightf1_test:tail\n')

    # 写内容
    fout = open(filename, 'a')
    for eachArg, value in argsDict.items():
        if value != None and eachArg != 'outpth' and eachArg != 'config' and eachArg != 'is_feat_preprocess':
            fout.write(f"{str(value)},")
    if 'tail' in config['split_type']:
        fout.write(f"{result['acc_test']*100},{result['weighted_f1_test']*100}\n")
    else:
        fout.write(f"{result['acc_test']*100},{result['weighted_f1_test']*100},")
        fout.write(f"{result['acc_test:head']*100},{result['weighted_f1_test:head']*100},")
        fout.write(f"{result['acc_test:tail']*100},{result['weighted_f1_test:tail']*100}\n")


def define_model(data, config):
    aug_model = SAILOR(config,nnodes=data.num_nodes, 
                       nfeat=data.num_feats, nclass=data.num_classes)
    return aug_model


if __name__ == "__main__":
    # configs
    args = get_args()
    config = get_config(args)
    if not os.path.exists(f"{config['outpth']}"):
        os.makedirs(f"{config['outpth']}") 
    if not os.path.exists(f"{config['outpth']}/saved_models"):
        os.makedirs(f"{config['outpth']}/saved_models") 
    device = torch.device(f"cuda:{bestGPU(True)}" if config['iscuda'] else "cpu")
    if config['gpu_id'] != -1:
        device=torch.device(f"cuda:{config['gpu_id']}")
        print(f"##########Manually set gpu: {config['gpu_id']}##########")
    config['device'] = device

    assert config['batch'] == 0

    # define dataset
    setup_seed(config['seed'], config['iscuda'])
    dataset = TailDataset(name=config['dataset'], seed=config['seed'], degthres=config['degthres'],
                          use_lcc=config['use_lcc'], use_undirected=config['use_undirected'], 
                          verbose=config['verbose'])

    if args.is_feat_preprocess:
        dataset.data.x = preprocess_features(dataset.data.x)

    # split data
    if 'tail' in config['split_type']:
        dataset.data = set_tail_train_val_test_split(config['seed'], dataset.data)
    dataset.data = dataset.data.to(device)

    # run
    dataset.data.drop_edge_index = get_drop_graph(dataset.data.edge_index, 
                                                torch.arange(dataset.data.y.shape[0]).to(config['device'])[dataset.data.head_mask],
                                                dataset.data.y.shape[0], 
                                                config['drop_prt'])
    result = run(dataset.data, config)
    print(result)
    write_result(result, config, args)