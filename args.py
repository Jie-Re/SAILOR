import argparse

def get_args():
    if not hasattr(get_args, "args"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=0, help='Random seed')
        parser.add_argument('--outpth', type=str, required=True)
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--is_feat_preprocess', action='store_true', default=False)
        parser.add_argument('--params', type=str, default=None)
        parser.add_argument('--n_trials', type=int, default=None)
        
        parser.add_argument('--dataset', type=str, default=None,
                            choices=['acm', 'citeseer', 'cora', 'email', 'pubmed', 'arxiv'])
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--batch', type=int, default=None)
        parser.add_argument('--num_neighbors', type=str, default=None)
        
        parser.add_argument('--temperature', type=float, default=None)
        parser.add_argument('--alpha', type=float, default=None)
        parser.add_argument('--beta', type=float, default=None)
        parser.add_argument('--eta', type=float, default=None)
        parser.add_argument('--theta', type=float, default=None)
        parser.add_argument('--epsilon', type=float, default=None)
        parser.add_argument('--drop_prt', type=float, default=None)
                            
        parser.add_argument('--weight_decay', type=float, default=None)
        parser.add_argument('--with_decay', type=int, default=None)
        parser.add_argument('--with_bias', type=int, default=None)
        parser.add_argument('--dropout', type=float, default=None)
        parser.add_argument('--hidden', type=int, default=None, help="Number of hidden units")
        parser.add_argument('--epoch', type=int, default=None)
        parser.add_argument('--patience', type=int, default=None)
        parser.add_argument('--gnn_optimizer', type=str, default=None)
        parser.add_argument('--augmentor_optimizer', type=str, default=None)
        parser.add_argument('--gnn_norm_type', type=str, default=None)
        parser.add_argument('--gnn_n_layers', type=int, default=None, help="Number of hidden layers")
        parser.add_argument('--augmentor_n_layers', type=int, default=None, help="Number of hidden layers")
        parser.add_argument('--gnn_lr', type=float, default=None)
        parser.add_argument('--augmentor_lr', type=float, default=None)

        parser.add_argument('--split_type', type=str, default=None)
        parser.add_argument('--visualize', type=int, default=None)
        parser.add_argument('--verbose', type=int, default=None)
        parser.add_argument('--with_gpu', type=int, default=None)
        parser.add_argument('--gpu_id', type=int, default=-1)
        
        get_args.args = parser.parse_args()
    return get_args.args