"""Amazon Photo <-> Computers cross-domain cells.

The split is a fresh RandomLinkSplit (5% validation, 10% test, shared random negatives) drawn from the global torch RNG,
so every seed has its own split: call `flex_www.utils.init_seed(seed)` immediately before loading. Self-loops are added
to the message-passing graph only, after the split (never sampled as positives).

Cross-domain warm start (`x_d`): the model is initialised from the SOURCE domain's checkpoint, so the target features
are truncated (Computers 767 -> 745) or zero-padded (Photo 745 -> 767) to the source width.
"""
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, to_undirected

from pathlib import Path

from . import AMAZON_CELLS, Split
from ..utils import data_root

FEATURE_DIM = {'Photo': 745, 'Computers': 767}


def amazon_data(name, root=None):
    """(data, split_edge) of one Amazon graph: ToUndirected, RandomLinkSplit(0.05, 0.10), train graph + self-loops."""
    from torch_geometric.datasets import Amazon
    from torch_geometric.transforms import RandomLinkSplit
    dataset = Amazon(root=str(Path(root or data_root()) / 'Amazon'), name=name, transform=T.Compose([T.ToUndirected()]))
    data = dataset[0]
    train_data, val_data, test_data = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, split_labels=True,
                                                      add_negative_train_samples=False)(data)
    data = data.clone()
    data.edge_index = add_self_loops(train_data.edge_index, num_nodes=data.num_nodes)[0]
    split_edge = {'train': {'edge': train_data.pos_edge_label_index.t()},
                  'valid': {'edge': val_data.pos_edge_label_index.t(), 'edge_neg': val_data.neg_edge_label_index.t()},
                  'test': {'edge': test_data.pos_edge_label_index.t(), 'edge_neg': test_data.neg_edge_label_index.t()}}
    return data, split_edge


def pad_features(x, target_dim):
    pad_size = target_dim - x.size(1)
    if pad_size > 0:
        x = torch.cat([x, torch.zeros((x.size(0), pad_size), device=x.device)], dim=1)
    return x


def truncate_features(x, target_dim):
    return x[:, :target_dim] if x.size(1) > target_dim else x


def load_cross_domain(cell, root=None, x_d=True):
    source, target = AMAZON_CELLS[cell]
    data, split_edge = amazon_data(target, root)
    # message passing on the split's training edges (+ self-loops) only
    data.edge_index = add_self_loops(to_undirected(split_edge['train']['edge'].t()), num_nodes=data.num_nodes)[0]
    if x_d:
        data.x = truncate_features(data.x, FEATURE_DIM[source]) if target == 'Computers' else pad_features(data.x, FEATURE_DIM[source])
    return Split(cell=cell, dataset=target, data=data, train_pos=split_edge['train']['edge'],
                 valid_pos=split_edge['valid']['edge'], valid_neg=split_edge['valid']['edge_neg'],
                 test_pos=split_edge['test']['edge'], test_neg=split_edge['test']['edge_neg'],
                 metric='Hits@50', layout='pooled', source_domain=source if x_d else None)
