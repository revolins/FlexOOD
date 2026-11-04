"""Leak-free few-shot DBLP (author -> paper link prediction).

Validation / test authors reveal one paper; their remaining papers are the queries. Author features are built from
visible papers only. Every query positive has 50 TYPED negatives (author -> paper, never an existing author-paper edge),
stored flat [P * 50, 2] in query order; metrics rank each positive against its own 50 (MRR is the headline metric).
Training negatives are typed too: the source is kept and the destination is drawn from the contiguous paper-id block,
so training optimises the task that is evaluated. The split is cached in `harness_k1_neg50.npz` (built by
data_prep/dblp_fewshot.py).
"""
import numpy as np
import torch
from torch_geometric.data import Data

from pathlib import Path

from . import Split
from ..utils import data_root

NEGATIVES = 50


def load(root=None, negatives=NEGATIVES):
    d = Path(root or data_root()) / 'dblp'
    z = np.load(d / f'harness_k1_neg{negatives}.npz')
    n = int(z['num_nodes'][0])
    edge_index = torch.from_numpy(np.asarray(z['edge_index'])).long()
    data = Data(edge_index=edge_index, x=torch.from_numpy(np.asarray(z['x'])).float(), num_nodes=n)
    split = {k: torch.from_numpy(np.asarray(z[k])).long() for k in ('train_pos', 'val_pos', 'val_neg', 'test_pos', 'test_neg')}
    for s in ('val', 'test'):
        assert split[f'{s}_neg'].size(0) == split[f'{s}_pos'].size(0) * negatives
    papers = np.flatnonzero(np.load(d / 'node_types.npy') == 1)
    assert int(papers[-1]) - int(papers[0]) + 1 == len(papers), 'paper ids are not contiguous'
    return Split(cell='dblp', dataset='dblp', data=data, train_pos=split['train_pos'], valid_pos=split['val_pos'],
                 valid_neg=split['val_neg'], test_pos=split['test_pos'], test_neg=split['test_neg'], metric='MRR', stop_metric='MRR',
                 layout='per_query', per_query=negatives, typed_neg_dst=(int(papers[0]), int(papers[-1]) + 1))
