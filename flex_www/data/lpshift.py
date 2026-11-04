"""LPShift splits (cora / citeseer / pubmed / ogbl-collab x CN / PA / SP shifts) with the corrected HeaRT negatives.

Each cell directory holds the graph (`<name>Dataset.pt`: x and the TRAIN message-passing edge_index), the split
(`<name>Dataset_split.pt`: train / valid / test positives) and `heart_{valid,test}_samples.npy`: 250 hard negatives PER
positive, [P, 250, 2] (HeaRT protocol; rebuilt with both halves of the (*, v) corruption masked). Validation and test
positives are ranked against their own negatives only.
"""
import numpy as np
import torch

from pathlib import Path

from . import Split
from ..utils import data_root


def cell_dir(cell, root=None):
    name = f'{cell}_hfix_seed1'
    return Path(root or data_root()) / 'lpshift' / f'{name}Dataset', name


def load(cell, root=None):
    d, name = cell_dir(cell, root)
    data = torch.load(d / f'{name}Dataset.pt', map_location='cpu', weights_only=False)
    split = torch.load(d / f'{name}Dataset_split.pt', map_location='cpu', weights_only=False)
    neg_valid = torch.from_numpy(np.load(d / 'heart_valid_samples.npy'))
    neg_test = torch.from_numpy(np.load(d / 'heart_test_samples.npy'))
    return Split(cell=cell, dataset=f'{cell}_hfix', data=data, train_pos=split['train']['edge'],
                 valid_pos=split['valid']['edge'], valid_neg=neg_valid, test_pos=split['test']['edge'], test_neg=neg_test,
                 metric='Hits@20', layout='heart')
