"""Real ogbl-collab (OGB time split): train <= 2017, validation 2018, test 2019; shared random negatives, Hits@50.

Edge weights are kept on the Data object (the model input is binarised by build_graph); validation weights are used only
when validation edges join the test-time input graph (build_graph(..., use_valedges_as_input=True), the OGB rule).
"""
from pathlib import Path

from . import Split
from ..utils import data_root, ogb_link_dataset


def load(root=None):
    dataset, s = ogb_link_dataset('ogbl-collab', Path(root or data_root()) / 'ogb')
    data = dataset[0]
    return Split(cell='ogbl-collab', dataset='ogbl-collab', data=data, train_pos=s['train']['edge'],
                 valid_pos=s['valid']['edge'], valid_neg=s['valid']['edge_neg'], test_pos=s['test']['edge'],
                 test_neg=s['test']['edge_neg'], metric='Hits@50', layout='pooled', stop_metric='Hits@50',
                 info={'train_weight': s['train']['weight'], 'train_year': s['train']['year'],
                       'valid_weight': s['valid']['weight'], 'valid_year': s['valid']['year']})
