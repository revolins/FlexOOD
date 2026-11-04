"""Dataset loading: every loader returns a `Split`, and `build_graph` turns it into the model input.

    from flex_www.data import load, build_graph
    split = load('cora_CN_0_1_2', seed=1)               # seed matters only for Amazon (one RandomLinkSplit per seed)
    graph = build_graph(split, device)                  # binary symmetric adj_t (+ full_adj_t for test-time input)

Evaluation layouts (see flex_www.evaluate):
  heart      every validation / test positive has its OWN negatives, shape [P, K, 2] (HeaRT); metrics rank each positive
             against its own K negatives (LPShift cells, including the ogbl-collab LPShift cells)
  pooled     one shared negative pool, OGB Hits@K (Amazon, real ogbl-collab)
  per_query  DBLP: each positive (author -> paper) is ranked against its own `per_query` typed negatives, which are
             stored flat [P * per_query, 2] in query order
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index, to_undirected
from torch_sparse import SparseTensor

LPSHIFT_BASES = ('cora', 'citeseer', 'pubmed', 'ogbl-collab')
LPSHIFT_SHIFTS = ('CN_0_1_2', 'CN_2_1_0', 'PA_0_50_100', 'PA_100_50_0', 'SP_00_017_026', 'SP_026_017_00')
LPSHIFT_CELLS = tuple(f'{b}_{s}' for b in LPSHIFT_BASES for s in LPSHIFT_SHIFTS)
AMAZON_CELLS = {'Photo_to_Computers': ('Photo', 'Computers'), 'Computers_to_Photo': ('Computers', 'Photo')}


@dataclass
class Split:
    """One link-prediction split. Edge tensors are [P, 2] (negatives [P, K, 2] under the heart layout)."""
    cell: str                     # cell key, e.g. cora_CN_0_1_2 (the configs/flex/<cell>.yaml stem)
    dataset: str                  # dataset argument of the training command (also the checkpoint-name stem)
    data: Data                    # x, edge_index (message-passing graph, before sparse conversion), num_nodes[, edge_weight]
    train_pos: torch.Tensor
    valid_pos: torch.Tensor
    valid_neg: torch.Tensor
    test_pos: torch.Tensor
    test_neg: torch.Tensor
    metric: str                   # the cell's headline metric (FLEX picks and the reported numbers use it)
    layout: str                   # 'heart' | 'pooled' | 'per_query'
    stop_metric: str = 'Hits@20'  # the trainer's own selection metric: early stopping and the saved checkpoint follow it
                                  # (Hits@20 everywhere except real ogbl-collab Hits@50 and DBLP MRR; so on Amazon the
                                  # checkpoint is the best-Hits@20 epoch while the cell reports Hits@50)
    per_query: int = 0            # negatives per query (per_query layout)
    typed_neg_dst: Optional[Tuple[int, int]] = None   # DBLP: training negatives keep the source, destination in [lo, hi)
    source_domain: Optional[str] = None               # Amazon cross-domain: the warm-start (source) domain
    info: dict = field(default_factory=dict)

    @property
    def num_nodes(self):
        return int(self.data.num_nodes)

    @property
    def family(self):
        if self.cell in LPSHIFT_CELLS:
            return 'lpshift'
        if self.cell in AMAZON_CELLS:
            return 'amazon'
        return self.cell                  # 'dblp' | 'ogbl-collab'


def cell_info(cell):
    """Shift metadata of a cell: base graph, named shift axis (None off LPShift) and family."""
    if cell in LPSHIFT_CELLS:
        base = next(b for b in LPSHIFT_BASES if cell.startswith(b + '_'))
        return {'key': cell, 'base': base, 'shift': cell[len(base) + 1:].split('_')[0], 'family': 'lpshift'}
    if cell in AMAZON_CELLS:
        return {'key': cell, 'base': AMAZON_CELLS[cell][1], 'shift': None, 'family': 'crossdomain'}
    return {'key': cell, 'base': cell, 'shift': None, 'family': 'realworld'}


def load(cell, seed=1, root=None, x_d=True):
    """Split for a cell key (root: the data bundle root, default flex_www.utils.data_root()). `seed` re-draws the Amazon RandomLinkSplit (call it right after init_seed(seed), as the
    trainer does); every other split is fixed."""
    if cell in LPSHIFT_CELLS:
        from . import lpshift
        return lpshift.load(cell, root=root)
    if cell in AMAZON_CELLS:
        from . import amazon
        return amazon.load_cross_domain(cell, root=root, x_d=x_d)
    if cell == 'dblp':
        from . import dblp
        return dblp.load(root=root)
    if cell == 'ogbl-collab':
        from . import collab
        return collab.load(root=root)
    raise KeyError(cell)


@dataclass
class Graph:
    """Model input built from a Split: `data` carries x, adj_t (binary, symmetric) and, when the test pairs are scored on
    train + validation edges, full_adj_t; `edge_index` is the raw message-passing edge_index (used for the training
    negatives of the train-metric column)."""
    data: Data
    edge_index: torch.Tensor
    val_input_for_test: bool = False


def build_graph(split, device, val_edges_test_input=False, use_valedges_as_input=False):
    """Sparse model input, exactly as the trainer built it: ToSparseTensor, binarised symmetric adj_t, optional test-time
    graphs, then moved to `device`.

    val_edges_test_input  LPShift PA-up cells (VALIN): TEST pairs are scored on train + validation edges (unweighted);
                          training and validation stay on the train graph.
    use_valedges_as_input real ogbl-collab (OGB rule): test pairs on train + validation edges, keeping edge weights."""
    data = split.data.clone()
    edge_index = data.edge_index
    edge_weight = None
    if getattr(data, 'edge_weight', None) is not None:
        edge_weight = data.edge_weight.to(torch.float)
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    data.edge_index = to_edge_index(data.adj_t)[0]
    if use_valedges_as_input:
        val_edge_index = to_undirected(split.valid_pos.t())
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        w = torch.cat([edge_weight, torch.ones([val_edge_index.size(1), 1], dtype=torch.float)], 0)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, w.view(-1), [data.num_nodes, data.num_nodes])
        data.full_edge_index = full_edge_index
    data.adj_t = data.adj_t.coalesce().bool().float()
    data.adj_t = data.adj_t.to_symmetric()
    if val_edges_test_input:
        assert not use_valedges_as_input
        ve = to_undirected(split.valid_pos.t().cpu().long())
        r_, c_, _ = data.adj_t.coo()
        data.full_adj_t = SparseTensor(row=torch.cat([r_.cpu(), ve[0]]), col=torch.cat([c_.cpu(), ve[1]]),
                                       value=torch.ones(r_.numel() + ve.size(1)),
                                       sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce().bool().float().to_symmetric()
    data = data.to(device)
    return Graph(data=data, edge_index=edge_index, val_input_for_test=bool(val_edges_test_input))
