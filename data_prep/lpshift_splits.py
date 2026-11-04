"""Build the LPShift splits: graph (`<name>Dataset.pt`) and positives (`<name>Dataset_split.pt`) of one or more cells.

A cell `<dataset>_<CN|PA|SP>_<a>_<b>_<c>` scores every edge of the dataset with one structural heuristic (common
neighbours, preferential attachment or inverse shortest-path length, the edge itself removed for SP) and buckets it by
two thresholds: score < b -> train, b <= score < max(a, c) -> valid, score >= max(a, c) -> test. When a > c the
direction is inverted (high scores train, low scores test), so `CN_0_1_2` tests on high-CN links and `CN_2_1_0` on
CN-0 links. SP thresholds are written without the decimal point (`017` = 0.17).

Valid / test edges whose endpoints do not both occur in a training edge are dropped, valid / test edges are removed
from the message-passing graph, and every valid / test positive gets 250 random negatives (125 corrupting each
endpoint, never an existing edge). The hard HeaRT negatives the benchmark evaluates on are built afterwards by
data_prep/lpshift_heart.py, which continues the same seeded random stream.

Inputs: the PyG Planetoid raw files (<raw>/<cora|citeseer|pubmed>/raw/) and the OGB ogbl-collab download
(<raw>/ogbl_collab/); PyG / OGB process them on first use.
Outputs: <out>/lpshift/<cell>_hfix_seed1Dataset/<cell>_hfix_seed1Dataset{.pt,_split.pt}

    python -m data_prep.lpshift_splits --raw <PyG/OGB root> --out data --cells cora_SP_026_017_00 pubmed_CN_0_1_2
"""
import argparse
import random
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected, to_undirected
from torch_sparse import SparseTensor

from flex_www.data import LPSHIFT_CELLS
from flex_www.data.lpshift import cell_dir
from flex_www.utils import init_seed, ogb_link_dataset

SEED = 1                  # every released cell is seed 1
NUM_RANDOM_NEG = 250      # random negatives per valid / test positive, half per corrupted endpoint
MAX_EVAL_POS = 100000     # valid / test positives are subsampled to at most this many before filtering

# The released cells were written by three revisions of the generator. The splits (graph, positives) follow the same
# protocol in all three; the revisions differ only in what else they stored and, for ogbl-collab, in which edge list
# was scored. Reproducing a cell field for field needs its revision:
#   store_random_neg  the drawn random negatives are stored as edge_neg; otherwise edge_neg holds a [[0, 0]]
#                     placeholder (the negatives are still drawn, so the random stream is the same)
#   store_cn_mask     the graph carries `z` (see cn_node_mask); flex_www does not read it
#   score_ogb_split   ogbl-collab: score the OGB train-split edge list (one row per collaboration record, one
#                     orientation) instead of the undirected training graph
FORMATS = {
    'planetoid_CN': dict(store_random_neg=True, store_cn_mask=True, score_ogb_split=False),
    'planetoid_PA': dict(store_random_neg=False, store_cn_mask=False, score_ogb_split=False),
    'planetoid_SP': dict(store_random_neg=False, store_cn_mask=False, score_ogb_split=False),
    'ogbl-collab_CN': dict(store_random_neg=True, store_cn_mask=False, score_ogb_split=False),
    'ogbl-collab_SP': dict(store_random_neg=True, store_cn_mask=False, score_ogb_split=False),
    'ogbl-collab_PA': dict(store_random_neg=False, store_cn_mask=False, score_ogb_split=True),
}


def parse_cell(cell):
    """'cora_SP_026_017_00' -> ('cora', 'sp', valid=0.17, test=0.26, inverse=True)."""
    base, kind, a, b, c = cell.rsplit('_', 4)
    if kind == 'SP':
        value = lambda s: float('0.' + s[1:]) if len(s) > 1 else 0.0  # noqa: E731
    else:
        value = int
    a, b, c = value(a), value(b), value(c)
    return base, kind.lower(), b, max(a, c), a > c


def cell_format(cell):
    base, kind = cell.rsplit('_', 4)[:2]
    return FORMATS[f'{base if base.startswith("ogbl") else "planetoid"}_{kind}']


def load_graph(base, raw):
    """Full undirected graph of the dataset: features, edge_index (numpy [2, E]), CSR adjacency, OGB split if any."""
    if base.startswith('ogbl'):
        dataset, ogb_split = ogb_link_dataset(base, raw)
        graph = dataset[0]
        x, num_nodes = graph.x, graph.num_nodes
        edge_index = to_undirected(graph.edge_index).numpy()
    else:
        from torch_geometric.datasets import Planetoid
        graph = Planetoid(root=raw, name=base)[0]
        x, num_nodes, ogb_split = graph.x, int(len(graph.y)), None
        edge_index = to_undirected(graph.edge_index).numpy()
    ones = np.ones(edge_index.shape[1], dtype=np.int64)
    A = sp.coo_matrix((ones, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    return x, num_nodes, edge_index, A, ogb_split


def common_neighbors(A, edge_index, batch_size=100000):
    scores = []
    for i in range(0, edge_index.size(1), batch_size):
        src, dst = edge_index[0, i:i + batch_size].numpy(), edge_index[1, i:i + batch_size].numpy()
        scores.append(np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten())
    return torch.FloatTensor(np.concatenate(scores, 0))


def preferential_attachment(A, edge_index):
    # networkx degree of the undirected graph: a self-loop counts twice
    degree = np.diff(A.indptr) + (A.diagonal() != 0)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    return torch.FloatTensor(degree[src] * degree[dst])


def inverse_shortest_path(A, edge_index):
    """1 / d(s, t) with the edge (s, t) itself removed; 1/999 when disconnected, 999 for a self-loop."""
    G = nx.from_scipy_sparse_array(A)
    scores = []
    for s, t in edge_index.t().tolist():
        if s == t:
            scores.append(999)
            continue
        removed = G.has_edge(s, t)
        if removed:
            G.remove_edge(s, t)
        d = nx.shortest_path_length(G, source=s, target=t) if nx.has_path(G, source=s, target=t) else 999
        if removed:
            G.add_edge(s, t)
        scores.append(1 / d)
    return torch.FloatTensor(scores)


HEURISTICS = {'cn': common_neighbors, 'pa': preferential_attachment, 'sp': inverse_shortest_path}


def heuristic_split(A, score_edges, kind, valid_thr, test_thr, inverse):
    """Bucket every scored edge into train / valid / test by its heuristic score (lists of (u, v) tuples)."""
    scores = HEURISTICS[kind](A, score_edges).double().numpy()
    low, mid, high = [], [], []
    for (u, v), s in zip(score_edges.t().tolist(), scores):
        if s < valid_thr:
            low.append((u, v))
        elif s < test_thr:
            mid.append((u, v))
        else:
            high.append((u, v))
    train, valid, test = (high, mid, low) if inverse else (low, mid, high)
    # set() order is deterministic for int tuples; the order of the split files depends on it
    train, valid, test = list(set(train)), list(set(valid)), list(set(test))
    if len(test) > MAX_EVAL_POS:
        test = random.sample(test, k=MAX_EVAL_POS)
    if len(valid) > MAX_EVAL_POS:
        valid = random.sample(valid, k=MAX_EVAL_POS)
    return train, valid, test


def cn_node_mask(A):
    """`z`: 1 for every id that occurs in the (edge position, common neighbour) incidence of A -- the positions of
    edges with at least one common neighbour and the common-neighbour nodes themselves -- restricted to ids that are
    endpoints of an edge. Stored with the binary sparse adjacency as the pair (z, adj)."""
    coo = A.tocoo()
    row, col = coo.row.astype(np.int64), coo.col.astype(np.int64)
    both = A[row].multiply(A[col]).tocoo()          # (e, w): w is adjacent to both endpoints of edge e
    ids = torch.from_numpy(np.concatenate([both.row, both.col]).astype(np.int64))
    endpoints = torch.from_numpy(np.concatenate([row, col]))
    z = torch.zeros(int(endpoints.max()) + 1, dtype=torch.int64)
    z = z.index_fill(0, torch.unique(ids[torch.isin(ids, endpoints)]), 1)
    adj = torch.sparse_coo_tensor(torch.from_numpy(np.stack([row, col])), torch.from_numpy(coo.data.astype(np.int64)))
    return z, adj.coalesce().bool().float()


def draw_random_negatives(adj, pos, k=NUM_RANDOM_NEG // 2):
    """For each positive (s, t): k pairs (s, *) and k pairs (*, t), * uniform (with replacement) over the nodes that
    are neither an endpoint nor a neighbour of the kept endpoint in the CSR graph `adj`. Consumes np.random."""
    n = adj.shape[0]
    out = []
    for s, t in pos.tolist():
        cand = []
        for node in (s, t):
            free = np.ones(n, dtype=bool)
            free[adj.indices[adj.indptr[node]:adj.indptr[node + 1]]] = False
            free[[s, t]] = False
            cand.append(np.flatnonzero(free))
        src_side = np.concatenate((np.repeat(s, k).reshape(-1, 1), np.random.choice(cand[0], k).reshape(-1, 1)), axis=-1)
        dst_side = np.concatenate((np.random.choice(cand[1], k).reshape(-1, 1), np.repeat(t, k).reshape(-1, 1)), axis=-1)
        out.append(np.concatenate((src_side, dst_side)))
    return np.stack(out)


def build_cell(cell, raw):
    """Return (graph Data, split dict) of one cell, exactly as released."""
    fmt = cell_format(cell)
    base, kind, valid_thr, test_thr, inverse = parse_cell(cell)
    init_seed(SEED)
    x, num_nodes, edge_index, A, ogb_split = load_graph(base, raw)
    score_edges = ogb_split['train']['edge'].t() if fmt['score_ogb_split'] else torch.tensor(edge_index)
    train, valid, test = heuristic_split(A, score_edges, kind, valid_thr, test_thr, inverse)

    valid_set, test_set = set(valid), set(test)
    assert not any(e in valid_set or e in test_set for e in train), 'train edges overlap valid / test'
    train_nodes = {u for e in train for u in e}
    valid = [e for e in valid if e[0] in train_nodes and e[1] in train_nodes]
    test = [e for e in test if e[0] in train_nodes and e[1] in train_nodes]

    # message-passing graph: every edge except the (unfiltered) valid / test edges, in either orientation
    held_out = valid_set | test_set
    kept = [e for e in map(tuple, edge_index.T.tolist()) if e not in held_out and (e[1], e[0]) not in held_out]
    kept_set = set(kept)
    kept += [e for e in train if e not in kept_set]
    graph_edges = to_undirected(coalesce(torch.tensor(kept, dtype=torch.long).t()))
    assert is_undirected(graph_edges) and int(graph_edges.max()) < num_nodes

    adj_t = SparseTensor.from_edge_index(graph_edges, torch.ones(graph_edges.shape[1]), [num_nodes, num_nodes])
    adj_t = adj_t.to_symmetric().coalesce().bool().float()
    fields = dict(edge_index=graph_edges, x=x, adj_t=adj_t, num_nodes=num_nodes)
    if fmt['store_cn_mask']:
        fields['z'] = cn_node_mask(A)
    data = Data(**fields)

    adj = adj_t.to_scipy(layout='csr')
    positives = {'train': torch.tensor(train), 'valid': torch.tensor(valid), 'test': torch.tensor(test)}
    in_graph = lambda e: np.asarray(adj[e[:, 0], e[:, 1]]).ravel() != 0  # noqa: E731
    assert in_graph(positives['train'].numpy()).all(), 'train positive missing from the graph'
    split = {}
    for name, pos in positives.items():
        if name == 'train':
            neg = torch.tensor(np.zeros((1, 1)))
        else:
            assert not in_graph(pos.numpy()).any(), f'{name} positive inside the graph'
            neg = torch.tensor(draw_random_negatives(adj, pos))
        split[name] = {'edge': pos, 'edge_neg': neg if fmt['store_random_neg'] else torch.tensor([[0, 0]])}
    return data, split


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--raw', required=True, help='PyG / OGB root holding cora/, citeseer/, pubmed/ and ogbl_collab/')
    ap.add_argument('--out', required=True, help='data root; writes <out>/lpshift/<cell>_hfix_seed1Dataset/')
    ap.add_argument('--cells', nargs='+', required=True, help=f'cell names, or "all" for the {len(LPSHIFT_CELLS)} released cells')
    args = ap.parse_args()
    cells = list(LPSHIFT_CELLS) if args.cells == ['all'] else args.cells
    for cell in cells:
        t0 = time.time()
        data, split = build_cell(cell, args.raw)
        d, name = cell_dir(cell, args.out)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(data, d / f'{name}Dataset.pt')
        torch.save(split, d / f'{name}Dataset_split.pt')
        sizes = {k: tuple(v['edge'].shape) for k, v in split.items()}
        print(f'{cell}: graph edges {data.edge_index.shape[1]}, positives {sizes}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
