"""Build the corrected HeaRT negatives of LPShift cells: heart_{valid,test}_samples.npy, [P, 250, 2] per split.

HeaRT gives every valid / test positive (s, t) its own 250 hard negatives: 125 pairs (s, w) whose w are the top-ranked
common-neighbour scores in s's row, and 125 pairs (w, t) from t's row. Before ranking, each row masks the nodes that
cannot be a negative: neighbours of the endpoint (training graph for valid; training plus valid edges for test),
nodes that occur in no training positive, and both endpoints. When fewer than 125 candidates have a positive score,
the rest are drawn uniformly from the zero-score nodes.

The correction: the LPShift generator masked the (s, *) row twice and never masked the (*, t) row, so the target-side
half could contain existing edges, the endpoints themselves and nodes unseen in training. Here both halves of the
(*, v) corruption are masked, as in the original HeaRT protocol.

The zero-score fallback draws from NumPy's global stream. To give the released files, that stream must be where the
split generator left it: re-seed with the cell's seed, then redraw the random valid and test negatives exactly as
data_prep/lpshift_splits.py did (this also checks them against the stored ones when the split keeps them).

Input / output: <root>/lpshift/<cell>_hfix_seed1Dataset/ (graph and split from data_prep/lpshift_splits.py); the two
.npy files are written next to them.

    python -m data_prep.lpshift_heart --root data --cells cora_SP_026_017_00 pubmed_CN_0_1_2
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from torch_geometric.utils import to_undirected

from flex_www.data import LPSHIFT_CELLS
from flex_www.data.lpshift import cell_dir
from flex_www.utils import init_seed

from .lpshift_splits import NUM_RANDOM_NEG, SEED, draw_random_negatives

K = NUM_RANDOM_NEG // 2
PRECOMPUTE_MAX_NODES = 200000     # above this the CN matrix A @ A is not materialised; rows are computed on demand


class CNRows:
    """Dense float32 rows of A (or of the common-neighbour matrix A @ A, square=True) for a binary CSR matrix A."""

    def __init__(self, A, square=False):
        A = A.astype(np.float32)
        self.n = A.shape[1]
        self.lazy = square and self.n > PRECOMPUTE_MAX_NODES
        self.A = A
        self.M = A if not square or self.lazy else (A @ A).tocsr()

    def __getitem__(self, i):
        r = self.A[i] @ self.A if self.lazy else self.M[i]
        out = torch.zeros(self.n)
        out[torch.from_numpy(r.indices.astype(np.int64))] = torch.from_numpy(r.data.astype(np.float32))
        return out


def rank_desc(row):
    """In place: positive scores -> dense-from-top 'min' ranks (1 = best), zeros -> max + 1, masked (-1) -> max + 2."""
    pos = row > 0
    if pos.any():
        r = rankdata(row[pos], method='min')
        r = r.max() - r + 1
        top = r.max()
    else:
        r, top = [], 0
    row[pos] = r
    row[row == 0] = top + 1
    row[row < 0] = top + 2
    return row


def top_k_nodes(scores, mask, num_nodes):
    """K nodes for one endpoint: best-ranked positive scores, padded with uniform draws from zero-score nodes."""
    zero = (scores == 0).numpy()
    ranks = rank_desc(scores.numpy())
    n_positive = num_nodes - zero.sum().item() - mask.sum().item()
    if n_positive >= K:
        top = torch.topk(torch.from_numpy(-ranks), K).indices.numpy()
    elif n_positive <= 0:
        top = np.random.choice(zero.nonzero()[0], K)
    else:
        best = torch.topk(torch.from_numpy(-ranks), n_positive).indices.numpy()
        top = np.concatenate((best, np.random.choice(zero.nonzero()[0], K - n_positive)))
    return top.reshape(-1, 1)


def heart_negatives(pos, cn, neighbors, not_in_train, num_nodes):
    out = []
    for s, t in pos.tolist():
        s_mask, t_mask = neighbors[s].bool(), neighbors[t].bool()
        s_scores, t_scores = cn[s], cn[t]
        s_mask[not_in_train] = 1
        t_mask[not_in_train] = 1
        s_mask[s], s_mask[t] = 1, 1
        t_mask[t], t_mask[s] = 1, 1
        s_scores[s_mask], t_scores[t_mask] = -1, -1        # both halves masked (the correction)
        src_side = np.concatenate((np.repeat(s, K).reshape(-1, 1), top_k_nodes(s_scores, s_mask, num_nodes)), axis=-1)
        dst_side = np.concatenate((top_k_nodes(t_scores, t_mask, num_nodes), np.repeat(t, K).reshape(-1, 1)), axis=-1)
        out.append(np.concatenate((src_side, dst_side)))
    return np.stack(out)


def csr_from_edges(edge_index, n):
    """Binary symmetric CSR adjacency of an edge list [2, E]."""
    A = sp.coo_matrix((np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0].numpy(), edge_index[1].numpy())),
                      shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    return A.tocsr()


def build_cell(cell, root):
    d, name = cell_dir(cell, root)
    data = torch.load(d / f'{name}Dataset.pt', map_location='cpu', weights_only=False)
    split = torch.load(d / f'{name}Dataset_split.pt', map_location='cpu', weights_only=False)
    n = int(data.num_nodes)
    graph = data.adj_t.to_scipy(layout='csr')

    init_seed(SEED)
    for tvt in ('valid', 'test'):            # advance np.random exactly as the split generator did
        neg = draw_random_negatives(graph, split[tvt]['edge'])
        stored = split[tvt]['edge_neg']
        if stored.dim() == 3:
            assert np.array_equal(stored.numpy(), neg), f'{cell}: {tvt} random negatives do not replay'

    graph = csr_from_edges(torch.from_numpy(np.stack(graph.nonzero())).long(), n)
    train_and_valid = torch.cat([to_undirected(split['train']['edge'].t()), to_undirected(split['valid']['edge'].t())], 1)
    neighbors = {'valid': CNRows(graph), 'test': CNRows(csr_from_edges(train_and_valid, n))}
    cn = CNRows(graph, square=True)          # common-neighbour counts on the training graph
    train_nodes = set(split['train']['edge'].flatten().tolist())
    not_in_train = torch.tensor(sorted(set(range(n)) - train_nodes), dtype=torch.long)
    for tvt in ('valid', 'test'):
        H = heart_negatives(split[tvt]['edge'], cn, neighbors[tvt], not_in_train, n)
        with open(d / f'heart_{tvt}_samples.npy', 'wb') as f:
            np.save(f, H)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True, help='data root holding lpshift/<cell>_hfix_seed1Dataset/')
    ap.add_argument('--cells', nargs='+', required=True, help=f'cell names, or "all" for the {len(LPSHIFT_CELLS)} released cells')
    args = ap.parse_args()
    for cell in (list(LPSHIFT_CELLS) if args.cells == ['all'] else args.cells):
        t0 = time.time()
        build_cell(cell, args.root)
        print(f'{cell}: HeaRT negatives written, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
