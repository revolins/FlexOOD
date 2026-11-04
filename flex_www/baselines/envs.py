"""Training environments for the environment-consuming baselines (IRM, VRex, GroupDRO, DANN, CORAL, SRGNN, GALA, GIGraph).

Every training positive gets an environment label from the cell's own shift heuristic, scored on the TRAIN graph:
SP (1 / shortest-path length with the edge itself removed), CN (common neighbours) or PA (degree product), chosen by
the tag in the dataset name; cells without a tag (Amazon, DBLP) use CN. The score is cut into `num_envs` buckets of
roughly equal size (quantile boundaries; tied scores can leave fewer buckets) and renumbered 0..k-1.

Two protocol details that change numbers:
  * the CN / PA scorers iterate a DataLoader, and creating a DataLoader iterator draws one value from the global CPU
    generator, so the labels are built at the trainer's position (after the optimiser) to keep the RNG stream;
  * the environment count is carried from seed to seed: after each seed it is reset to (number of non-empty buckets),
    and the next seed buckets with that count (`EnvLabels` keeps it across the seeds of one run).
"""
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.loader import DataLoader


def cn_scores(A, edge_index, batch_size=100000):
    """Common neighbours of each pair (scipy CSR A; edge_index [2, E])."""
    edge_index = edge_index.cpu()
    scores = []
    for ind in DataLoader(range(edge_index.size(1)), batch_size):
        src, dst = edge_index[0, ind].numpy(), edge_index[1, ind].numpy()
        scores.append(np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten())
    return torch.FloatTensor(np.concatenate(scores, 0))


def pa_scores(A, edge_index, batch_size=100000):
    """Degree product of each pair (networkx degree of A)."""
    G = nx.from_scipy_sparse_array(A)
    deg = np.array(G.degree(np.array(G.nodes())))
    edge_index = edge_index.cpu()
    scores = []
    for ind in DataLoader(range(edge_index.size(1)), batch_size):
        src, dst = edge_index[0, ind].numpy(), edge_index[1, ind].numpy()
        scores.append(deg[src][:, 1] * deg[dst][:, 1])
    return torch.FloatTensor(np.concatenate(scores, 0))


def sp_scores(A, edge_index, remove=True):
    """1 / shortest-path length of each pair (999 when unreachable or a self pair); remove=True drops the pair's own
    edge while measuring it."""
    G = nx.from_scipy_sparse_array(A)
    edge_index = edge_index.cpu()
    scores = []
    for i in range(edge_index.size(1)):
        s, t = edge_index[0][i].item(), edge_index[1][i].item()
        if s == t:
            scores.append(999)
            continue
        removed = []
        if remove:
            for a, b in ((s, t), (t, s)):
                if (a, b) in G.edges:
                    G.remove_edge(a, b)
                    removed.append((a, b))
        sp = nx.shortest_path_length(G, source=s, target=t) if nx.has_path(G, source=s, target=t) else 999
        for a, b in removed:
            G.add_edge(a, b)
        scores.append(1 / sp)
    return torch.FloatTensor(scores)


def heuristic_tags(dataset):
    """Tags present in the dataset name, in the order the trainer scored them (the last one wins); CN if none."""
    tags = [t for t in ('SP', 'CN', 'PA') if t in dataset]
    return tags or ['CN']


def train_pair_scores(edge_index, train_pos, num_nodes, dataset):
    """Shift-heuristic score of every training positive on the train graph (`edge_index`: the raw message-passing
    edges, duplicates summed by the CSR constructor, then symmetrised)."""
    ei = edge_index.cpu()
    A = ssp.csr_matrix((np.ones(ei.size(1), dtype=np.int64), (ei[0].numpy(), ei[1].numpy())), shape=(num_nodes, num_nodes))
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    pairs = train_pos.t().cpu()
    out = None
    for tag in heuristic_tags(dataset):
        out = {'SP': lambda: sp_scores(A, pairs, True), 'CN': lambda: cn_scores(A, pairs), 'PA': lambda: pa_scores(A, pairs)}[tag]()
    return out


def det_environ(e_t, num_bins):
    """1-based bucket of each score: quantile boundaries (equal-sized buckets), duplicate boundaries dropped."""
    scores = e_t.float().flatten()
    max_q = 16_777_216                       # torch.quantile input limit: estimate the boundaries on a regular subsample
    basis = scores if scores.numel() <= max_q else scores[torch.linspace(
        0, scores.numel() - 1, max_q, dtype=torch.long, device=scores.device)]
    qs = torch.quantile(basis, torch.linspace(0, 1, num_bins + 1, device=basis.device)[1:-1])
    qs = torch.unique(qs)
    return torch.bucketize(scores, qs, right=True).reshape(e_t.shape) + 1


def env_labels(scores, num_envs):
    """(env_idx [E] in 0..k-1, mask [E] of ones, k) from heuristic scores."""
    if torch.unique(scores).numel() > num_envs:
        env_idx = det_environ(scores, num_envs)
    else:
        env_idx = scores.long()
    unique_values, inverse = torch.unique(env_idx, return_inverse=True)
    env_idx = torch.tensor([0, 1, 2, 3, 4][:len(unique_values)])[inverse]
    return env_idx, torch.ones(len(env_idx)), int(torch.max(env_idx)) + 1


class EnvLabels:
    """Setup hook that writes ctx.extra['env_idx'], ['mask'], ['num_envs']; the environment count carries over between
    the seeds of one run."""

    def __init__(self, num_envs=3):
        self.num_envs = int(num_envs)

    def __call__(self, ctx):
        s = ctx.split
        scores = train_pair_scores(ctx.graph.edge_index, ctx.edges[0], s.num_nodes, s.dataset)
        env_idx, mask, self.num_envs = env_labels(scores, self.num_envs)
        ctx.extra.update(env_idx=env_idx, mask=mask, num_envs=self.num_envs)
