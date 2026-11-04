"""gMPNN (Zhou, Kutyniok & Ribeiro, NeurIPS 2022, "OOD Link Prediction Generalization Capabilities of Message-Passing
GNNs in Larger Test Graphs"), the 2-gMPNN of the authors' real-data script (learn-2-gmpnn-ddi.py), evaluated at the
queried pairs only.

A feature-free pairwise model replaces the GCN encoder entirely:
  f0[i,j]   = 1
  de2[i,j]  = |N(i) ∩ N(j)|, clipped to >= 1
  agg1[i,j] = (deg i + deg j) / (2 de2[i,j])
  f1[i,j]   = relu(nn1([f0, agg1]))                        (hid channels)
  agg2[i,j] = (sum_{z∈N(i)} f1[j,z] + sum_{z∈N(j)} f1[i,z]) / de2[i,j]
  f2[i,j]   = nn2([f1[i,j], agg2[i,j]])                    (scalar)
  score     = LinkPredictor(f2)                           (MLP + sigmoid)
trained with the ERM link loss (one uniform negative per positive, no gradient clipping). f2 depends only on degrees
and common-neighbour counts within one hop of i and j, so it is computed for the queried pairs (float64; identical to
the reference's dense N x N x hid tensor).

Evaluation: f2 is rounded to 1e-9 so mathematically tied pairs get bit-identical states, every distinct state goes
through the readout once and all pairs read their score from that table, and pairs are ranked by the readout's float64
logit (the sigmoid is monotone, but saturates to exactly 1.0 and would tie pairs the model orders). Validation / train
pairs are scored on the train graph, test pairs on the test-time graph (train + validation edges on the PA-up VALIN
cells).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_edge_index

from ..evaluate import metrics
from ..train import Hooks, typed_negatives

ROUND = 1e9  # f2 is rounded to 1e-9 before the readout


class GraphIndex:
    """CSR adjacency plus sorted edge keys, for vectorised common-neighbour counts.

    The reference graph (ogbl-ddi) is undirected with no self-loops and 0/1 entries,
    so the edge list is symmetrised (as the harness does for every other method with
    adj_t.to_symmetric()), self-loops are dropped (the harness adds them to the
    Amazon graphs for GCN message passing) and duplicates are collapsed.
    """

    def __init__(self, edge_index, num_nodes):
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        keep = row != col
        row, col = row[keep], col[keep]
        n = int(num_nodes)
        key = torch.unique(row * n + col)
        row, col = key // n, key % n
        self.n = n
        self.keys = key
        self.col = col
        self.deg = torch.bincount(row, minlength=n)
        self.rowptr = torch.zeros(n + 1, dtype=torch.long, device=key.device)
        self.rowptr[1:] = torch.cumsum(self.deg, 0)

    def neighbours(self, u, total=None):
        """Flattened neighbour lists of nodes u -> (index into u, neighbour id).

        `total` (= deg[u].sum()) can be passed when the caller already knows it. The
        output size is given to repeat_interleave so the expansion costs one host sync
        instead of one per call; on collab-sized graphs at batch 128 those syncs, not
        GPU work, dominated the step time.
        """
        cnt = self.deg[u]
        if total is None:
            total = int(cnt.sum())
        owner = torch.repeat_interleave(torch.arange(u.numel(), device=u.device), cnt, output_size=total)
        first = torch.cumsum(cnt, 0) - cnt
        offs = torch.arange(total, device=u.device) - first[owner]
        return owner, self.col[self.rowptr[u][owner] + offs]

    def prep(self, u, v):
        """Orient each pair so the lower-degree endpoint is expanded; returns the plan for
        common() and its total expansion size as a 0-d tensor (not yet synced)."""
        swap = self.deg[u] > self.deg[v]
        a_, b_ = torch.where(swap, v, u), torch.where(swap, u, v)
        da = self.deg[a_]
        return (a_, b_, torch.cumsum(da, 0)), da.sum()

    def common(self, u, v, chunk=8_000_000, plan=None, total=None):
        """Exact |N(u) ∩ N(v)| per pair, as float64. `plan`/`total` come from prep();
        passing them lets a caller fetch several totals in one host sync."""
        out = torch.zeros(u.numel(), dtype=torch.float64, device=u.device)
        if u.numel() == 0:
            return out
        if plan is None:
            plan, total = self.prep(u, v)
        a_, b_, cost = plan
        total = int(total)
        start = 0
        while start < u.numel():
            if start == 0 and total <= chunk:      # common case: one pass, no extra syncs
                stop, part = u.numel(), total
            else:
                base = cost[start - 1] if start > 0 else 0
                stop = max(int(torch.searchsorted(cost, base + chunk, right=True)), start + 1)
                part = None
            owner, w = self.neighbours(a_[start:stop], part)
            k = b_[start:stop][owner] * self.n + w
            pos = torch.searchsorted(self.keys, k).clamp(max=max(self.keys.numel() - 1, 0))
            hit = (self.keys[pos] == k).to(torch.float64) if self.keys.numel() else torch.zeros_like(k, dtype=torch.float64)
            out[start:stop] = torch.zeros(stop - start, dtype=torch.float64,
                                          device=u.device).scatter_add_(0, owner, hit)
            start = stop
        return out


class gMPNNRef(nn.Module):
    """jgMPNN (learn-2-gmpnn-ddi.py:45-80), hid > 1 path, evaluated at query pairs."""

    def __init__(self, hidden_channels=5):
        super().__init__()
        assert hidden_channels > 1, 'the reference hid == 1 path is a different recursion'
        self.hid = hidden_channels
        self.nn1 = nn.Linear(2, hidden_channels).double()
        self.nn2 = nn.Linear(2 * hidden_channels, 1).double()

    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()

    def _f1(self, g, u, v, plan, total):
        de2 = g.common(u, v, plan=plan, total=total).clamp(min=1)
        agg1 = (g.deg[u] + g.deg[v]).to(torch.float64) / (2 * de2)
        return F.relu(self.nn1(torch.stack([torch.ones_like(agg1), agg1], 1))), de2

    def forward(self, g, qi, qj):
        # Shapes are data-dependent; every size the step needs is fetched in two host
        # syncs (neighbour totals, then common-neighbour expansion totals).
        t_i, t_j = torch.stack([g.deg[qi].sum(), g.deg[qj].sum()]).tolist()
        o1, z1 = g.neighbours(qi, t_i)     # sum_{z in N(i)} f1[j, z]
        o2, z2 = g.neighbours(qj, t_j)     # sum_{z in N(j)} f1[i, z]
        pairs = [(qi, qj), (qj[o1], z1), (qi[o2], z2)]
        preps = [g.prep(u, v) for u, v in pairs]
        totals = torch.stack([tot for _, tot in preps]).tolist()
        (f1_ij, de2_ij), (f1_a, _), (f1_b, _) = [
            self._f1(g, u, v, plan, tot) for (u, v), (plan, _), tot in zip(pairs, preps, totals)]
        agg2 = torch.zeros(qi.numel(), self.hid, dtype=torch.float64, device=qi.device)
        agg2 = agg2.index_add(0, o1, f1_a).index_add(0, o2, f1_b)
        agg2 = agg2 / de2_ij.unsqueeze(1)
        return self.nn2(torch.cat([f1_ij, agg2], 1)).squeeze(1)


class LinkPredictorRef(nn.Module):
    """LinkPredictor (learn-2-gmpnn-ddi.py:82-105), in_channels = out_channels = 1."""

    def __init__(self, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()
        assert num_layers >= 2, 'the reference LinkPredictor always has an input and output layer'
        self.lins = nn.ModuleList([nn.Linear(1, hidden_channels)])
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 1))
        self.lins.double()
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def logit(self, x):
        x = x.unsqueeze(-1)
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x).squeeze(-1)

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def pair_state(model, g, pairs, budget=4_000_000):
    """Rounded f2 for [P, 2] pairs (no grad).

    Pairs are chunked by neighbourhood volume (deg i + deg j), not by count: a pair's
    working set is its endpoints' neighbour expansion, so a fixed pair count blew up
    to ~20 GiB on collab chunks full of high-degree endpoints. Each pair is computed
    independently, so chunking never changes a value.
    """
    if pairs.size(0) == 0:
        return torch.zeros(0, dtype=torch.float64, device=pairs.device)
    cum = torch.cumsum(g.deg[pairs[:, 0]] + g.deg[pairs[:, 1]] + 1, 0)
    cuts = torch.searchsorted(cum, torch.arange(budget, int(cum[-1]) + budget, budget,
                                                device=cum.device), right=True).tolist()
    out, start = [], 0
    for stop in cuts:
        stop = min(max(stop, start + 1), pairs.size(0))
        if stop <= start:
            continue
        c = pairs[start:stop]
        out.append(torch.round(model(g, c[:, 0], c[:, 1]) * ROUND) / ROUND)
        start = stop
    if start < pairs.size(0):
        c = pairs[start:]
        out.append(torch.round(model(g, c[:, 0], c[:, 1]) * ROUND) / ROUND)
    return torch.cat(out)


@torch.no_grad()
def score_sets(model, predictor, sets):
    """Score several pair sets consistently; sets = [(GraphIndex, pairs[..., 2]), ...]. All rounded states are pooled,
    each distinct one goes through the readout once (float64 logit). Returns CPU tensors shaped like each pairs tensor
    minus its last dim."""
    model.eval(); predictor.eval()
    states = [pair_state(model, g, p.reshape(-1, 2)) for g, p in sets]
    uniq, inv = torch.unique(torch.cat(states), return_inverse=True)
    scores = predictor.logit(uniq)[inv].cpu()
    out, s = [], 0
    for (_, p), st in zip(sets, states):
        out.append(scores[s:s + st.numel()].reshape(p.shape[:-1]))
        s += st.numel()
    return out


def train_epoch(model, predictor, g, train_pos, optimizer, batch_size, neg_fn, clip=None):
    """One epoch of -log(pos) - log(1 - neg) (learn-2-gmpnn-ddi.py); neg_fn(edge [2, b]) -> [2, b] negatives."""
    model.train(); predictor.train()
    total = n = 0
    for perm in torch.split(torch.randperm(train_pos.size(0), device=train_pos.device), batch_size):
        edge = train_pos[perm].t()
        neg = neg_fn(edge)
        optimizer.zero_grad()
        # one forward over positives and negatives together (pairs are independent, so
        # this equals two separate calls; the reference also scores all pairs per step)
        b = edge.size(1)
        out = predictor(model(g, torch.cat([edge[0], neg[0]]), torch.cat([edge[1], neg[1]])))
        pos_out, neg_out = out[:b], out[b:]
        loss = -torch.log(pos_out + 1e-15).mean() - torch.log(1 - neg_out + 1e-15).mean()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), clip)
        optimizer.step()
        total = total + loss.detach() * edge.size(1); n += edge.size(1)   # synced once, below
    return float(total) / max(n, 1)


# ------------------------------------------------------------------------------------------------ hooks
def build(ctx, input_channel):
    """gMPNNRef + LinkPredictorRef (readout width = hidden width), and the train / test-time graph indices."""
    hp = ctx.spec.hp
    model = gMPNNRef(hp['gmpnn_hidden_channels']).to(ctx.device)
    predictor = LinkPredictorRef(hp['gmpnn_hidden_channels'], hp['gmpnn_predictor_layers'], hp['dropout']).to(ctx.device)
    data = ctx.graph.data
    g_train = GraphIndex(to_edge_index(data.adj_t)[0], data.num_nodes)
    g_test = GraphIndex(to_edge_index(data.full_adj_t)[0], data.num_nodes) if ctx.graph.val_input_for_test else g_train
    ctx.extra['gmpnn_graphs'] = (g_train, g_test)
    model.reset_parameters()
    predictor.reset_parameters()
    return model, predictor


def evaluate(ctx):
    """(results, preds) in flex_www.evaluate's layouts; every set scored in one score_sets call."""
    g_train, g_test = ctx.extra['gmpnn_graphs']
    pos_train, pos_valid, neg_valid, pos_test, neg_test, neg_train = ctx.edges
    if ctx.split.layout != 'heart':              # flat (shared or per-query) negatives
        preds = score_sets(ctx.model, ctx.predictor, [(g_train, pos_train), (g_train, pos_valid), (g_train, neg_valid),
                                                      (g_test, pos_test), (g_test, neg_test)])
        return metrics(ctx.split.layout, preds, ctx.split.per_query), preds
    if neg_test.dim() == 2:
        neg_test = neg_test.unsqueeze(1)
    if neg_train.dim() == 2:
        neg_train = neg_train.unsqueeze(1)
    ptr, ntr, pv, nv, pt, nt = score_sets(ctx.model, ctx.predictor, [(g_train, pos_train), (g_train, neg_train),
                                                                    (g_train, pos_valid), (g_train, neg_valid),
                                                                    (g_test, pos_test), (g_test, neg_test)])
    preds = [ptr, pv, nv, pt, nt, ntr]
    return metrics('heart', preds), preds


def hooks(opts):
    def step(ctx, epoch, extra_loss_fn):
        n = ctx.split.num_nodes
        return train_epoch(ctx.model, ctx.predictor, ctx.extra['gmpnn_graphs'][0], ctx.edges[0], ctx.optimizer,
                           ctx.spec.hp['gmpnn_batch_size'], lambda e: typed_negatives(e, n, ctx.split.typed_neg_dst))
    return Hooks(build_models=build, train_step=step, evaluate=evaluate)
