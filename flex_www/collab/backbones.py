"""Non-GCN backbones on real ogbl-collab: NCN, Refined-GAE (RGAE) and gMPNN_ref, scored from their trained checkpoints.

Each backbone gives the same pair scores the GCN slot of FLEX uses (flex_www.collab.gcn_scores), on the backbone's own
model input (flex_www.data.build_graph with the OGB rule: the test-time input adds the validation edges):
  'val'         half B on the train graph                                       (R* scalar picks)
  'val_age10'   half B on the train graph minus pairs last seen <= 2007          (ISS weight pick)
  'test_age10'  test pairs on the test-time graph minus the age10 pairs          (the backbone's age10 row and its FLEX row)
Scores are sigmoid(logit) in float32 (the R* inputs); the logits are kept in float64 for the backbone's own age10 row
(a float32 sigmoid saturates at 1.0 and would tie pairs the backbone orders).

Protocol details that move numbers
  * RGAE trains on rows of year >= 2011 only: its message-passing graph is rebuilt from those rows (duplicate pairs'
    weights summed), as its recorded command did (--train_year_min 2011).
  * NCN's common-neighbour scores depend on the batch composition at the float level, so the test pairs are scored
    exactly as the trainer scored them: the official test positives, then the official negatives, as two calls in the
    official order with the training batch size, and only then mapped onto the canonical PT order.
  * NCN reads an adjacency at scoring time: the same graph its encoder ran on.
  * gMPNN_ref scores every pair set in one call (its readout runs once per distinct pair state) on graph indices of the
    dataset's train edges, train + validation edges, and the latter minus the age10 pairs.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from .config import config
from .data import N, pair_key
from .. import models
from ..data import build_graph
from ..evaluate import predict


def _split_for(c, year_min):
    """The collab Split, with the train rows and the message-passing graph restricted to years >= year_min."""
    if not year_min:
        return c.split
    import copy
    s = copy.copy(c.split)
    keep = torch.as_tensor(c.Y >= year_min)
    s.train_pos = c.split.train_pos[keep]
    ei, ew = to_undirected(s.train_pos.t(), torch.as_tensor(c.W[c.Y >= year_min]).float(), reduce='add')
    s.data = c.split.data.clone()
    s.data.edge_index, s.data.edge_weight = ei, ew.view(-1, 1)
    return s


def _without(A, keys):
    """A minus every entry whose (unordered) pair key is in `keys` (entries kept in their order, values kept)."""
    r, col, v = A.coo()
    k = (torch.minimum(r, col) * N + torch.maximum(r, col)).cpu().numpy()
    m = torch.as_tensor(~np.isin(k, keys), device=r.device)
    return SparseTensor(row=r[m], col=col[m], value=None if v is None else v[m], sparse_sizes=A.sparse_sizes())


def _scores(predictor, h, pairs, bs, adj):
    out = [predict(predictor, h, pairs[perm].t(), adj).cpu() for perm in DataLoader(range(pairs.size(0)), bs)]
    return torch.cat(out).flatten().double().numpy()


def _sig(z):
    return (1 / (1 + np.exp(-z))).astype(np.float32)


@torch.no_grad()
def score_backbone(c, method, ckpts, device):
    """{('val' | 'val_age10' | 'test_age10', seed): float32 sigmoid scores} and {seed: float64 test_age10 logits}."""
    cfg = config()['backbones'][method]
    graph = build_graph(_split_for(c, cfg['train_year_min']), device, use_valedges_as_input=True)
    data = graph.data
    _, drop = c.test_age10()
    drop_v = np.unique(pair_key(c.E)[~c.val_age10()])
    A_tr, A_u = data.adj_t, data.full_adj_t.to(device)
    A_e, A_ve = _without(A_u, drop), _without(A_tr, drop_v)
    PB = torch.as_tensor(c.PB, dtype=torch.long, device=device)
    tp, tn = c.split.test_pos.to(device), c.split.test_neg.to(device)
    k_off = np.r_[pair_key(c.Tp), pair_key(c.Tn)]
    order = np.argsort(k_off)
    to_pt = order[np.searchsorted(k_off[order], pair_key(c.PT))]
    assert np.array_equal(k_off[to_pt], pair_key(c.PT))
    bs = cfg['batch_size']
    out, logits = {}, {}
    for s, (mp, pp) in sorted(ckpts.items()):
        model, pred = models.build(method, data.x.size(1), cfg['hp'], device, num_nodes=c.split.num_nodes)
        model.load_state_dict(torch.load(mp, map_location=device), strict=True)
        pred.load_state_dict(torch.load(pp, map_location=device), strict=True)
        model.eval()
        pred.eval()
        needs = getattr(pred, 'needs_adj', False)
        for which, A in (('val', A_tr), ('val_age10', A_ve)):
            out[(which, s)] = _sig(_scores(pred, model(data.x, A), PB, bs, A))
        h = model(data.x, A_e)
        adj = A_e if needs else A_tr
        z = np.r_[_scores(pred, h, tp, bs, adj), _scores(pred, h, tn, bs, adj)][to_pt]
        out[('test_age10', s)], logits[s] = _sig(z), z
        del model, pred
    return out, logits


@torch.no_grad()
def score_gmpnn(c, ckpts, device):
    """{seed: float64 test_age10 logits} of gMPNN_ref (its age10 row)."""
    from ..baselines import gmpnn_ref as GM
    hp = config()['backbones']['gMPNN_ref']['hp']
    _, drop = c.test_age10()
    ei = c.split.data.edge_index
    ve = c.split.valid_pos.t()
    full = torch.cat([ei, ve, ve.flip(0)], 1)
    keep = torch.as_tensor(~np.isin(pair_key(full.t().numpy()), drop))
    g_tr, g_u, g_e = (GM.GraphIndex(e.to(device), N) for e in (ei, full, full[:, keep]))
    P = lambda p: torch.as_tensor(p, dtype=torch.long, device=device)  # noqa: E731
    out = {}
    for s, (mp, pp) in sorted(ckpts.items()):
        model = GM.gMPNNRef(hp['gmpnn_hidden_channels']).to(device)
        pred = GM.LinkPredictorRef(hp['gmpnn_hidden_channels'], hp['gmpnn_predictor_layers'], hp['dropout']).to(device)
        model.load_state_dict(torch.load(mp, map_location=device), strict=True)
        pred.load_state_dict(torch.load(pp, map_location=device), strict=True)
        _, _, te = GM.score_sets(model, pred, [(g_tr, P(c.PB)), (g_u, P(c.PT)), (g_e, P(c.PT))])
        out[s] = te.double().numpy()
    return out
