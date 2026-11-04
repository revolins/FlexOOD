"""GCN pair scores on the collab graphs FLEX needs (trained checkpoints, inference only).

Every score is sigmoid(float64 logit) stored as float32, from the 3-layer GCN + 3-layer Hadamard MLP of the GCN (ERM) row
(trained by flex_www.train with the recorded collab config). The graph the encoder runs on decides what the score means:

  'test_age10'   test-time input (train + validation edges, co-authorship weights kept, validation edges weight 1) minus
                 the age10 pairs; this is the GCN slot of every real-collab row (the "+age10" in GCN+age10)
  'test'         the same graph without the edit (the GCN alone)
  'val'          validation-time input: the train graph, binary (validation half-B pairs)
  'val_age10'    the validation-time graph minus pairs last seen <= 2007, binary
"""
import numpy as np
import torch
from torch_sparse import SparseTensor

from .config import config
from .data import N, canon, canon_keep_dups, pair_key


def binary_adj(edges, device):
    """Binary symmetric adjacency of an edge list."""
    e = np.vstack([edges, edges[:, ::-1]])
    ei = torch.as_tensor(e.T, dtype=torch.long, device=device)
    return SparseTensor.from_edge_index(ei, sparse_sizes=(N, N)).coalesce().bool().float().to_symmetric()


def weighted_test_adj(c, device, age10=True):
    """Weighted test-time adjacency: the dataset's train edge_index / edge_weight (both directions) minus the age10 pairs,
    plus the validation edges (both directions, weight 1). Duplicate entries are kept, as in the original runs."""
    ei, ew = c.data.edge_index, c.data.edge_weight
    if age10:
        _, drop = c.test_age10()
        keep = torch.as_tensor(~np.isin(pair_key(ei.numpy().T), drop))
        ei, ew = ei[:, keep], ew[keep]
    ve = torch.as_tensor(canon_keep_dups(c.Ev), dtype=torch.long, device=device).T
    ve = torch.cat([ve, ve.flip(0)], dim=1)
    ei = torch.cat([ei.to(device), ve], dim=1)
    w = torch.cat([ew.view(-1).to(device=device, dtype=torch.float32), torch.ones(ve.size(1), dtype=torch.float32, device=device)])
    return SparseTensor.from_edge_index(ei, w, sparse_sizes=(N, N))


def graph(c, which, device):
    if which == 'test_age10':
        return weighted_test_adj(c, device, age10=True)
    if which == 'test':
        return weighted_test_adj(c, device, age10=False)
    if which == 'val':
        return binary_adj(canon(c.E), device)
    if which == 'val_age10':
        return binary_adj(canon(c.E[c.val_age10()]), device)
    raise ValueError(which)


def load_gcn(model_pt, predictor_pt, device, in_channels=128, ood_method=None, hp=None):
    """Encoder + predictor of a recorded collab run (hp: the GCN hyper-parameters of configs/collab/collab.yaml; every
    baseline shares them)."""
    from ..models.gcn import unGCN
    from ..models.predictor import mlp_score
    hp = hp or config()['gcn']['hp']
    h = hp['hidden_channels']
    state = torch.load(model_pt, map_location='cpu')
    if ood_method == 'VGAE':
        # at inference the VGAE encoder returns its mean embedding, i.e. exactly the unGCN forward; its log-std head is unused
        state = {k: v for k, v in state.items() if not k.startswith('logstd_conv.')}
        ood_method = None
    model = unGCN(in_channels, h, h, hp['num_layers'], hp['dropout'], hp['edge_drop'], ood_method)
    predictor = mlp_score(h, h, 1, hp['num_layers_predictor'], hp['dropout'])
    model.load_state_dict(state, strict=True)
    predictor.load_state_dict(torch.load(predictor_pt, map_location='cpu'), strict=True)
    return model.to(device).eval(), predictor.to(device).eval()


@torch.no_grad()
def pair_scores(model, predictor, x, adj, pairs, batch=65536):
    """sigmoid(logit) of pairs [P, 2] as float32 (logits in float32, sigmoid in float64)."""
    h = model(x, adj)
    h = h[0] if isinstance(h, tuple) else h
    out = []
    for i in range(0, len(pairs), batch):
        q = torch.as_tensor(pairs[i:i + batch], dtype=torch.long, device=x.device)
        out.append(predictor(x_i=h[q[:, 0]], x_j=h[q[:, 1]], logits=True).view(-1).cpu().numpy())
    z = np.concatenate(out).astype(np.float32)
    return (1 / (1 + np.exp(-np.asarray(z, np.float64)))).astype(np.float32)


def score_checkpoints(c, ckpts, jobs, device, ood_method=None):
    """{(which, seed): scores}. ckpts: {seed: (model_pt, predictor_pt)}; jobs: [(which graph, pairs name 'PT' | 'PB')]."""
    x = torch.as_tensor(c.x, device=device)
    out = {}
    for which, pname in jobs:
        adj = graph(c, which, device)
        pairs = getattr(c, pname)
        for s, (mp, pp) in sorted(ckpts.items()):
            model, pred = load_gcn(mp, pp, device, c.x.shape[1], ood_method)
            out[(which, s)] = pair_scores(model, pred, x, adj, pairs)
            del model, pred
        del adj
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return out
