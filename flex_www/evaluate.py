"""Scoring and metrics.

Prediction lists: every evaluation produces the list the metric function reads,
    heart layout       [pos_train, pos_valid, neg_valid, pos_test, neg_test, neg_train]   (neg_* are [P, K])
    pooled / per_query [pos_train, pos_valid, neg_valid, pos_test, neg_test]
(all CPU float tensors). `metrics(layout, preds, per_query)` maps such a list to {metric: (train, valid, test)}, with every
value rounded to 4 decimals exactly as in the paper's runs (except AUC on the heart layout, which was not rounded).

Metric protocols
  heart      each positive is ranked against its OWN negatives; rank = 1 + (#neg > pos + #neg >= pos) / 2 (ties split);
             Hits@K = mean(rank <= K), MRR = mean(1 / rank). The train column uses one random negative per train positive.
  pooled     OGB Hits@K: a positive is a hit if it scores strictly above the K-th highest of the shared negatives.
  per_query  DBLP: heart-style ranking of each positive against its own block of `per_query` negatives (MRR and Hits@K for
             K <= per_query); the train column is a 0.0 placeholder.
"""
import numpy as np
import torch
from ogb.linkproppred import Evaluator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

K_LIST_HEART = [1, 3, 10, 20, 50, 100]
K_LIST_POOLED = [20, 50, 100]


# ------------------------------------------------------------------------------------------------ ranking metrics
def eval_mrr(y_pred_pos, y_pred_neg):
    """Per-positive rank lists. y_pred_pos [P], y_pred_neg [P, K]."""
    y_pred_pos = y_pred_pos.view(-1, 1)
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    out = {f'hits@{k}_list': (ranking_list <= k).to(torch.float) for k in (1, 3, 20, 50, 10, 100)}
    out['mrr_list'] = 1. / ranking_list.to(torch.float)
    return out


def rank_metric(pos, neg, metric):
    """One heart-layout metric ('MRR' or 'Hits@K') of pos [P] against neg [P, K] (rounded as in the full suite)."""
    r = eval_mrr(pos, neg.view(pos.shape[0], -1))
    v = r['mrr_list'] if metric == 'MRR' else r[f"hits@{int(metric.split('@')[1])}_list"]
    return round(v.mean().item(), 4)


def evaluate_mrr(pos, neg):
    """MRR and per-positive Hits@K (K in K_LIST_HEART) of pos against its own row of neg."""
    out = eval_mrr(pos, neg.view(pos.shape[0], -1))
    res = {'MRR': round(out['mrr_list'].mean().item(), 4)}
    for k in K_LIST_HEART:
        res[f'mrr_hit{k}'] = round(out[f'hits@{k}_list'].mean().item(), 4)
    return res


def compute_auc(pos_pred, neg_pred):
    all_pred = torch.cat([pos_pred, neg_pred.flatten()]).cpu().numpy()
    all_true = np.concatenate([np.ones(pos_pred.size(0), dtype=int), np.zeros(neg_pred.numel(), dtype=int)])
    try:
        return roc_auc_score(all_true, all_pred)
    except ValueError:
        return 0.5


def _ap(pos, neg):
    pred = torch.cat([pos, neg.flatten()]).detach().cpu().numpy()
    true = np.concatenate([np.ones(pos.size(0), dtype=int), np.zeros(neg.numel(), dtype=int)])
    return round(float(average_precision_score(true, pred)), 4)


def heart_metrics(pos_train, pos_val, neg_val, pos_test, neg_test, neg_train):
    """Full metric suite of the heart layout (per-positive ranking + AP + unrounded AUC)."""
    tr, va, te = evaluate_mrr(pos_train, neg_train), evaluate_mrr(pos_val, neg_val), evaluate_mrr(pos_test, neg_test)
    result = {'MRR': (tr['MRR'], va['MRR'], te['MRR'])}
    for k in K_LIST_HEART:
        result[f'Hits@{k}'] = (tr[f'mrr_hit{k}'], va[f'mrr_hit{k}'], te[f'mrr_hit{k}'])
    result['AP'] = (_ap(pos_train, neg_train), _ap(pos_val, neg_val), _ap(pos_test, neg_test))
    try:
        result['AUC'] = (compute_auc(pos_train, neg_train), compute_auc(pos_val, neg_val), compute_auc(pos_test, neg_test))
    except Exception:
        result['AUC'] = (0.5, 0.5, 0.5)
    return result


_HITS_EVAL = None


def _hits_evaluator():
    global _HITS_EVAL
    if _HITS_EVAL is None:
        _HITS_EVAL = Evaluator(name='ogbl-collab')
    return _HITS_EVAL


def pooled_hits(pos, neg, k, rounded=True):
    """OGB Hits@K against a shared negative pool."""
    ev = _hits_evaluator()
    ev.K = k
    v = ev.eval({'y_pred_pos': pos, 'y_pred_neg': neg})[f'hits@{k}']
    return round(v, 4) if rounded else v


def _auc_ap(pred, true):
    return round(roc_auc_score(true, pred), 4), round(average_precision_score(true, pred), 4)


def per_query_metrics(pos_val, neg_val, pos_test, neg_test, n):
    out = {}
    v = _mrr_lists(pos_val, neg_val.view(pos_val.size(0), n))
    t = _mrr_lists(pos_test, neg_test.view(pos_test.size(0), n))
    out['MRR'] = (0.0, v['MRR'], t['MRR'])                  # no per-query train block: 0.0 placeholder
    for k in (1, 3, 10, 20):
        if k >= n + 1:
            continue
        out[f'Hits@{k}'] = (0.0, v[f'hits@{k}'], t[f'hits@{k}'])
    auc, ap = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for i, (pp, nn_) in ((1, (pos_val, neg_val)), (2, (pos_test, neg_test))):
        pred = torch.cat([pp, nn_.reshape(-1)])
        true = torch.cat([torch.ones(pp.size(0), dtype=int), torch.zeros(nn_.numel(), dtype=int)])
        auc[i], ap[i] = _auc_ap(pred, true)
    out['AUC'], out['AP'] = tuple(auc), tuple(ap)
    return out


def _mrr_lists(pos, neg):
    r = eval_mrr(pos, neg)
    d = {'MRR': round(r['mrr_list'].mean().item(), 4)}
    for k in (1, 3, 10, 20, 50, 100):
        d[f'hits@{k}'] = round(r[f'hits@{k}_list'].mean().item(), 4)
    return d


def pooled_metrics(pos_train, pos_val, neg_val, pos_test, neg_test, per_query=0):
    """Metric suite of the shared-negative layouts (pooled OGB Hits@K, or per-query ranking when per_query > 0)."""
    if per_query:
        n = per_query
        assert neg_val.numel() == pos_val.numel() * n and neg_test.numel() == pos_test.numel() * n, \
            'per-query ranking needs per_query negatives for every positive'
        return per_query_metrics(pos_val, neg_val, pos_test, neg_test, n)
    result = {}
    for k in K_LIST_POOLED:
        # the train column ranks train positives against the VALIDATION negatives
        result[f'Hits@{k}'] = (pooled_hits(pos_train, neg_val, k), pooled_hits(pos_val, neg_val, k), pooled_hits(pos_test, neg_test, k))
    ones = lambda t: torch.ones(t.size(0), dtype=int)  # noqa: E731
    zeros = lambda t: torch.zeros(t.size(0), dtype=int)  # noqa: E731
    tr = _auc_ap(torch.cat([pos_train, neg_val]), torch.cat([ones(pos_train), zeros(neg_val)]))
    va = _auc_ap(torch.cat([pos_val, neg_val]), torch.cat([ones(pos_val), zeros(neg_val)]))
    te = _auc_ap(torch.cat([pos_test, neg_test]), torch.cat([ones(pos_test), zeros(neg_test)]))
    result['AUC'] = (tr[0], va[0], te[0])
    result['AP'] = (tr[1], va[1], te[1])
    return result


def metrics(layout, preds, per_query=0):
    """{metric: (train, valid, test)} of a prediction list (see module docstring)."""
    if layout == 'heart':
        return heart_metrics(*preds)
    return pooled_metrics(*preds[:5], per_query=per_query if layout == 'per_query' else 0)


def metric_value(layout, preds, metric, split=1, per_query=0):
    """One metric on one split (1 = valid, 2 = test), bit-identical to metrics(...)[metric][split] but computing only it."""
    pos, neg = (preds[1], preds[2]) if split == 1 else (preds[3], preds[4])
    if layout == 'heart' and (metric == 'MRR' or metric.startswith('Hits@')):
        return rank_metric(pos, neg, metric)
    if layout == 'per_query' and (metric == 'MRR' or metric.startswith('Hits@')):
        k = per_query
        assert neg.numel() == pos.numel() * k
        r = _mrr_lists(pos, neg.view(pos.size(0), k))
        return r['MRR'] if metric == 'MRR' else r[f"hits@{int(metric.split('@')[1])}"]
    if layout == 'pooled' and metric.startswith('Hits@'):
        return pooled_hits(pos, neg, int(metric.split('@')[1]))
    return float(metrics(layout, preds, per_query)[metric][split])


# ------------------------------------------------------------------------------------------------ scoring
def predict(predictor, h, edge, adj=None):
    """Score edges [2, ...]: Hadamard predictors take endpoint embeddings; NCN (`needs_adj`) takes (h, adj, edges)."""
    if not getattr(predictor, 'needs_adj', False):
        return predictor(x_i=h[edge[0]], x_j=h[edge[1]])
    if adj is None:
        raise ValueError('predictor requires an adjacency')
    flat = edge.reshape(2, -1)
    out = predictor(h, adj, flat)
    lead = tuple(edge.shape[1:])
    return out.reshape(*lead, *out.shape[1:]) if len(lead) > 1 else out


def _score_heart(predictor, pos, h, batch_size, neg=None, adj=None):
    """Scores of pos [P, 2] and, per positive, of its negatives neg [P, K, 2] (batched over positives)."""
    pos_preds, neg_preds = [], []
    for perm in DataLoader(range(pos.size(0)), batch_size):
        pos_preds += [predict(predictor, h, pos[perm].t(), adj).cpu()]
        if neg is not None:
            neg_preds += [predict(predictor, h, torch.permute(neg[perm], (2, 0, 1)), adj).cpu()]
    return torch.cat(pos_preds, dim=0), (torch.cat(neg_preds, dim=0) if neg is not None else None)


def _score_pairs(predictor, pairs, h, batch_size, adj=None):
    if pairs.dim() == 3:
        pairs = pairs.view(-1, 2)
    return torch.cat([predict(predictor, h, pairs[perm].t(), adj).cpu()
                      for perm in DataLoader(range(pairs.size(0)), batch_size)], dim=0)


def _encode(model, x, adj_t):
    h = model(x, adj_t)
    return h[0] if isinstance(h, tuple) else h        # DANN returns (h, domain logits)


@torch.no_grad()
def predict_heart(model, predictor, graph, edges, batch_size, val_batch_size, test_batch_size):
    """Heart-layout prediction list. Validation and train pairs are scored on the train graph; test pairs on the train
    graph, or on train + validation edges when the graph was built with val_edges_test_input (PA-up VALIN cells)."""
    model.eval()
    predictor.eval()
    data = graph.data
    pos_train, pos_valid, neg_valid, pos_test, neg_test, neg_train = edges
    x = data.x
    h = _encode(model, x, data.adj_t.to(x.device))
    dev = x.device
    pos_valid_pred, neg_valid_pred = _score_heart(predictor, pos_valid.to(dev), h, val_batch_size, neg_valid.to(dev), data.adj_t)
    if graph.val_input_for_test:
        h_t = _encode(model, x, data.full_adj_t.to(dev))
        pos_test_pred, neg_test_pred = _score_heart(predictor, pos_test.to(dev), h_t, test_batch_size, neg_test.to(dev), data.full_adj_t)
    else:
        pos_test_pred, neg_test_pred = _score_heart(predictor, pos_test.to(dev), h, test_batch_size, neg_test.to(dev), data.adj_t)
    neg_train = neg_train.to(dev)
    if neg_train.dim() == 2:
        neg_train = neg_train.unsqueeze(1)
    pos_train_pred, neg_train_pred = _score_heart(predictor, pos_train.to(dev), h, batch_size, neg_train, data.adj_t)
    return [torch.flatten(pos_train_pred), torch.flatten(pos_valid_pred), neg_valid_pred.squeeze(-1),
            torch.flatten(pos_test_pred), neg_test_pred.squeeze(-1), neg_train_pred.squeeze(-1)]


@torch.no_grad()
def predict_pooled(model, predictor, graph, edges, batch_size, use_valedges_as_input=False):
    """Shared-negative prediction list (all pairs in batches of the TRAINING batch size, as the paper's runs did)."""
    model.eval()
    predictor.eval()
    data = graph.data
    train_pos, pos_valid, neg_valid, pos_test, neg_test = edges[:5]
    x = data.x
    dev = x.device
    h = _encode(model, x, data.adj_t.to(dev))
    neg_valid_pred = _score_pairs(predictor, neg_valid.to(dev), h, batch_size, data.adj_t)
    pos_valid_pred = _score_pairs(predictor, pos_valid.to(dev), h, batch_size, data.adj_t)
    if use_valedges_as_input:
        h = _encode(model, x, data.full_adj_t.to(dev))
    test_adj = data.full_adj_t.to(dev) if (use_valedges_as_input and getattr(predictor, 'needs_adj', False)) else data.adj_t
    pos_test_pred = _score_pairs(predictor, pos_test.to(dev), h, batch_size, test_adj)
    neg_test_pred = _score_pairs(predictor, neg_test.to(dev), h, batch_size, test_adj)
    pos_train_pred = _score_pairs(predictor, train_pos.to(dev), h, batch_size, data.adj_t)
    return [torch.flatten(pos_train_pred), torch.flatten(pos_valid_pred), torch.flatten(neg_valid_pred),
            torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)]


def evaluate(model, predictor, graph, split, edges, batch_sizes, use_valedges_as_input=False):
    """(results, preds) of one evaluation. batch_sizes = (train, valid, test) after the trainer's halving loops."""
    bs, vbs, tbs = batch_sizes
    if split.layout == 'heart':
        preds = predict_heart(model, predictor, graph, edges, bs, vbs, tbs)
    else:
        preds = predict_pooled(model, predictor, graph, edges, bs, use_valedges_as_input)
    return metrics(split.layout, preds, split.per_query), preds


def eval_once(model, predictor, graph, split, edges, batch_sizes, model_state, predictor_state, use_valedges_as_input=False):
    """Load encoder + predictor states and evaluate once: the same (results, preds) the trainer produced at the epoch the
    states were saved. States are file paths or state dicts."""
    dev = graph.data.x.device
    for mod, st in ((model, model_state), (predictor, predictor_state)):
        if not isinstance(st, dict):
            st = torch.load(st, map_location=dev)
        mod.load_state_dict(st)
    return evaluate(model, predictor, graph, split, edges, batch_sizes, use_valedges_as_input)
