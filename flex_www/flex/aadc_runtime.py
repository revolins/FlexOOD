"""AA-DC runtime glue: graph objects, scoring with the full model (progressive gate + L3 rescue with anchor scaling), the
validation-only calibration, and the per-cell baseline run whose calibration the FLEX +AD channel reuses.

Protocol
  graph      the message-passing graph the GCN sees for the split, unit weights
  calibrate  on VALIDATION only, on the train graph, with AA-DC's own calibrate_gate_thresholds / optimize_anchor_q; those
             consume a shared negative pool, so per-positive negatives (HeaRT, DBLP queries) are pooled for calibration
  evaluate   HeaRT / DBLP: each positive against its own negatives (Hits@20 / MRR); Amazon: OGB Hits@50 on shared negatives
  PA-up      on the Planetoid PA_0_50_100 cells test pairs are scored with validation edges added to the graph (as every
             method in those columns); calibration stays on the train graph
"""
import numpy as np
import torch

from . import aadc as AD
from ..data import AMAZON_CELLS, LPSHIFT_CELLS
from ..evaluate import evaluate_mrr, pooled_hits
from ..utils import init_seed

PARAMS = dict(ext_threshold=AD.EXT_THRESHOLD, ext_penalty=AD.EXT_PENALTY, anchor_scale=AD.ANCHOR_SCALE, anchor_q=AD.ANCHOR_Q)


def graph_objects(n, edges):
    e = np.unique(np.sort(np.asarray(edges, np.int64), 1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    A = AD.build_weighted_adj(n, e, np.ones(len(e), np.float32))
    inv = AD.precompute_inv_log_deg(A)
    return dict(A=A, A_invlog=AD.precompute_aa_matrix(A, inv), inv_log_deg=inv, adj=AD.build_adj(n, e),
                lcc=AD.compute_exact_lcc(A))


def score(G, edges, P=PARAMS, gate_thresholds=None, anchor_scale=None, gate_mode='progressive', collect_l3=False):
    y, aa, gate, ext, stats = AD.score_edges(
        edges=np.asarray(edges, np.int64).reshape(-1, 2), A=G['A'], A_invlog=G['A_invlog'], inv_log_deg=G['inv_log_deg'],
        adj=G['adj'], use_gate=True, gate_mode=gate_mode, ext_threshold=P['ext_threshold'], ext_penalty=P['ext_penalty'],
        lcc=G['lcc'], use_l3=True, rescue_mode='anchor', anchor_scale=P['anchor_scale'] if anchor_scale is None else anchor_scale,
        gate_thresholds=gate_thresholds, collect_l3=collect_l3, show_progress=False)
    return np.asarray(y, np.float64), np.asarray(aa, np.float64), ext, stats


def calibrate(G, pv, nv, P=PARAMS):
    """AA-DC's --auto_gate (progressive) + --auto_beta chain on validation and the train graph -> (gate thresholds or None,
    anchor scale, info)."""
    y_p, aa_p, ext_p, _ = score(G, pv, P, gate_mode='threshold')
    y_n, aa_n, ext_n, _ = score(G, nv, P, gate_mode='threshold')
    gth, note = None, 'gate: no AA>0 edges on one side, README defaults'
    pm, nm = aa_p > 0, aa_n > 0
    if pm.sum() > 0 and nm.sum() > 0:
        K = 50
        pi = np.where(pm)[0][np.argsort(y_p[pm])[-K:]]
        ni = np.where(nm)[0]
        ni = ni[np.argsort(y_n[nm])[-K:]] if len(ni) >= K else ni
        lcc = G['lcc']
        avg = lambda E: np.array([(lcc[u] + lcc[v]) / 2.0 for u, v in E], dtype=np.float32)  # noqa: E731
        gth = AD.calibrate_gate_thresholds(pos_ext_ratio=np.asarray(ext_p)[pi], pos_avg_lcc=avg(pv[pi]),
                                           neg_ext_ratio=np.asarray(ext_n)[ni], neg_avg_lcc=avg(nv[ni]),
                                           pos_protection_q=0.95, margin=0.10, neg_quantiles=(0.33, 0.67))
        note = 'gate calibrated'
    y_p, aa_p, _, st_p = score(G, pv, P, gth, anchor_scale=0.0, collect_l3=True)
    y_n, aa_n, _, st_n = score(G, nv, P, gth, anchor_scale=0.0, collect_l3=True)
    neg_aa = y_n[aa_n > 0]
    K50 = 50
    kth = (float(np.max(neg_aa)) if neg_aa.size else 0.0) if neg_aa.size < K50 else float(np.partition(neg_aa, -K50)[-K50])
    opt = AD.optimize_anchor_q(y_pos_base=y_p, y_neg_base=y_n, l3_pos_vals=st_p.get('l3_values', np.array([], np.float32)),
                               l3_neg_vals=st_n.get('l3_values', np.array([], np.float32)), kth_aa=kth, K=K50)
    return gth, float(opt['best_anchor_scale']), dict(note=note, best_q=opt.get('best_q'), kth50=kth)


def load_split(cell, seed=None, root=None):
    """(n, train graph edges [E, 2], (pos_v, neg_v), (pos_t, neg_t), metric, neg_mode) for the AA-DC run of a cell.
    Amazon with `seed`: that seed's own split (init_seed(seed) then the loader, as the trainer draws it); without a seed
    the split is drawn from the torch RNG as it stands (a fresh process gives the reference split)."""
    from ..data import load
    if cell in LPSHIFT_CELLS:
        s = load(cell, root=root)
        return (s.num_nodes, s.data.edge_index.cpu().numpy().T, (s.valid_pos.numpy(), s.valid_neg.numpy()),
                (s.test_pos.numpy(), s.test_neg.numpy()), 'Hits@20', 'per_pos')
    if cell == 'dblp':
        s = load('dblp', root=root)
        out = [(p.numpy(), q.numpy().reshape(len(p), -1, 2)) for p, q in ((s.valid_pos, s.valid_neg), (s.test_pos, s.test_neg))]
        return s.num_nodes, s.data.edge_index.numpy().T, out[0], out[1], 'MRR', 'per_pos'
    if cell in AMAZON_CELLS:
        from ..data.amazon import amazon_data
        if seed is not None:
            init_seed(seed)
        data, split = amazon_data(AMAZON_CELLS[cell][1], root)
        out = [(split[s]['edge'].numpy(), split[s]['edge_neg'].numpy()) for s in ('valid', 'test')]
        return int(data.num_nodes), split['train']['edge'].numpy(), out[0], out[1], 'Hits@50', 'shared'
    raise ValueError(cell)


def metric(name, pos, neg, mode):
    """Percent; per-positive -> each positive against its own row of neg, shared -> OGB Hits@K."""
    pos_t = torch.as_tensor(pos, dtype=torch.float32)
    if mode == 'per_pos':
        r = evaluate_mrr(pos_t, torch.as_tensor(neg, dtype=torch.float32).view(len(pos), -1))
        return 100 * float(r['MRR'] if name == 'MRR' else r[f"mrr_hit{name.split('@')[1]}"])
    return 100 * float(pooled_hits(pos_t, torch.as_tensor(neg, dtype=torch.float32), int(name.split('@')[1]), rounded=False))


def run_cell(cell, seed=None, root=None, P=PARAMS):
    """AA-DC on one cell: calibration (gate thresholds, anchor scale) on validation + valid / test metric."""
    n, tr, (pv, nv), (pt, nt), mname, mode = load_split(cell, seed, root)
    G = graph_objects(n, tr)
    nv_pool = nv.reshape(-1, 2) if mode == 'per_pos' else nv
    gth, s, info = calibrate(G, pv, nv_pool, P)
    yv, _, _, _ = score(G, pv, P, gth, s)
    yn, _, _, _ = score(G, nv_pool, P, gth, s)
    valid = metric(mname, yv, yn.reshape(nv.shape[:-1]) if mode == 'per_pos' else yn, mode)
    if cell.endswith('PA_0_50_100') and not cell.startswith('ogbl-collab'):
        G = graph_objects(n, np.vstack([tr, pv]))           # PA-up: validation edges join the test-time graph
    y_pt, _, _, _ = score(G, pt, P, gth, s)
    y_nt, _, _, _ = score(G, nt.reshape(-1, 2) if mode == 'per_pos' else nt, P, gth, s)
    test = metric(mname, y_pt, y_nt.reshape(nt.shape[:-1]) if mode == 'per_pos' else y_nt, mode)
    return dict(cell=cell, metric=mname, valid=valid, test=test, anchor_scale=s, gate_thresholds=gth, gate_status=info,
                neg_mode=mode, n=n, train_edges=int(len(tr)))


class AADCChannel:
    """AA-DC as a label-free FLEX channel: the cell's validation-only calibration (run_cell / its saved JSON), scores on
    whatever graph the other channels use."""
    def __init__(self, gate_thresholds, anchor_scale, P=PARAMS):
        self.gth, self.scale, self.P = gate_thresholds, float(anchor_scale), P
        self._G = {}

    def __call__(self, A, pairs):
        k = (A.shape[0], int(A.nnz), int(A.indices[:64].sum()) if A.nnz else 0)
        if k not in self._G:
            r, c = A.nonzero()
            m = r < c
            self._G = {k: graph_objects(A.shape[0], np.c_[r[m], c[m]])}     # one graph cached
        y, *_ = score(self._G[k], np.asarray(pairs, np.int64).reshape(-1, 2), self.P, self.gth, self.scale)
        return np.asarray(y, np.float64)
