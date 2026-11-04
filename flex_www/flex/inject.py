"""FLEX INJ arm (RSTAR_ISS_INJ: the NCN / Refined-GAE + FLEX rows on DBLP).

A label-free proposer adds N links to the INPUT graph; the encoder is re-run on the injected graph at every captured
selection state (the in-memory encoder + predictor state of each seed's selected epoch), the channels are recomputed on
the injected graph, and the fusion picks are re-run; N in {0} + INJ_GRID (shares of the unique train pairs) is chosen on
validation with the same adoption rule against the un-injected fused row (N = 0).
  proposer   HistGradientBoosting on the train-graph pseudo-labels (pair features only)
  candidates sampled distance-2 pairs + random pairs between non-isolated nodes, minus existing edges and minus every
             evaluation pair (validation / test positives and negatives; their labels are never read)
"""
import numpy as np
import torch
from torch_sparse import SparseTensor

from . import direction as DIR
from . import features as F
from .channels import PSEUDO_SEED, chans_on
from .fusion import fuse, pick, run_picks
from ..data import Graph
from ..evaluate import evaluate

INJ_GRID = (0.01, 0.03, 0.10)


def fit_proposer(Xp, Xn, seed):
    """Class-balanced HGB on pseudo-labels."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sw = np.where(y > .5, 0.5 / len(Xp), 0.5 / len(Xn)) * len(y)
    m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    m.fit(X, y, sample_weight=sw)
    return m


def candidate_pool(A, n, rng, ban_keys, cap2=300000, n_rand=50000):
    """Distance-2 pairs (sampled to cap2; on large graphs only from 20000 sampled source nodes) plus random pairs between
    non-isolated nodes; existing edges and banned keys removed."""
    deg0 = np.asarray(A.sum(1)).ravel()
    if A.nnz > 200000:
        src = rng.choice(np.flatnonzero(deg0 > 0), min(20000, int((deg0 > 0).sum())), replace=False)
        A2 = (A[src] @ A).tocoo()
        row = src[A2.row]
        m = row != A2.col
        c2 = np.sort(np.c_[row[m], A2.col[m]].astype(np.int64), 1)
    else:
        A2 = (A @ A).tocoo()
        m = A2.row < A2.col
        c2 = np.c_[A2.row[m], A2.col[m]].astype(np.int64)
    if len(c2) > cap2:
        c2 = c2[rng.choice(len(c2), cap2, replace=False)]
    E = F.canon(np.c_[A.tocoo().row, A.tocoo().col])
    deg = np.asarray(A.sum(1)).ravel()
    nodes = np.flatnonzero(deg > 0)
    cr = np.c_[rng.choice(nodes, n_rand), rng.choice(nodes, n_rand)]
    C = F.canon(np.vstack([c2, cr]))
    ek, ck = F.pk(E, n), F.pk(C, n)
    return C[~np.isin(ck, ek) & ~np.isin(ck, ban_keys)]


def injected_adj(adj_t, inj):
    """adj_t plus the injected undirected links (weight 1; a weighted adj_t keeps its values)."""
    r, c, _ = adj_t.coo()
    ie = torch.as_tensor(inj.T, device=r.device)
    rr, cc = torch.cat([r, ie[0], ie[1]]), torch.cat([c, ie[1], ie[0]])
    v0 = adj_t.storage.value()
    if v0 is None:
        out = SparseTensor(row=rr, col=cc, value=torch.ones(len(rr), device=r.device), sparse_sizes=adj_t.sparse_sizes())
        return out.coalesce().bool().float().to_symmetric()
    return SparseTensor(row=rr, col=cc, value=torch.cat([v0.view(-1).float(), torch.ones(2 * ie.size(1), device=r.device)]),
                        sparse_sizes=adj_t.sparse_sizes()).coalesce()


def run_injection(mf, cell, CH, A, ctx, pairs, seeds, metrics, metric, best, g0, pinfo0, ch0, log=print):
    """best[s][m] = {'P', 'state': {'model', 'predictor'}, 'ep'}; ctx: an evaluation context (model / predictor objects,
    graph, split, edges, batch sizes) whose model is re-loaded with each state. Returns
    (final_ch {s: channels}, final_sel(s, m), info, g)."""
    assert DIR.spec(cell) is None, 'INJ is ported for the non-LPShift rows only (no direction filter)'
    assert getattr(ctx.graph.data, 'full_adj_t', None) is None, 'INJ with a separate test-time graph is not ported'
    n = A.shape[0]
    prop = fit_proposer(CH.Xp, CH.Xn, PSEUDO_SEED)
    ban = np.unique(np.concatenate([F.pk(p, n) for p in pairs]))
    C = candidate_pool(A, n, np.random.default_rng(PSEUDO_SEED + 2), ban)
    Xc = F.pair_feats(A, C, sp=CH.sp)
    order = np.argsort(-prop.predict_proba(Xc)[:, 1], kind='stable')
    coo = A.tocoo()
    ntr = len(F.canon(np.c_[coo.row, coo.col]))
    data = ctx.graph.data
    res = {0: {'g': g0, 'val': pinfo0['val_mean']}}
    store = {}
    for frac in INJ_GRID:
        N = min(int(frac * ntr), len(order))
        inj = C[order[:N]]
        assert not np.isin(F.pk(inj, n), ban).any(), 'injected link equals an evaluation pair'
        Ai = F.adj(np.vstack([np.c_[coo.row, coo.col], inj]), n)
        di = data.clone()
        di.adj_t = injected_adj(data.adj_t, inj)
        gi = Graph(data=di, edge_index=ctx.graph.edge_index, val_input_for_test=False)
        chi = chans_on(CH, Ai, None, pairs)
        P_inj = {}
        for s in seeds:
            P_inj[s] = {}
            for m in metrics:
                b = best[s][m]
                if b['ep'] in P_inj[s]:
                    P_inj[s][m] = P_inj[s][b['ep']]
                    continue
                ctx.model.load_state_dict(b['state']['model'])
                ctx.predictor.load_state_dict(b['state']['predictor'])
                with torch.no_grad():
                    _, P = evaluate(ctx.model, ctx.predictor, gi, ctx.split, ctx.edges, ctx.batch_sizes)
                P_inj[s][m] = P_inj[s][b['ep']] = P
        epP = {s: P_inj[s][metric] for s in seeds}
        g_f, p_f = run_picks(mf, seeds, epP, {s: chi for s in seeds}, metric, log)
        res[frac] = {'g': g_f, 'val': p_f['val_mean'], 'N': int(N), 'picks': p_f}
        store[frac] = (chi, P_inj)
        log('inj', frac, 'N', N, 'val', p_f['val_mean'])

    def seedvals(frac):
        if frac == 0:
            return [mf.value(fuse(best[s][metric]['P'], ch0, s, g0), metric, 1) for s in seeds]
        chi, P_inj = store[frac]
        return [mf.value(fuse(P_inj[s][metric], chi, s, res[frac]['g']), metric, 1) for s in seeds]
    V = {(f,): seedvals(f) for f in (0,) + INJ_GRID}
    choice, info = pick(V, (0,))
    frac = choice[0]
    log('inj choice', frac, info)
    out = {'by_frac': {str(k): {kk: (list(vv) if kk == 'g' else vv) for kk, vv in v.items()} for k, v in res.items()},
           'choice': frac, 'choice_info': info}
    if frac == 0:
        return {s: ch0 for s in seeds}, (lambda s, m: best[s][m]['P']), out, g0
    chi, P_inj = store[frac]
    return {s: chi for s in seeds}, (lambda s, m: P_inj[s][m]), out, res[frac]['g']
