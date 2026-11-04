"""Label-free FLEX channels for evaluation pairs.

For a graph A and lists of pairs, every channel is standardised (z-score) on the same uniform non-edge anchors of A:
  act   activity proxy -(log1p du + log1p dv)
  rel   REL: logistic model on pair features + endpoint-feature relation, fit on train-graph pseudo-labels
  iss   ISS (one model per seed): invariant stability selection over shift-axis environments, then an MLP
  xc    endpoint raw-feature cosine
  ad    AA-DC score (only when an AADCChannel is attached)
Models are fit once, on pseudo-labels of the train graph; validation / test labels are never read.
"""
import numpy as np

from . import features as F

PSEUDO_SEED = 20261017


def graph_of(adj_t):
    """Binary symmetric CSR (no self-loops) of a torch_sparse adjacency."""
    r, c, _ = adj_t.coo()
    return F.adj(np.c_[r.cpu().numpy(), c.cpu().numpy()], int(adj_t.sparse_size(0)))


def eval_pairs(edges):
    """(pos_valid, neg_valid, pos_test, neg_test) of an evaluation edge list as (k, 2) int arrays."""
    return [t.detach().cpu().numpy().reshape(-1, 2) for t in (edges[1], edges[2], edges[3], edges[4])]


class Channels:
    """Channel models of one train graph. cell: flex_www.data.cell_info(...); x: node features as the model sees them;
    seeds: one ISS model per seed. source_pseudo: callable -> (Xp, Xn) pseudo rows of the source domain (Amazon
    cross-domain: the two domains are the ISS environments). sp: SP columns in the pair features (SP / gated rows)."""

    def __init__(self, cell, A_eval, x, train_pairs, seeds, log=print, year_of=None, sp=False, source_pseudo=None):
        self.cell, self.x, self.log, self.sp = cell, x, log, sp
        self.n = n = A_eval.shape[0]
        R = F.pseudo_rows(train_pairs, n, PSEUDO_SEED, x, sp=sp)
        Xp, Xn, Rp, Rn = R['Xp'], R['Xn'], R['Rp'], R['Rn']
        if source_pseudo is not None:
            Sp, Sn = source_pseudo()          # drawn here, after the target pseudo rows, as in the paper's runs
            ep = np.r_[np.zeros(len(Xp), int), np.ones(len(Sp), int)]
            en = np.r_[np.zeros(len(Xn), int), np.ones(len(Sn), int)]
            self.env_rule = 'domains'
            Ip, In = np.vstack([Xp, Sp]), np.vstack([Xn, Sn])
        else:
            yr = year_of(R['hold']) if year_of is not None else None
            ep, self.env_rule = F.envs_for(cell, Xp, R['dp0'], R['dist'], yr, sp=sp)
            en = ep[R['nsrc']]
            Ip, In = Xp, Xn
        self.env_data, _, self.env_info = F.env_data(Ip, In, ep, en)
        log('pseudo', {'hold': len(R['hold']), 'neg': len(R['neg']), 'env_rule': self.env_rule, 'envs': self.env_info})
        if len(self.env_data) < 2:            # ISS needs >= 2 envs: ERM MLP on all columns instead
            self.env_data, _, self.env_info = F.env_data(Ip, In, np.zeros(len(Ip), int), np.zeros(len(In), int))
            self.env_rule += ' -> single env: ERM V-hat'
        self.rel = F.fit_rel(Rp, Rn, PSEUDO_SEED)
        self.iss = {s: F.ISSModel(self.env_data, seed=s, select=len(self.env_data) >= 2) for s in seeds}
        k0 = self.iss[seeds[0]]
        self.iss_keep = [c for c, k in zip(F.cols(sp), k0.keep) if k]
        self.iss_fallback = k0.fallback
        log('ISS keeps', self.iss_keep, 'fallback' if self.iss_fallback else '')
        self.Xp, self.Xn = Xp, Xn
        self.aadc = None
        self._pf_cache = {}

    def pf(self, A, P):
        """pair_feats memoised on (graph, pairs); the last two entries are kept."""
        P = np.asarray(P, np.int64).reshape(-1, 2)
        key = (id(A), A.nnz, P.shape, hash(P.tobytes()))
        if key not in self._pf_cache:
            if len(self._pf_cache) >= 2:
                self._pf_cache.pop(next(iter(self._pf_cache)))
            self._pf_cache[key] = F.pair_feats(A, P, sp=self.sp)
        return self._pf_cache[key]

    def on_graph(self, A, pairs):
        """pairs: list of (k, 2) arrays -> one dict of channels per list (iss: {seed: scores})."""
        rng = np.random.default_rng(PSEUDO_SEED + 1)
        anc = F.uniform_anchor(A, self.n, rng)
        Xa = self.pf(A, anc)
        Ra = np.c_[Xa, F.x_feats(self.x, anc)]
        act_a, rel_a = F.act_t(A, anc), self.rel(Ra)
        z = lambda v, ref: (v - ref.mean()) / (ref.std() + 1e-9)  # noqa: E731
        iss_a = {s: m.score(Xa, Xa) for s, m in self.iss.items()}
        allP = np.vstack(pairs)
        X = self.pf(A, allP)
        R = np.c_[X, F.x_feats(self.x, allP)]
        ncol = len(F.cols(self.sp))
        act, rel = z(F.act_t(A, allP), act_a), z(self.rel(R), rel_a)
        xc = z(R[:, ncol].astype(np.float64), Ra[:, ncol].astype(np.float64))      # feature cosine (x_feats column 0)
        iss = {s: z(m.score(X, Xa), iss_a[s]) for s, m in self.iss.items()}
        ad = z(self.aadc(A, allP), self.aadc(A, anc)) if self.aadc is not None else None
        out, o = [], 0
        for p in pairs:
            sl = slice(o, o + len(p))
            o += len(p)
            out.append({'act': act[sl], 'rel': rel[sl], 'iss': {s: v[sl] for s, v in iss.items()}, 'xc': xc[sl]})
            if ad is not None:
                out[-1]['ad'] = ad[sl]
        return out


def chans_on(CH, A, A_full, pairs):
    """Channels for (pos_valid, neg_valid, pos_test, neg_test): validation pairs on the train graph; test pairs on A_full
    when the cell scores test on train + validation edges (VALIN), else everything on A."""
    if A_full is None:
        return CH.on_graph(A, pairs)
    return CH.on_graph(A, pairs[:2]) + CH.on_graph(A_full, pairs[2:])


def source_domain_pseudo(target, root=None):
    """Amazon cross-domain: pseudo rows of the SOURCE domain graph. Its RandomLinkSplit is drawn from the global torch RNG
    as it stands at the call (the paper's runs called this right after the previous seed's ISS MLP fit)."""
    from ..data.amazon import amazon_data
    src = {'Computers': 'Photo', 'Photo': 'Computers'}[target]
    data, split = amazon_data(src, root)
    R = F.pseudo_rows(split['train']['edge'].numpy(), int(data.num_nodes), PSEUDO_SEED + 7)
    return R['Xp'], R['Xn']
