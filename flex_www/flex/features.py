"""Year-free pair features, pseudo-labels from the train graph, shift-axis environments and the REL / ISS models.

Everything here is label-free with respect to validation and test: models are fit on PSEUDO-labels made from the train
graph only (K folds of the unique train pairs; each fold's pairs are pseudo-positives with features computed on the train
graph minus that fold, plus tail-corrupted pseudo-negatives). Evaluation pairs only enter as rows to be scored.

Columns (no node feature; computable on any undirected graph):
  local   cn aa ra jac, path counts lcn lw3 w3n lw4, 2-hop sizes n2min n2max, w3pos
  degree  dmin dmax dsum (log1p degree)
  walks   RW2 RW3 RW4 (log of the symmetrised random-walk probability)
  sp      conn spc inv_sp, only when `sp=True` (SP / gated rows): same component; distance class 2 (CN > 0), 3 (3-walk),
          4 (4-walk), 5 (reachable, >= 5 hops), 6 (disconnected); inv_sp = 1 / spc (0 when disconnected). Appended, so
          every other column keeps its index.
REL additionally reads the endpoint feature cosine and norms (x_feats).
"""
import functools
import os

import numpy as np
import scipy.sparse as ss

EPS = 1e-9
HOLD = 0.10            # share of unique train pairs held out as pseudo-positives
NPSEUDO = 20000        # cap on pseudo-positives
NEG_RATIO = 4          # tail-corrupted negatives per pseudo-positive
NANCHOR = 20000        # uniform non-edges used to standardise label-free channels
KFOLD = 5              # pseudo-label folds
MIN_ENV = 30           # rows an environment needs
BASE_COLS = ['cn', 'aa', 'ra', 'jac', 'lcn', 'lw3', 'w3n', 'lw4', 'n2min', 'n2max', 'w3pos', 'dmin', 'dmax', 'dsum',
             'RW2', 'RW3', 'RW4']
SP_COLS = ['conn', 'spc', 'inv_sp']


def cols(sp=False):
    return BASE_COLS + SP_COLS if sp else list(BASE_COLS)


def canon(e):
    """Unique undirected pairs (u < v), self-pairs removed."""
    e = np.sort(np.asarray(e, np.int64).reshape(-1, 2), 1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


def pk(e, n):
    e = np.sort(np.asarray(e, np.int64).reshape(-1, 2), 1)
    return e[:, 0] * n + e[:, 1]


def adj(edges, n):
    """Binary symmetric CSR adjacency without self-loops."""
    e = canon(edges)
    A = ss.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


# ------------------------------------------------------------------------------------------------------------ features
PF_WORKERS = int(os.environ.get('FLEX_PF_WORKERS', '8'))   # CPU workers for large pair sets (exact: rows are independent)
PF_MIN_PARALLEL = 50000
_PF_A = None


def _pf_piece(sp, P):
    return _pair_feats(_PF_A, P, sp=sp)


def pair_feats(A, P, sp=False, chunk=4000):
    """Feature rows for pairs P on graph A. Each row depends only on (A, its pair), so large P is computed once per unique
    pair, in pieces over forked workers, and reassembled in input order (identical to the serial computation)."""
    global _PF_A
    P = np.asarray(P, np.int64).reshape(-1, 2)
    if len(P) < PF_MIN_PARALLEL:
        return _pair_feats(A, P, chunk, sp)
    U, inv = np.unique(P, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    w = max(1, min(PF_WORKERS, len(U) // PF_MIN_PARALLEL + 1))
    if w == 1:
        return _pair_feats(A, U, chunk, sp)[inv]
    import multiprocessing as mp
    _PF_A = A                                     # inherited by the forked workers
    bounds = np.linspace(0, len(U), 4 * w + 1).astype(int)
    pieces = [U[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    try:
        with mp.get_context('fork').Pool(w) as pool:
            X = np.vstack(pool.map(functools.partial(_pf_piece, sp), pieces, chunksize=1))
    finally:
        _PF_A = None
    return X[inv]


def _pair_feats(A, P, chunk=4000, sp=False):
    P = np.asarray(P, np.int64).reshape(-1, 2)
    u, v = P[:, 0], P[:, 1]
    m = len(P)
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = ss.diags(1 / np.maximum(deg, 1))
    T = (Dinv @ A).tocsr()                       # row-normalised walk matrix
    ld2 = 1 / np.log(np.maximum(deg, 2))
    id1 = 1 / np.maximum(deg, 1)
    C = cols(sp)
    out = {k: np.zeros(m) for k in C}
    for s in range(0, m, chunk):
        b = slice(s, s + chunk)
        Au, Av = A[u[b]], A[v[b]]
        M = Au.multiply(Av)
        cn = np.asarray(M.sum(1)).ravel()
        out['cn'][b] = cn
        out['aa'][b] = np.asarray(M.multiply(ld2).sum(1)).ravel()
        out['ra'][b] = np.asarray(M.multiply(id1).sum(1)).ravel()
        out['jac'][b] = cn / np.maximum(deg[u[b]] + deg[v[b]] - cn, 1)
        Au2, Av2 = Au @ A, Av @ A
        w3 = np.asarray(Au2.multiply(Av).sum(1)).ravel()
        w4 = np.asarray(Au2.multiply(Av2).sum(1)).ravel()
        w3n = np.asarray(((Au @ Dinv) @ (A @ Dinv)).multiply(Av @ Dinv).sum(1)).ravel()
        n2u = np.diff((Au2 > 0).astype(np.int8).tocsr().indptr)
        n2v = np.diff((Av2 > 0).astype(np.int8).tocsr().indptr)
        out['lcn'][b] = np.log1p(cn)
        out['lw3'][b] = np.log1p(w3)
        out['w3n'][b] = w3n
        out['lw4'][b] = np.log1p(w4)
        out['n2min'][b] = np.log1p(np.minimum(n2u, n2v))
        out['n2max'][b] = np.log1p(np.maximum(n2u, n2v))
        out['w3pos'][b] = (w3 > 0)
        # walk probabilities (P^k)[u, v], symmetrised over u->v and v->u; P^4 uses reversibility
        # (P^2)[w, v] = deg_v / deg_w (P^2)[v, w]
        Tu, Tv = T[u[b]], T[v[b]]
        Tu2, Tv2 = Tu @ T, Tv @ T
        ii = np.arange(len(u[b]))
        p2uv = np.asarray(Tu2[ii, v[b]]).ravel()
        p2vu = np.asarray(Tv2[ii, u[b]]).ravel()
        AvD, AuD = Av @ Dinv, Au @ Dinv
        p3uv = np.asarray(Tu2.multiply(AvD).sum(1)).ravel()
        p3vu = np.asarray(Tv2.multiply(AuD).sum(1)).ravel()
        Tv2D, Tu2D = Tv2 @ Dinv, Tu2 @ Dinv
        p4uv = np.asarray(Tu2.multiply(Tv2D).sum(1)).ravel() * deg[v[b]]
        p4vu = np.asarray(Tv2.multiply(Tu2D).sum(1)).ravel() * deg[u[b]]
        out['RW2'][b] = np.log(EPS + .5 * (p2uv + p2vu))
        out['RW3'][b] = np.log(EPS + .5 * (p3uv + p3vu))
        out['RW4'][b] = np.log(EPS + .5 * (p4uv + p4vu))
    ld = np.log1p(deg)
    out['dmin'], out['dmax'] = np.minimum(ld[u], ld[v]), np.maximum(ld[u], ld[v])
    out['dsum'] = ld[u] + ld[v]
    if sp:
        from scipy.sparse.csgraph import connected_components
        lab = connected_components(A, directed=False)[1]
        conn = lab[u] == lab[v]
        spc = np.where(out['cn'] > 0, 2, np.where(out['lw3'] > 0, 3, np.where(out['lw4'] > 0, 4, np.where(conn, 5, 6))))
        out['conn'], out['spc'], out['inv_sp'] = conn.astype(np.float64), spc.astype(np.float64), np.where(spc == 6, 0.0, 1.0 / spc)
    return np.column_stack([out[k] for k in C]).astype(np.float32)


def x_feats(x, P):
    """Endpoint feature cosine and the two norms (zeros without features)."""
    if x is None:
        return np.zeros((len(P), 3), np.float32)
    P = np.asarray(P, np.int64).reshape(-1, 2)
    xu, xv = x[P[:, 0]], x[P[:, 1]]
    nu, nv = np.linalg.norm(xu, axis=1), np.linalg.norm(xv, axis=1)
    cos = (xu * xv).sum(1) / np.maximum(nu * nv, 1e-8)
    return np.column_stack([cos, np.minimum(nu, nv), np.maximum(nu, nv)]).astype(np.float32)


def act_t(A, P):
    """Activity proxy: -(log1p deg_u + log1p deg_v)."""
    deg = np.asarray(A.sum(1)).ravel()
    P = np.asarray(P, np.int64).reshape(-1, 2)
    return -(np.log1p(deg[P[:, 0]]) + np.log1p(deg[P[:, 1]]))


# ------------------------------------------------------------------------------------------------------------ pseudo labels
def tail_negatives(pos, n_neg, ban_keys, n, rng, tails=None, return_src=False):
    """(u, v') with u from the positives' heads and v' from their empirical tail distribution (keeps node types on
    bipartite graphs). return_src: also the index of the positive each negative's head came from."""
    tails = pos[:, 1] if tails is None else tails
    got = np.zeros((0, 2), np.int64)
    src = np.zeros(0, np.int64)
    tries = 0
    seen = set()
    while len(got) < n_neg and tries < 50:
        k = 2 * (n_neg - len(got)) + 100
        si = rng.integers(0, len(pos), k)
        c = np.c_[pos[si, 0], tails[rng.integers(0, len(tails), k)]]
        ok = (c[:, 0] != c[:, 1])
        c, si = c[ok], si[ok]
        ok = ~np.isin(pk(c, n), ban_keys)
        c, si = c[ok], si[ok]
        keep = []
        for j, key in enumerate(pk(c, n)):
            if key not in seen:
                seen.add(key)
                keep.append(j)
        got = np.vstack([got, np.sort(c[keep], 1)])
        src = np.r_[src, si[keep]]
        tries += 1
    i = rng.permutation(len(got))[:n_neg]
    return (got[i], src[i]) if return_src else got[i]


def pseudo_rows(train_pairs, n, seed, x=None, folds=KFOLD, sp=False):
    """K-fold pseudo-labels: up to NPSEUDO unique train pairs in `folds` disjoint folds; each fold's positives and their
    tail-corrupted negatives get features on the train graph MINUS that fold.
    Returns dict(Xp, Xn, Rp, Rn, hold, neg, dp0, dn0, nsrc, dist): dp0 / dn0 = degree of endpoint 0 on the fold graph,
    nsrc = row of the positive each negative was corrupted from, dist = exact hop distance (small graphs)."""
    rng = np.random.default_rng(seed)
    P = canon(train_pairs)
    ban = pk(P, n)
    S = P[rng.permutation(len(P))[:min(len(P), NPSEUDO)]]
    fold = np.arange(len(S)) % folds
    out = {k: [] for k in ('Xp', 'Xn', 'Rp', 'Rn', 'hold', 'neg', 'dp0', 'dn0', 'nsrc', 'dist')}
    for f in range(folds):
        hold = S[fold == f]
        if not len(hold):
            continue
        keep = P[~np.isin(ban, pk(hold, n))]
        A = adj(keep, n)
        deg = np.asarray(A.sum(1)).ravel()
        neg, src = tail_negatives(hold, NEG_RATIO * len(hold), ban, n, rng, return_src=True)
        Xp, Xn = pair_feats(A, hold, sp=sp), pair_feats(A, neg, sp=sp)
        out['nsrc'].append(src + sum(len(h) for h in out['hold']))
        out['Xp'].append(Xp)
        out['Xn'].append(Xn)
        out['Rp'].append(np.c_[Xp, x_feats(x, hold)])
        out['Rn'].append(np.c_[Xn, x_feats(x, neg)])
        out['hold'].append(hold)
        out['neg'].append(neg)
        out['dp0'].append(deg[hold[:, 0]])
        out['dn0'].append(deg[neg[:, 0]])
        out['dist'].append(hop_dist(A, hold))
    return {k: (np.vstack(v) if v[0].ndim == 2 else np.concatenate(v)) for k, v in out.items()}


def hop_dist(A, P, max_n=50000, cap=12):
    """Exact shortest-path hops (unreachable / > cap -> cap + 1); NaN on graphs above max_n nodes."""
    if A.shape[0] > max_n:
        return np.full(len(P), np.nan)
    from scipy.sparse.csgraph import dijkstra
    heads, inv = np.unique(P[:, 0], return_inverse=True)
    D = dijkstra(A, directed=False, unweighted=True, indices=heads, limit=cap)
    d = D[inv, P[:, 1]]
    return np.where(np.isfinite(d), d, cap + 1)


def uniform_anchor(A, n, rng, m=NANCHOR):
    """m uniform non-edges (the label-free reference every channel is standardised on)."""
    u = rng.integers(0, n, 3 * m)
    v = rng.integers(0, n, 3 * m)
    c = canon(np.c_[u, v][u != v])
    c = c[np.asarray(A[c[:, 0], c[:, 1]]).ravel() == 0]
    return c[rng.permutation(len(c))[:m]]


# ------------------------------------------------------------------------------------------------------------ environments
def envs_for(cell, Xp, dp0=None, dist=None, year=None, sp=False):
    """Environment id per pseudo POSITIVE along the cell's shift axis; each negative inherits the env of the positive it
    was corrupted from (same head), so the binning variable cannot separate the labels within an env.
      CN cells   common-neighbour class {0, 1, >=2}, fallback Adamic-Adar tertiles
      PA cells   tertiles of log1p du + log1p dv
      SP cells   distance class (with sp columns: the five classes incl. disconnected), fallbacks: hop-distance classes,
                 RW4 tertiles
      dblp       query-author degree tertiles (seniority)
      collab     tertiles of each train pair's latest edge year (real ogbl-collab: the shift axis is time)
    (Amazon cross-domain environments are the two domains, assigned by the caller.) Returns (env per positive, rule)."""
    c = {k: j for j, k in enumerate(cols(sp))}
    shift = cell.get('shift')
    tert = lambda v: np.digitize(v, np.unique(np.quantile(v, [1 / 3, 2 / 3])))  # noqa: E731
    ok = lambda e: len([k for k in np.unique(e) if (e == k).sum() >= max(MIN_ENV, 0.05 * len(e))]) >= 2  # noqa: E731
    if shift == 'PA':
        return tert(Xp[:, c['dsum']]), 'PA tertiles'
    if cell.get('base') == 'dblp' and not shift:
        return tert(np.log1p(dp0)), 'seniority tertiles'
    if cell.get('base') == 'ogbl-collab' and not shift:
        assert year is not None, 'real-world collab environments need train edge years'
        return tert(year.astype(np.float64)), 'year tertiles'
    if shift == 'CN':
        tries = [('CN classes', np.digitize(Xp[:, c['cn']], [0.5, 1.5])), ('CN collapsed -> AA tertiles', tert(Xp[:, c['aa']]))]
    elif shift == 'SP' and sp:
        tries = [('SP classes v2 (incl. disconnected)', np.unique(Xp[:, c['spc']], return_inverse=True)[1]),
                 ('SP classes', np.where(Xp[:, c['cn']] > 0, 0, np.where(Xp[:, c['w3pos']] > 0, 1, 2))),
                 ('SP collapsed -> RW4 tertiles', tert(Xp[:, c['RW4']]))]
    elif shift == 'SP':
        tries = [('SP classes', np.where(Xp[:, c['cn']] > 0, 0, np.where(Xp[:, c['w3pos']] > 0, 1, 2)))]
        if dist is not None and np.isfinite(dist).all():
            tries.append(('SP collapsed -> hop-distance classes',
                          np.unique(dist, return_inverse=True)[1] if len(np.unique(dist)) <= 3 else tert(dist)))
        tries.append(('SP collapsed -> RW4 tertiles', tert(Xp[:, c['RW4']])))
    else:
        raise ValueError(f'no environment rule for {cell}')
    for rule, e in tries:
        if ok(e):
            return e, rule
    return tries[-1][1], tries[-1][0] + ' (still collapsed)'


def env_data(Xp, Xn, ep, en, w_far=3.0):
    """Per-env (X standardised on that env's negatives, y, w) with weight w_far on CN = 0 positives; envs with fewer than
    MIN_ENV positives or negatives are dropped."""
    out, stats, info = [], [], []
    cn = BASE_COLS.index('cn')
    for e in sorted(set(ep.tolist())):
        xp, xn = Xp[ep == e], Xn[en == e]
        if len(xp) < MIN_ENV or len(xn) < MIN_ENV:
            continue
        mu, sd = xn.mean(0), xn.std(0) + 1e-6
        X = (np.vstack([xp, xn]) - mu) / sd
        y = np.r_[np.ones(len(xp)), np.zeros(len(xn))]
        w = np.where((y > 0.5) & (np.r_[xp[:, cn], xn[:, cn]] == 0), w_far, 1.0)
        out.append((X.astype(np.float32), y.astype(np.float32), w.astype(np.float32)))
        stats.append((mu, sd))
        info.append({'env': int(e), 'pos': int(len(xp)), 'neg': int(len(xn))})
    return out, stats, info


# ------------------------------------------------------------------------------------------------------------ models
def iss_select(data, seed, boot=50, C=0.05, keep_frac=0.8):
    """Invariant stability selection: L1 logistic fits on half-bootstraps of every env; a column is kept when it is selected
    in >= keep_frac of all fits and its sign agrees across envs. NOTE: liblinear seeds itself from the global NumPy RNG,
    so the caller fixes np.random's state (the paper's runs reached this point right after init_seed(last seed))."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    d = data[0][0].shape[1]
    sel = np.zeros(d)
    sgn = []
    nfit = 0
    for X, y, w in data:
        se = np.zeros(d)
        for b in range(boot):
            i = rng.choice(len(X), len(X) // 2, replace=False)
            m = LogisticRegression(penalty='l1', C=C, solver='liblinear', class_weight='balanced', max_iter=200).fit(X[i], y[i])
            cf = m.coef_.ravel()
            sel += cf != 0
            se += np.sign(cf)
            nfit += 1
        sgn.append(np.sign(se))
    S = np.array(sgn)
    consistent = (np.abs(S.sum(0)) == (S != 0).sum(0)) & ((S != 0).sum(0) == len(data))
    keep = (sel / nfit >= keep_frac) & consistent
    return keep, {'freq': (sel / nfit).tolist(), 'consistent': consistent.tolist()}


def train_mlp(data, seed, steps=1500, lr=1e-3, wd=1e-5, drop=0.1, bp=256, bn=256, hid=(64, 32)):
    """ERM MLP with per-env class-balanced BCE and a per-env bias kept out of prediction (re-seeds torch with `seed`)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    E = len(data)
    d = data[0][0].shape[1]
    L, k = [], d
    for h in hid:
        L += [nn.Linear(k, h), nn.ReLU(), nn.Dropout(drop)]
        k = h
    net = nn.Sequential(*L, nn.Linear(k, 1))
    bias = torch.zeros(E, requires_grad=True)
    opt = torch.optim.Adam([{'params': net.parameters(), 'weight_decay': wd}, {'params': [bias], 'weight_decay': 0.0}], lr=lr)
    for _ in range(steps):
        R = []
        Xs, ys, es, ws = [], [], [], []
        for e, (X, y, w) in enumerate(data):
            p, q = np.flatnonzero(y > .5), np.flatnonzero(y <= .5)
            i = np.r_[rng.choice(p, min(bp, len(p)), replace=False), rng.choice(q, min(bn, len(q)), replace=False)]
            Xs.append(X[i])
            ys.append(y[i])
            es.append(np.full(len(i), e))
            ws.append(w[i])
        Xb, yb, wb = (torch.as_tensor(np.concatenate(a) if a[0].ndim == 1 else np.vstack(a)) for a in (Xs, ys, ws))
        eb = torch.as_tensor(np.concatenate(es)).long()
        loss = Fn.binary_cross_entropy_with_logits(net(Xb).squeeze(1) + bias[eb], yb, reduction='none') * wb
        for e in range(E):
            me = eb == e
            pe, ne = me & (yb > .5), me & (yb <= .5)
            R.append(0.5 * loss[pe].sum() / wb[pe].sum().clamp_min(1e-6) + 0.5 * loss[ne].mean())
        obj = torch.stack(R).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
    net.eval()
    return net


def mlp_predict(net, X):
    import torch
    with torch.no_grad():
        return net(torch.as_tensor(np.asarray(X, np.float32))).squeeze(1).numpy().astype(np.float64)


class ISSModel:
    """Invariant stability selection across the shift-axis envs, then the ERM MLP on the kept columns (all columns when
    nothing is kept or there is a single env). Scoring standardises rows on the target graph's uniform non-edges."""
    def __init__(self, data, seed, select=True):
        if select:
            self.keep, self.info = iss_select(data, seed)
        else:
            self.keep, self.info = np.zeros(data[0][0].shape[1], bool), {'skipped': 'single environment'}
        self.fallback = not self.keep.any()
        if self.fallback:
            self.keep = np.ones(data[0][0].shape[1], bool)
        self.net = train_mlp([(X[:, self.keep], y, w) for X, y, w in data], seed)

    def score(self, X, X_anchor):
        mu, sd = X_anchor.mean(0), X_anchor.std(0) + 1e-6
        return mlp_predict(self.net, ((X - mu) / sd)[:, self.keep])


def fit_rel(Xp, Xn, seed):
    """REL: class-balanced logistic model on [pair features + x relation] fit on the pseudo-labels; returns its decision
    function."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=seed).fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))
