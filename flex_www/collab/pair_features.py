"""Label-free pair descriptors on a dated collab graph.

b6_features   the inputs of R*'s residual model ("B6"): 14 'semantic' endpoint / temporal columns (feature cosine, norms,
              degrees, common-neighbour and 3-walk counts, pair-in-graph flag, endpoint row counts and last years) and 18
              'rrwp' random-walk return probabilities (1-3 steps, both directions, from exact CSR walk counts)
xfree         the 31 node-feature-free columns of the ISS channel: local closure (CN, AA, RA, Jaccard, prior weight, hop,
              2-4 path counts), degree / recency / activity of the endpoints and binary / weighted walk probabilities
"""
import numpy as np
import scipy.sparse as ss
from numba import njit

from .data import N, pair_key

EPS = 1e-12


# ---------------------------------------------------------------------------------------------------------- B6 (R*)
def build_csr(edges, n=N):
    e = np.sort(np.asarray(edges, np.int64).reshape(-1, 2), axis=1)
    e = np.unique(e[e[:, 0] != e[:, 1]], axis=0)
    directed = np.vstack((e, e[:, ::-1]))
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(rowptr, directed[:, 0] + 1, 1)
    rowptr = np.cumsum(rowptr)
    return rowptr, directed[:, 1].astype(np.int64, copy=False), np.diff(rowptr).astype(np.float64)


@njit(cache=True)
def _has(rowptr, col, u, v):
    lo, hi = rowptr[u], rowptr[u + 1]
    while lo < hi:
        m = (lo + hi) // 2
        if col[m] < v:
            lo = m + 1
        else:
            hi = m
    return lo < rowptr[u + 1] and col[lo] == v


@njit(cache=True)
def _row_value(rowptr, col, deg, u, v):
    return 1.0 / deg[u] if deg[u] > 0 and _has(rowptr, col, u, v) else 0.0


@njit(cache=True)
def _walk_kernel(rowptr, col, deg, pairs):
    """Per pair (u, v), 17 exact walk statistics (columns of WALK_COLUMNS) over the CSR graph."""
    n = pairs.shape[0]
    out = np.zeros((n, 17), dtype=np.float64)
    for i in range(n):
        u, v = int(pairs[i, 0]), int(pairs[i, 1])
        du, dv = deg[u], deg[v]
        out[i, 0] = _row_value(rowptr, col, deg, u, v)
        out[i, 1] = _row_value(rowptr, col, deg, v, u)
        w2f = w2r = c2f = c2r = w3f = w3r = c3f = c3r = nb2f = nb2r = nb3f = nb3r = ret2u = ret2v = 0.0
        idmean = 0.0
        ninter = 0
        for aa in range(rowptr[u], rowptr[u + 1]):
            a = int(col[aa])
            if deg[a] > 0:
                ret2u += 1.0 / (du * deg[a]) if du > 0 else 0.0
            if _has(rowptr, col, a, v):
                c2f += 1
                w2f += 1.0 / (du * deg[a]) if du > 0 and deg[a] > 0 else 0.0
                nb2f += 1 if v != u else 0
                ninter += 1
                idmean += deg[a]
            for bb in range(rowptr[a], rowptr[a + 1]):
                b = int(col[bb])
                if _has(rowptr, col, b, v):
                    c3f += 1
                    w3f += 1.0 / (du * deg[a] * deg[b]) if du > 0 and deg[a] > 0 and deg[b] > 0 else 0.0
                    if b != u and v != a:
                        nb3f += 1
        for aa in range(rowptr[v], rowptr[v + 1]):
            a = int(col[aa])
            if deg[a] > 0:
                ret2v += 1.0 / (dv * deg[a]) if dv > 0 else 0.0
            if _has(rowptr, col, a, u):
                c2r += 1
                w2r += 1.0 / (dv * deg[a]) if dv > 0 and deg[a] > 0 else 0.0
                nb2r += 1 if u != v else 0
            for bb in range(rowptr[a], rowptr[a + 1]):
                b = int(col[bb])
                if _has(rowptr, col, b, u):
                    c3r += 1
                    w3r += 1.0 / (dv * deg[a] * deg[b]) if dv > 0 and deg[a] > 0 and deg[b] > 0 else 0.0
                    if b != v and u != a:
                        nb3r += 1
        out[i, 2] = ret2u
        out[i, 3] = ret2v
        out[i, 4] = w2f
        out[i, 5] = w2r
        out[i, 6] = w3f
        out[i, 7] = w3r
        out[i, 8] = c2f
        out[i, 9] = c2r
        out[i, 10] = c3f
        out[i, 11] = c3r
        out[i, 12] = nb2f
        out[i, 13] = nb2r
        out[i, 14] = nb3f
        out[i, 15] = nb3r
        out[i, 16] = idmean / ninter if ninter else 0.0
    return out


WALK_COLUMNS = ('rw1_forward', 'rw1_reverse', 'self_return2_u', 'self_return2_v', 'rw2_mass_forward', 'rw2_mass_reverse',
                'rw3_mass_forward', 'rw3_mass_reverse', 'walk2_count_forward', 'walk2_count_reverse', 'walk3_count_forward',
                'walk3_count_reverse', 'nbw2_forward', 'nbw2_reverse', 'nbw3_forward', 'nbw3_reverse', 'interior_degree_mean')


def _rrwp(raw):
    """18 columns: for walk lengths 1-3, forward / reverse probability, their mean, |difference| and logs."""
    cols = []
    for f, r in ((0, 1), (4, 5), (6, 7)):
        cols += [raw[:, f], raw[:, r], .5 * (raw[:, f] + raw[:, r]), np.abs(raw[:, f] - raw[:, r]),
                 np.log(raw[:, f] + EPS), np.log(raw[:, r] + EPS)]
    return np.column_stack(cols).astype(np.float32)


def _semantic(pairs, x, deg, raw, history, years):
    """14 endpoint / temporal columns. NOTE (kept from the fitted model's inputs): the endpoint row counts and last years
    pair the canonical (unique) history rows with the years of the raw rows position by position, truncated to the shorter
    list; the residual model was fit on exactly this convention."""
    u, v = pairs[:, 0], pairs[:, 1]
    nu = np.linalg.norm(x[u], axis=1)
    nv = np.linalg.norm(x[v], axis=1)
    cosine = (x[u] * x[v]).sum(1) / (nu * nv + 1e-12)
    m = min(len(history), len(years))
    h, y = history[:m], np.asarray(years[:m], np.float32)
    counts = np.zeros(len(x), dtype=np.float32)
    np.add.at(counts, h[:, 0], 1)
    np.add.at(counts, h[:, 1], 1)
    last = np.full(len(x), -1, dtype=np.float32)
    np.maximum.at(last, h[:, 0], y)
    np.maximum.at(last, h[:, 1], y)
    in_graph = np.isin(pairs[:, 0] * N + pairs[:, 1], history[:, 0] * N + history[:, 1])   # orientation-exact lookup
    rel = np.column_stack([cosine, nu, nv, np.abs(nu - nv), deg[u], deg[v], np.minimum(deg[u], deg[v]),
                           np.maximum(deg[u], deg[v]), raw[:, 8], raw[:, 10], deg[u] * deg[v], in_graph.astype(np.float32),
                           counts[u], counts[v], last[u], last[v]]).astype(np.float32)
    return rel[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]]


def b6_features(history, years, pairs, x):
    """{'semantic': [P, 14], 'semantic_rrwp': [P, 32]} float32 for pairs on `history` (canonical rows) dated by `years`."""
    pairs = np.asarray(pairs, np.int64)
    rowptr, col, deg = build_csr(history, len(x))
    raw = _walk_kernel(rowptr, col, deg, pairs).astype(np.float32)
    sem = _semantic(pairs, x, deg, raw, history, years)
    return {'semantic': sem, 'semantic_rrwp': np.column_stack([sem, _rrwp(raw)])}


# ---------------------------------------------------------------------------------------------------------- x-free (ISS)
XCOLS = ['cn', 'aa', 'ra', 'jac', 'cnpos', 'prev_w', 'hop', 'lcn', 'lw3', 'w3n', 'lw4', 'n2min', 'n2max', 'w3pos',
         'dmin', 'dmax', 'dsum', 'since_min', 'since_max', 'cnt_min', 'cnt_max', 'r1_min', 'r1_max', 'r2_min', 'r2_max',
         'age_min', 'age_max', 'RW2', 'RW3', 'RW4', 'MULT_RW23']
WALK_EPS = 1e-9
CHUNK = 2000


def _rownorm(M):
    d = np.asarray(M.sum(1)).ravel()
    return ss.diags(np.where(d > 0, 1 / np.maximum(d, 1e-300), 0)) @ M


def _rowdot(L, R):
    return np.asarray(L.multiply(R).sum(1)).ravel()


def _walks(M, P, ks):
    """Forward / reverse k-step walk probabilities of pairs under the row-stochastic M (chunks of 2000 pairs)."""
    M = M.tocsr()
    MT = M.T.tocsr()
    out = {k: ([], []) for k in ks}
    for s in range(0, len(P), CHUNK):
        u, v = P[s:s + CHUNK, 0], P[s:s + CHUNK, 1]
        Lu1, Lv1, Rv1, Ru1 = M[u], M[v], MT[v], MT[u]
        if 2 in ks:
            out[2][0].append(_rowdot(Lu1, Rv1))
            out[2][1].append(_rowdot(Lv1, Ru1))
        if 3 in ks or 4 in ks:
            Lu2, Lv2 = Lu1 @ M, Lv1 @ M
            if 3 in ks:
                out[3][0].append(_rowdot(Lu2, Rv1))
                out[3][1].append(_rowdot(Lv2, Ru1))
            if 4 in ks:
                out[4][0].append(_rowdot(Lu2, Rv1 @ MT))
                out[4][1].append(_rowdot(Lv2, Ru1 @ MT))
    return {k: (np.concatenate(a), np.concatenate(b)) for k, (a, b) in out.items()}


def _sym(g, w):
    return ss.coo_matrix((np.r_[w, w], (np.r_[g[:, 0], g[:, 1]], np.r_[g[:, 1], g[:, 0]])), shape=(N, N)).tocsr()


class DatedGraph:
    """Graph visible before year Y: unique pairs with summed weights, binary adjacency, per-node degree / activity."""

    def __init__(self, rows, years, wts, Y):
        k = pair_key(rows)
        uk, inv = np.unique(k, return_inverse=True)
        self.g = np.c_[uk // N, uk % N]
        self.w = np.bincount(inv, weights=wts)
        self.keys, self.Y = uk, Y
        A = ss.coo_matrix((np.ones(2 * len(self.g)), (np.r_[self.g[:, 0], self.g[:, 1]], np.r_[self.g[:, 1], self.g[:, 0]])),
                          shape=(N, N)).tocsr()
        A.data[:] = 1
        self.A = A
        self.deg = np.asarray(A.sum(1)).ravel()
        last = np.full(N, Y - 60.0)
        np.maximum.at(last, rows[:, 0], years)
        np.maximum.at(last, rows[:, 1], years)
        self.since = Y - last
        self.cnt = np.bincount(rows.ravel(), minlength=N).astype(float)
        self.r1 = np.bincount(rows[years == Y - 1].ravel(), minlength=N).astype(float)
        self.r2 = np.bincount(rows[years >= Y - 2].ravel(), minlength=N).astype(float)
        f = np.full(N, 9999)
        np.minimum.at(f, rows[:, 0], years)
        np.minimum.at(f, rows[:, 1], years)
        self.first = f

    def prev_w(self, P):
        k = pair_key(P)
        i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return np.where(self.keys[i] == k, self.w[i], 0.0)

    def _paths(self, P):
        A = self.A
        deg = np.asarray(A.sum(1)).ravel()
        u, v = P[:, 0], P[:, 1]
        Dinv = ss.diags(1 / np.maximum(deg, 1))
        AD = (A @ Dinv).tocsr()
        cn, w3, w3n, w4, n2u, n2v = (np.zeros(len(u)) for _ in range(6))
        for s in range(0, len(u), CHUNK):
            b = slice(s, s + CHUNK)
            Au, Av = A[u[b]], A[v[b]]
            Au2, Av2 = Au @ A, Av @ A
            cn[b] = np.asarray(Au.multiply(Av).sum(1)).ravel()
            w3[b] = np.asarray(Au2.multiply(Av).sum(1)).ravel()
            w4[b] = np.asarray(Au2.multiply(Av2).sum(1)).ravel()
            w3n[b] = np.asarray(((Au @ Dinv) @ AD).multiply(Av @ Dinv).sum(1)).ravel()
            n2u[b] = np.diff((Au2 > 0).astype(np.int8).tocsr().indptr)
            n2v[b] = np.diff((Av2 > 0).astype(np.int8).tocsr().indptr)
        return np.column_stack([np.log1p(cn), np.log1p(w3), w3n, np.log1p(w4), np.log1p(np.minimum(n2u, n2v)),
                                np.log1p(np.maximum(n2u, n2v)), (w3 > 0).astype(float)])

    def xfree(self, P):
        """[P, 31] float32 in XCOLS order."""
        P = np.asarray(P, np.int64)
        A, deg = self.A, self.deg
        u, v = P[:, 0], P[:, 1]
        M = A[u].multiply(A[v])
        cn = np.asarray(M.sum(1)).ravel()
        D = {'cn': cn, 'aa': np.asarray(M.multiply(1 / np.log(np.maximum(deg, 2))).sum(1)).ravel(),
             'ra': np.asarray(M.multiply(1 / np.maximum(deg, 1)).sum(1)).ravel(),
             'jac': cn / np.maximum(deg[u] + deg[v] - cn, 1), 'cnpos': (cn > 0).astype(float),
             'prev_w': np.log1p(self.prev_w(P))}
        for name, a in (('d', np.log1p(deg)), ('since', self.since), ('cnt', np.log1p(self.cnt)), ('r1', np.log1p(self.r1)),
                        ('r2', np.log1p(self.r2))):
            lo, hi = np.minimum(a[u], a[v]), np.maximum(a[u], a[v])
            if name == 'd':
                D['dmin'], D['dmax'], D['dsum'] = lo, hi, lo + hi
            else:
                D[f'{name}_min'], D[f'{name}_max'] = lo, hi
        Ab = _sym(self.g, np.ones(len(self.g)))
        Ab.data[:] = 1
        wb = _walks(_rownorm(Ab), P, (2, 3, 4))
        ww = _walks(_rownorm(_sym(self.g, self.w)), P, (2, 3))
        L = lambda a: np.log(WALK_EPS + a)  # noqa: E731
        for k in (2, 3, 4):
            D[f'RW{k}'] = L(.5 * (wb[k][0] + wb[k][1]))
        D['MULT_RW23'] = L(.25 * (ww[2][0] + ww[2][1] + ww[3][0] + ww[3][1]))
        # the source computed the four walk columns as one float32 block before deriving `hop` from them
        for k in ('RW2', 'RW3', 'RW4', 'MULT_RW23', 'cn', 'cnpos', 'prev_w'):
            D[k] = D[k].astype(np.float32).astype(np.float64)
        rw0 = np.log(WALK_EPS) + 1e-12
        D['hop'] = np.where(D['prev_w'] > 0, 1, np.where(D['cnpos'] > 0, 2, np.where(D['RW3'] > rw0, 3,
                            np.where(D['RW4'] > rw0, 4, 5)))).astype(float)
        FP = self._paths(P)
        for j, k in enumerate(('lcn', 'lw3', 'w3n', 'lw4', 'n2min', 'n2max', 'w3pos')):
            D[k] = FP[:, j]
        f = self.first
        a = np.log1p(np.maximum(self.Y - f[P], 0))
        a = np.where(f[P] > 2100, np.log1p(60), a)
        D['age_min'], D['age_max'] = a.min(1), a.max(1)
        return np.column_stack([D[k] for k in XCOLS]).astype(np.float32)
