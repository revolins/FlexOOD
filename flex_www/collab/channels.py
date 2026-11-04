"""The two heuristic channels FLEX adds on top of R* + age10 + ISS10 on real ogbl-collab.

AD (AA-DC)   Adamic-Adar on the time-decayed co-authorship graph with a degree-calibrated gate and an L3 rescue for pairs
             without common neighbours (the authors' AA-DC; flex_www.flex.aadc, verbatim). Edge weight = co-authorship
             count x 0.95^(2018 - year). Gate thresholds and the L3 anchor scale are calibrated on the validation split
             against the train graph, exactly as AA-DC's own --auto_gate / --auto_beta run on ogbl-collab. The channel is
             then read on the same age10 graphs as the other channels: half B on the validation-time age10 graph, the test
             pairs on the age10 test-time graph. AA-DC alone on that graph is also the paper's "AA-DC" collab row.
             It adds near-threshold triadic-closure magnitude that the GCN and R* under-rank.
CN_REC2      common neighbours counted on the last two years of rows only (validation: 2016-2017; test: 2017-2018) of the
             same age10 graphs: recent shared collaborators are the strongest single sign of a new co-authorship.
             Channel = log1p(count), standardised with the half-B mean / std.
"""
import numpy as np
import scipy.sparse as ss

from .data import N, pair_key
from ..flex import aadc as AD
from ..flex import aadc_runtime as ADR

DECAY = 0.95
MAX_YEAR = 2018.0          # AA-DC's max_year on collab: the latest train / validation year


# ---------------------------------------------------------------------------------------------------------- AA-DC
def train_weights(c):
    """AA-DC edge weights of the train rows: base weight = the dataset edge_weight of the LAST occurrence of (u, v) in the
    dataset's edge_index (else of (v, u), else 1), times 0.95^(2018 - year), in float32 (AA-DC's edge_to_weight)."""
    ei = c.data.edge_index.numpy()
    w = c.data.edge_weight.numpy().astype(np.float32).ravel()
    dk = ei[0].astype(np.int64) * N + ei[1]
    uk, inv = np.unique(dk, return_inverse=True)
    last = np.full(len(uk), -1, np.int64)
    np.maximum.at(last, inv, np.arange(len(dk)))
    wk = w[last]

    def lookup(keys):
        i = np.minimum(np.searchsorted(uk, keys), len(uk) - 1)
        return np.where(uk[i] == keys, wk[i], np.nan)
    fwd = lookup(c.E[:, 0] * N + c.E[:, 1])
    rev = lookup(c.E[:, 1] * N + c.E[:, 0])
    base = np.where(np.isnan(fwd), np.where(np.isnan(rev), 1.0, rev), fwd).astype(np.float32)
    return base * np.power(DECAY, MAX_YEAR - c.Y.astype(np.float32))


def graph_objects(edges, weights):
    """AA-DC's graph objects for one weighted row list (rows are used as given: duplicates sum, as in AA-DC)."""
    A = AD.build_weighted_adj(N, edges, weights)
    inv = AD.precompute_inv_log_deg(A)
    return dict(A=A, A_invlog=AD.precompute_aa_matrix(A, inv), inv_log_deg=inv, adj=AD.build_adj(N, edges),
                lcc=AD.compute_exact_lcc(A))


class AADC:
    """AA-DC calibrated on validation (train graph); scores on any edited graph."""

    def __init__(self, c):
        self.c = c
        self.w = train_weights(c)
        G = graph_objects(c.E, self.w)
        self.gth, self.scale, self.info = ADR.calibrate(G, c.Ev, c.Nv)

    def score(self, G, pairs):
        y, *_ = ADR.score(G, np.asarray(pairs, np.int64), gate_thresholds=self.gth, anchor_scale=self.scale)
        return y

    def val_age10(self):
        """Half-B scores on the validation-time age10 graph (train rows of pairs last seen <= 2007 removed)."""
        keep = self.c.val_age10()
        return self.score(graph_objects(self.c.E[keep], self.w[keep]), self.c.PB)

    def test_age10(self):
        """Test-pair scores on the age10 test-time graph (train + validation rows, validation weight 0.95^0 = 1). The L3
        rescue is directed, so pairs are scored in the dataset's orientation and then mapped to the canonical PT order."""
        c = self.c
        _, drop = c.test_age10()
        rows = np.vstack([c.E, c.Ev])
        wv = np.power(DECAY, MAX_YEAR - c.valid_year.astype(np.float32))
        w = np.concatenate([self.w, wv])
        keep = ~np.isin(pair_key(rows), drop)
        G = graph_objects(rows[keep], w[keep])
        y = np.r_[self.score(G, c.Tp), self.score(G, c.Tn)].astype(np.float64)
        k = np.r_[pair_key(c.Tp), pair_key(c.Tn)]
        order = np.argsort(k)
        idx = order[np.searchsorted(k[order], pair_key(c.PT))]
        assert np.array_equal(k[idx], pair_key(c.PT))
        return y[idx]


# ---------------------------------------------------------------------------------------------------------- CN_REC2
def recent_cn(rows, years, since, pairs, chunk=2000):
    """Common neighbours of each pair on the binary graph of the rows with year >= since."""
    k = np.unique(pair_key(rows[years >= since]))
    u, v = k // N, k % N
    A = ss.csr_matrix((np.ones(2 * len(k), np.float32), (np.r_[u, v], np.r_[v, u])), shape=(N, N))
    out = np.zeros(len(pairs))
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        out[s:s + chunk] = np.asarray(A[p[:, 0]].multiply(A[p[:, 1]]).sum(1)).ravel()
    return out


def cn_rec2_val(c):
    m = c.val_age10()
    return recent_cn(c.E[m], c.Y[m], 2016, c.PB)


def cn_rec2_test(c):
    rows, yrs, _ = c.test_rows()
    m, _ = c.test_age10()
    return recent_cn(rows[m], yrs[m], 2017, c.PT)
