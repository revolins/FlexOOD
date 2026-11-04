"""Real ogbl-collab arrays shared by every FLEX stage: the dated train rows, the two evaluation pair sets and the age10 edit.

Pair sets (both fixed; labels are the layout, positives first)
  test   PT = canonical (sorted, unique) test positives, then the canonical 100,000 official test negatives
  val B  PB = "validation half B": a fixed random half of the canonical validation positives and of the official validation
         negatives (numpy default_rng(20260930)); every validation pick of FLEX is made on half B

Graphs
  validation time  train rows (years <= 2017)
  test time        train rows + the 2018 validation edges (the OGB rule: validation edges are test-time input)
  age10            on either graph, drop EVERY row of a pair whose last collaboration year is >= 10 years before the label
                   year (test: last year <= 2008; validation: <= 2007). Stale ties carry little signal about who links next
                   year; the edit is fixed (no fitted scalar). Validation edges are never dropped.
"""
from dataclasses import dataclass

import numpy as np

N = 235868                     # ogbl-collab nodes
VAL_HALF_SEED = 20260930       # draws validation half B
AGE = 10                       # age10: drop pairs whose last year is <= label_year - 1 - AGE


def pair_key(p):
    """Orientation-free integer key of each pair."""
    p = np.sort(np.asarray(p, np.int64).reshape(-1, 2), 1)
    return p[:, 0] * N + p[:, 1]


def canon(p):
    """Sorted endpoints, unique rows (row order = lexicographic)."""
    return np.unique(np.sort(np.asarray(p, np.int64).reshape(-1, 2), 1), axis=0)


def canon_keep_dups(p):
    """Sorted endpoints, self-loops removed, duplicates and row order kept."""
    p = np.sort(np.asarray(p, np.int64).reshape(-1, 2), 1)
    return p[p[:, 0] != p[:, 1]]


def stale_rows(E, Y, last_ok):
    """Row mask of rows whose pair was last seen after `last_ok` (age edit: rows of pairs last seen <= last_ok are dropped)."""
    k = pair_key(E)
    uk, inv = np.unique(k, return_inverse=True)
    last = np.full(len(uk), -1)
    np.maximum.at(last, inv, Y)
    return ~(last[inv] <= last_ok)


def node_activity(E, Y, T):
    """Years since each node's last row before year T (99 = never seen)."""
    last = np.full(N, -1.0)
    np.maximum.at(last, E[:, 0], Y.astype(float))
    np.maximum.at(last, E[:, 1], Y.astype(float))
    return np.where(last >= 0, T - last, 99.0)


@dataclass
class Collab:
    E: np.ndarray            # train rows [R, 2] (OGB order)
    Y: np.ndarray            # their years
    W: np.ndarray            # their co-authorship weights
    Ev: np.ndarray           # validation positives (2018) [V, 2] (OGB order)
    Nv: np.ndarray           # official validation negatives
    Tp: np.ndarray           # test positives (OGB order; scored only, labels = layout)
    Tn: np.ndarray           # official test negatives
    x: np.ndarray            # node features float32
    data: object             # the PyG Data (edge_index both directions + edge_weight), used by the GCN graphs
    valid_weight: np.ndarray = None   # validation edge weights (test-time graph of the ISS / CN_REC2 channels)
    valid_year: np.ndarray = None     # all 2018
    split: object = None              # the flex_www.data Split (backbones build their model input from it)
    PT: np.ndarray = None
    npos: int = 0
    PB: np.ndarray = None
    npB: int = 0

    # ------------------------------------------------------------------ pair sets
    def build_pairs(self):
        tp, tn = canon(self.Tp), canon(self.Tn)
        self.PT, self.npos = np.vstack([tp, tn]), len(tp)
        vpos, vneg = canon(self.Ev), canon(self.Nv)
        rng = np.random.default_rng(VAL_HALF_SEED)
        ha, hn = rng.random(len(vpos)) < 0.5, rng.random(len(vneg)) < 0.5
        self.PB, self.npB = np.vstack([vpos[~ha], vneg[~hn]]), int((~ha).sum())
        return self

    # ------------------------------------------------------------------ test-time graph (train + validation edges)
    def test_rows(self):
        """(rows, years, weights) of the test-time input: train rows then the validation edges (year 2018, their weights)."""
        Wv = np.asarray(self.valid_weight, float)
        return (np.vstack([self.E, self.Ev]), np.r_[self.Y, np.full(len(self.Ev), 2018, np.int64)], np.r_[self.W, Wv])

    def test_age10(self):
        """Row mask over test_rows() (age10: pairs last seen <= 2008 dropped) and the dropped pair keys."""
        rows, yrs, _ = self.test_rows()
        m = stale_rows(rows, yrs, 2018 - AGE)
        assert m[len(self.E):].all(), 'validation edges are never dropped'
        return m, np.unique(pair_key(rows)[~m])

    # ------------------------------------------------------------------ validation-time graph (train rows)
    def val_age10(self):
        """Row mask over the train rows (age10 at validation time: pairs last seen <= 2007 dropped)."""
        return stale_rows(self.E, self.Y, 2017 - AGE)


def load_collab(root=None):
    """Collab arrays from the OGB dataset (flex_www.data loader), with the PT / PB pair sets built."""
    from ..data import load
    s = load('ogbl-collab', root=root)
    A = lambda t: np.asarray(t.numpy() if hasattr(t, 'numpy') else t)  # noqa: E731
    c = Collab(E=A(s.train_pos).astype(np.int64), Y=A(s.info['train_year']).ravel().astype(np.int64),
               W=A(s.info['train_weight']).ravel().astype(float), Ev=A(s.valid_pos).astype(np.int64),
               Nv=A(s.valid_neg).astype(np.int64), Tp=A(s.test_pos).astype(np.int64), Tn=A(s.test_neg).astype(np.int64),
               x=s.data.x.numpy().astype(np.float32), data=s.data,
               valid_weight=A(s.info['valid_weight']).ravel().astype(float), valid_year=A(s.info['valid_year']).ravel(), split=s)
    assert (c.valid_year == 2018).all()
    return c.build_pairs()
