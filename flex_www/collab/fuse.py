"""Validation picks and the additive late fusion used by every FLEX stage on real ogbl-collab.

Every channel c enters a score S as   S + g * a6 * csd * c   where a6 is the run's B6 residual weight and csd the standard
deviation of the residual on validation half B (so g is unit-free), and g is picked on validation half B:

  pick      g from a grid by the net-repair criterion: (# positives that enter the top-K_eff minus # that leave), averaged
            over K in {20, 50}; ties -> smaller |g|, then the negative sign; a non-zero g is kept only if half-B Hits@50
            strictly improves, otherwise g = 0 (the channel is off for that run)
  K_eff     half B has ~50k negatives instead of the official 100k, so Hits@K is read at round(K * nneg / 100000)

Hits@K on the official test layout: a positive is a hit if it scores strictly above the K-th highest negative.
"""
import numpy as np

GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
SIGNED = GAMMAS + tuple(-g for g in GAMMAS if g)
BETAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)      # grid of the B6 / twin weights (choose)


def keff(K, nneg):
    return max(1, int(round(K * nneg / 100000)))


def hits(scores, npos, k):
    """Per-positive Hits@k (positives first, then the shared negatives)."""
    return scores[:npos] > np.partition(scores[npos:], -k)[-k]


class ESet:
    """Selection set: positive and negative row indices of half B, Hits at K_eff."""

    def __init__(self, npos, nrows):
        self.pi, self.ni = np.arange(npos), npos + np.arange(nrows - npos)
        self.k = {K: keff(K, len(self.ni)) for K in (20, 50)}

    def hits(self, S, K):
        k = self.k[K]
        return (S[self.pi] > np.partition(S[self.ni], -k)[-k]).astype(float)


def nrcrit(sel, S0, S1):
    """Net repairs of S1 over S0, averaged over K in {20, 50}."""
    c = []
    for K in (20, 50):
        h0, h1 = sel.hits(S0, K) > 0, sel.hits(S1, K) > 0
        c.append((h1 & ~h0).sum() - (h0 & ~h1).sum())
    return float(np.mean(c))


def pick(S, c, a, sel, grid=SIGNED):
    """(g, fell_back): the weight of channel c on base S (fusion S + g * a * c) picked on sel."""
    best = None
    for gm in grid:
        key = (nrcrit(sel, S, S + gm * a * c), -abs(gm), -gm)
        if best is None or key > best[0]:
            best = (key, gm)
    gm = best[1]
    if gm != 0 and not (sel.hits(S + gm * a * c, 50).mean() > sel.hits(S, 50).mean()):
        return 0.0, True
    return float(gm), False


def net_repair(base, new, npos, k):
    b, h = hits(base, npos, k), hits(new, npos, k)
    return int((h & ~b).sum() - (b & ~h).sum())


def choose(vb, vd, npos, grid=BETAS):
    """Weight a of an additive term vd on base vb: max mean net repair at K = 10 and 25 on half B; ties -> smaller a."""
    c = {a: np.mean([net_repair(vb, vb + a * vd, npos, k) for k in (10, 25)]) for a in grid}
    return max(grid, key=lambda v: (c[v], -v))


def unit(x):
    """(x - mean) / std over the validation rows, and the (mean, std) used for the test side."""
    x = np.asarray(x, np.float64)
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / (sd if sd > 0 else 1.0), (mu, sd if sd > 0 else 1.0)


def test_hits(S, npos, K):
    """Per-positive Hits@K on the official test layout (K-th of the 100,000 negatives)."""
    return (S[:npos] > np.partition(S[npos:], -K)[-K]).astype(float)
