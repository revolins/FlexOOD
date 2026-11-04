"""FLEX fusion and its validation-only picks.

Fused score of a pair (per seed s):  S = b * z(logit p_GCN) + g1 zACT + g2 zREL + g3 zISS_s [+ g4 zX] [+ g5 zAD]
where z(logit p) is standardised on the seed's validation predictions. Weights are picked in stages on the cell's
selection metric over VALIDATION, averaged over seeds; a stage's candidate is adopted only if its mean gain over the
stage's base is > 0 and >= ADOPT_FRAC of the seeds gain (otherwise the base is kept), so FLEX equals the encoder when
nothing transfers. Ties prefer the smallest sum |g|.
  stage 1  (g1, g2) jointly over GRID x GRID
  stage 2  g3 (ISS) on top
  stage 3  g4 (feature cosine), X rows only
  stage AD g5 (AA-DC), _AD rows
  stage B  (b, g4, g5) jointly with b in BASE_GRID, _B rows (ties: b = 1 first, then small |g4| + |g5|)
Optional constraints: `fold` (DIR / SP rows: the metric is computed on a validation fold along the shift, SP rows add it
to the full-validation metric), `gate` (gated rows: a candidate is eligible only if its fold metric is >= the encoder's),
`guard` (_G / _B rows: a channel whose validation-positive scores have zero variance in any seed gets weight 0).
"""
import numpy as np
import torch

from .. import evaluate as E

GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
BASE_GRID = (1.0, 0.5, 0.25, 0.0)
ADOPT_FRAC = 0.6


class MetricFn:
    """The cell's metric code on prediction lists: `full(P)` -> {metric: (train, valid, test)}, `value(P, metric, split)`
    -> one value (bit-identical to full(P)[metric][split])."""

    def __init__(self, layout, per_query=0):
        self.layout, self.per_query = layout, per_query

    def full(self, P):
        return E.metrics(self.layout, P, self.per_query)

    def value(self, P, metric, split=1):
        return float(E.metric_value(self.layout, P, metric, split, self.per_query))

    def sub_neg(self, nv, npos, idx):
        """Negatives of the positives idx: rows of a per-positive [P, K] block, or per-query blocks of a flat list."""
        if nv.dim() == 2 and nv.size(0) == npos:
            return nv[idx]
        q = self.per_query
        if q and nv.numel() == npos * q:
            return nv.view(npos, q)[idx].reshape(-1)
        return nv


def logit(p, prob):
    """Probabilities -> logits (float64, clamped at the float64 limits); raw scores (logit-emitting predictors) unchanged."""
    p = p.double()
    if not prob:
        return p
    p = p.clamp(1e-300, 1 - 2.0 ** -52)
    return torch.log(p) - torch.log1p(-p)


def fuse(P, ch, s, g, bw=1.0):
    """Fused prediction list: P = [pos_train, pos_valid, neg_valid, pos_test, neg_test, ...], ch = channels of
    (pos_valid, neg_valid, pos_test, neg_test), g = weights (g1..g5), bw = weight of the encoder term."""
    pv, nv, pt, nt = P[1], P[2], P[3], P[4]
    prob = bool(min(float(t.min()) for t in (pv, nv, pt, nt)) >= 0 and max(float(t.max()) for t in (pv, nv, pt, nt)) <= 1)
    allv = torch.cat([logit(pv, prob).view(-1), logit(nv, prob).view(-1)])
    mu, sd = allv.mean(), allv.std().clamp_min(1e-9)

    def f(base, c):
        assert base.numel() == len(c['act']), f'pred/pair layout mismatch {tuple(base.shape)} vs {len(c["act"])}'
        b = (logit(base, prob) - mu) / sd
        if bw != 1.0:
            b = bw * b
        add = g[0] * c['act'] + g[1] * c['rel'] + g[2] * c['iss'][s]
        if len(g) > 3 and g[3] != 0.0:
            add = add + g[3] * c['xc']
        if len(g) > 4 and g[4] != 0.0:
            add = add + g[4] * c['ad']
        return (b.view(-1) + torch.as_tensor(add, dtype=torch.float64)).view(base.shape)
    Q = list(P)
    Q[1], Q[2], Q[3], Q[4] = f(pv, ch[0]), f(nv, ch[1]), f(pt, ch[2]), f(nt, ch[3])
    return Q


def pick(values_by_g, base_key, allowed=None, cost=None):
    """Adoption rule. values_by_g: {g: [per-seed validation value]}; allowed: eligible candidates (base always eligible);
    cost: tie-break, lowest wins (default sum |g|). Returns (g, info)."""
    cands = values_by_g if allowed is None else {g: v for g, v in values_by_g.items() if g in allowed or g == base_key}
    cost = cost or (lambda g: sum(abs(x) for x in g))
    b = np.array(values_by_g[base_key])
    best = max(cands, key=lambda g: (np.mean(cands[g]), -cost(g)))
    d = np.array(values_by_g[best]) - b
    ok = d.mean() > 0 and (d > 0).mean() >= ADOPT_FRAC
    return (best if ok else base_key), {'cand': list(best), 'mean_gain': float(d.mean()), 'frac_pos': float((d > 0).mean()),
                                        'adopted': bool(ok)}


def degenerate(chans, seeds, name):
    """True if the channel's validation-positive scores have zero variance in some seed."""
    for s in seeds:
        v = np.asarray(chans[s][0][name], np.float64)
        if v.size == 0 or np.ptp(v) == 0 or v.std() < 1e-12:
            return True
    return False


def fold_value(mf, Q, idx, metric):
    """Metric on the validation positives idx and their own negatives."""
    npos = Q[1].numel()
    Q = list(Q)
    Q[2] = mf.sub_neg(Q[2], npos, idx)
    Q[1] = Q[1][idx]
    assert Q[2].dim() == 2 and Q[2].size(0) == len(idx), f'fold needs per-positive negatives, got {tuple(Q[2].shape)}'
    return mf.value(Q, metric, 1)


def run_picks(mf, seeds, epP, chans, metric, log=print, fold=None, combined=False, gate=None, xstage=False, guard=False):
    """Stages 1, 2 (and 3 on X rows) at the selection epoch's predictions epP[s]. Returns (g, info)."""
    if fold is not None:
        idx = torch.as_tensor(np.flatnonzero(fold))
        assert len(idx) > 0, 'empty direction fold'
    if gate is not None:
        assert fold is None, 'gate and fold are exclusive'
        gidx = torch.as_tensor(np.flatnonzero(gate))
        assert len(gidx) > 0, 'empty gate fold'

    def one(Q):
        if fold is not None and combined:
            return mf.value(Q, metric, 1) + fold_value(mf, Q, idx, metric)
        if fold is not None:
            return fold_value(mf, Q, idx, metric)
        return mf.value(Q, metric, 1)

    def vals(g):
        return [one(fuse(epP[s], chans[s], s, g)) for s in seeds]

    def gvals(g):
        return float(np.mean([fold_value(mf, fuse(epP[s], chans[s], s, g), gidx, metric) for s in seeds]))
    V1 = {(a, b, 0.0): vals((a, b, 0.0)) for a in GRID for b in GRID}
    allowed1 = allowed2 = None
    ginfo = {}
    if gate is not None:
        G1 = {g: gvals(g) for g in V1}
        g0 = G1[(0.0, 0.0, 0.0)]
        allowed1 = {g for g, v in G1.items() if v >= g0}
        ginfo = {'gate_fold_at_0': g0, 'stage1_eligible': len(allowed1), 'stage1_total': len(G1)}
    g12, i1 = pick(V1, (0.0, 0.0, 0.0), allowed1)
    log('stage1 R*_t', g12, i1, ginfo)
    V2 = {(g12[0], g12[1], c): (V1[g12] if c == 0.0 else vals((g12[0], g12[1], c))) for c in GRID}
    if gate is not None:
        G2 = {g: (G1[g] if g in G1 else gvals(g)) for g in V2}
        allowed2 = {g for g, v in G2.items() if v >= g0}
        ginfo.update(stage2_eligible=len(allowed2), gate_fold_at_g=None)
    g, i2 = pick(V2, g12, allowed2)
    log('stage2 ISS', g, i2)
    if gate is not None:
        ginfo['gate_fold_at_g'] = G2[g]
        i2 = dict(i2, gate=ginfo)
    out = {'stage1': i1, 'stage2': i2, 'val_mean': float(np.mean(V2[g])), 'val_base': float(np.mean(V1[(0.0, 0.0, 0.0)]))}
    if not xstage:
        return g, out
    g3 = (g[0], g[1], g[2], 0.0)                       # stage 3: feature-cosine weight g4 on top of the stage-2 pick
    V3 = {(g[0], g[1], g[2], d): (V2[g] if d == 0.0 else vals((g[0], g[1], g[2], d))) for d in GRID}
    allowed3 = None
    if gate is not None:
        G3 = {k: (G2[g] if k == g3 else gvals(k)) for k in V3}
        allowed3 = {k for k, v in G3.items() if v >= g0}
    rej = []
    if guard and degenerate(chans, seeds, 'xc'):
        allowed3 = {k for k in (V3 if allowed3 is None else allowed3) if k[3] == 0.0}
        rej = ['xc']
    gx, i3 = pick(V3, g3, allowed3)
    log('stage3 X', gx, i3, *(['guard rejects xc'] if rej else []))
    out.update(stage3=i3, g_no_x=list(g), val_mean=float(np.mean(V3[gx])))
    if guard:
        out['guard_rejected'] = rej
    return gx, out


def gate_at_0(mf, seeds, sel, chans, metric, gidx):
    """Gate-fold metric of the encoder alone (b = 1, every g = 0)."""
    return float(np.mean([fold_value(mf, fuse(sel(s, metric), chans[s], s, (0.0,) * 5), gidx, metric) for s in seeds]))


def stage_ad(mf, seeds, sel, chans, metric, g, log=print, gate=None, rules=False):
    """AA-DC weight g5 on top of the pick. rules (_G / _B): guard + fold gate. Returns (g, info, g_before, values, rejected)."""
    gb = tuple(list(g) + [0.0] * (4 - len(g)))
    VA = {gb + (d,): [mf.value(fuse(sel(s, metric), chans[s], s, gb + (d,)), metric, 1) for s in seeds] for d in GRID}
    if not rules:
        g, iad = pick(VA, gb + (0.0,))
        log('stage AD', g, iad)
        return g, iad, gb, VA, []
    ok, extra, rej = set(VA), {}, []
    if gate is not None:
        gidx = torch.as_tensor(np.flatnonzero(gate))
        assert len(gidx) > 0, 'empty gate fold'
        GA = {k: float(np.mean([fold_value(mf, fuse(sel(s, metric), chans[s], s, k), gidx, metric) for s in seeds])) for k in VA}
        g0 = gate_at_0(mf, seeds, sel, chans, metric, gidx)
        ok = {k for k, v in GA.items() if v >= g0}
        extra['gate'] = {'gate_fold_at_0': g0, 'gate_fold_at_pick': GA[gb + (0.0,)], 'eligible': len(ok), 'total': len(GA)}
    if degenerate(chans, seeds, 'ad'):
        ok = {k for k in ok if k[4] == 0.0}
        rej = ['ad']
    g, iad = pick(VA, gb + (0.0,), ok)
    iad.update(extra)
    if gate is not None:
        iad['gate']['gate_fold_at_g'] = GA[g]
    log('stage AD', g, iad, *(['guard rejects ad'] if rej else []))
    return g, iad, gb, VA, rej


def stage_base(mf, seeds, sel, chans, metric, g, log=print, gate=None):
    """Joint (b, g4, g5) re-pick with g1..g3 fixed (a guarded channel's current weight is zeroed first; that rebased pick is
    the adoption base). Returns (g, b, info, g_before, best b = 0 candidate (readout only), rejected)."""
    gf = tuple(float(x) for x in list(g) + [0.0] * (5 - len(g)))
    has_ad = all('ad' in chans[s][0] for s in seeds)
    full = lambda k: gf[:3] + (k[1], k[2])  # noqa: E731
    cost = lambda k: 100.0 * (1.0 - k[0]) + abs(k[1]) + abs(k[2])  # noqa: E731  (b first: |g4| + |g5| <= 2 < 100 * 0.25)
    VB = {(bw, d4, d5): [mf.value(fuse(sel(s, metric), chans[s], s, full((bw, d4, d5)), bw), metric, 1) for s in seeds]
          for bw in BASE_GRID for d4 in GRID for d5 in (GRID if has_ad else (0.0,))}
    ok, extra, rej = set(VB), {}, []
    if degenerate(chans, seeds, 'xc'):
        ok = {k for k in ok if k[1] == 0.0}
        rej.append('xc')
    if has_ad and degenerate(chans, seeds, 'ad'):
        ok = {k for k in ok if k[2] == 0.0}
        rej.append('ad')
    kb = (1.0, 0.0 if 'xc' in rej else gf[3], 0.0 if 'ad' in rej else gf[4])
    assert kb in VB, f'final pick {gf} outside the base-stage grid'
    if kb != (1.0, gf[3], gf[4]):
        extra['rebased_from'] = [1.0, gf[3], gf[4]]
    if gate is not None:
        gidx = torch.as_tensor(np.flatnonzero(gate))
        assert len(gidx) > 0, 'empty gate fold'
        GB_ = {k: float(np.mean([fold_value(mf, fuse(sel(s, metric), chans[s], s, full(k), k[0]), gidx, metric) for s in seeds]))
               for k in VB}
        g0 = gate_at_0(mf, seeds, sel, chans, metric, gidx)
        ok = {k for k in ok if GB_[k] >= g0}
        extra['gate'] = {'gate_fold_at_0': g0, 'gate_fold_at_pick': GB_[kb], 'eligible': len(ok), 'total': len(GB_)}
    kB, ib = pick(VB, kb, ok, cost=cost)
    ib.update(extra, key='(b, g4, g5)', n_cand=len(VB))
    assert not ('xc' in rej and kB[1]) and not ('ad' in rej and kB[2]), f'guarded channel weighted in {kB}'
    if gate is not None:
        ib['gate']['gate_fold_at_g'] = GB_[kB]
    log('stage base', kB, ib, *(['guard rejects ' + ','.join(rej)] if rej else []))
    z = [k for k in VB if k[0] == 0.0 and k in ok] or [k for k in VB if k[0] == 0.0]
    k0 = max(z, key=lambda k: (np.mean(VB[k]), -cost(k)))
    b0 = {'g': list(full(k0)), 'b': 0.0, 'val_mean': float(np.mean(VB[k0])), 'eligible': k0 in ok}
    return full(kB), kB[0], dict(ib, val_mean=float(np.mean(VB[kB]))), gf, b0, rej
