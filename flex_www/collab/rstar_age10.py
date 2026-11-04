"""R* + age10: the GCN re-ranked by a fixed residual stack, on the age10-edited test-time graph.

What each piece contributes
  GCN      tb = the GCN's pair score; on the test side it runs on the age10-edited test-time graph ("+age10")
  B6       td = a logistic residual model ("B6", fitted once on a 2015 history and frozen) over pair descriptors
           (pair_features.b6_features) + the GCN score + a feature-teacher logit; it corrects the GCN toward pairs that
           actually form next year. base = tb + a6 * td, a6 picked on validation half B (fuse.choose)
  twin     a frozen generative scorer's z-score (DiGress "twin"; a data input of this release) added with weight `at`
  ACT      activity penalty: -(log1p(years since u last published) + log1p(same for v)), standardised; stale authors rarely
           co-author next year
  REFA     the residual of an activity-aware refit of B6 (mA) over td
  R*       one of three candidate combinations of base, ACT and REFA (weights g1-g3 and the candidate chosen on half B):
             ACTPEN_FUSE    base + g1 a6 ACT
             ACT_REFIT      base + g2 a6 REFA
             ACT_REFIT_FUSE base + g2 a6 REFA + g3 a6 ACT
Every scalar (a6, at, g1, g2, g3, the candidate) is picked per (GCN seed, twin seed) run on validation half B with the
validation-time graph; the test composite applies them to the test-time channels. dtypes follow the original runs
(float32 test channels, float64 validation channels) because Hits@K compares scores exactly.
"""
import numpy as np

from . import fuse
from .data import canon, node_activity
from .pair_features import b6_features

RSTAR_CANDS = ('ACTPEN_FUSE', 'ACT_REFIT', 'ACT_REFIT_FUSE')   # order = simplicity (tie-break)
FEATURE_KEY = 'semantic_rrwp'                                  # B6 input block (+ GCN score + teacher logit)


# ---------------------------------------------------------------------------------------------------------- frozen models
class Linear:
    """A fitted standardiser + logistic regression, applied exactly as scikit-learn does (in-place standardisation in the
    input dtype, float64 decision values)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    def decision(self, X):
        X = np.array(X, copy=True)
        X -= self.mean
        X /= self.scale
        return (X @ self.coef.T + self.intercept).ravel()


class RStarModels:
    """B6 residual model (m6), its feature teacher, and the activity-aware refit (mA); frozen inputs (rstar_models.npz)."""

    def __init__(self, path):
        Z = np.load(path)
        self.m6, self.teacher, self.mA = (Linear(Z[f'{k}_mean'], Z[f'{k}_scale'], Z[f'{k}_coef'], Z[f'{k}_intercept'])
                                          for k in ('m6', 'teacher', 'mA'))

    def teacher_logit(self, feats):
        p = 1 / (1 + np.exp(-self.teacher.decision(feats['semantic'])))
        return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

    def residual_x(self, feats, tb, tl):
        return np.column_stack([feats[FEATURE_KEY], tb[:, None], tl[:, None]])

    def channels(self, feats, tb_by_seed, ys, pairs):
        """{'tb', 'td', 'refa'} per seed (float32) and FA (float32) for pairs: B6 residual, activity refit, activity."""
        u, v = pairs[:, 0], pairs[:, 1]
        tl = self.teacher_logit(feats)
        XA = act_cols(ys[u], ys[v])
        out = {'tb': {}, 'td': {}, 'refa': {}, 'FA': (-(np.log1p(ys[u]) + np.log1p(ys[v]))).astype(np.float32)}
        for s, tb in tb_by_seed.items():
            X = self.residual_x(feats, tb, tl)
            td = self.m6.decision(X).astype(np.float32)
            out['tb'][s], out['td'][s] = tb, td
            out['refa'][s] = (self.mA.decision(np.column_stack([X, XA])) - td).astype(np.float32)
        return out


def act_bin(st):
    return np.select([st == 0, st == 1, st == 2, st <= 4, st < 99], [0, 1, 2, 3, 4], 5)


def act_cols(ysu, ysv):
    """Activity design of mA: log1p of the endpoints' min / max years-since-active and their one-hot activity bins."""
    ysmin, ysmax = np.minimum(ysu, ysv), np.maximum(ysu, ysv)
    oh = lambda b: (act_bin(b.astype(int))[:, None] == np.arange(1, 6)[None, :]).astype(float)  # noqa: E731
    return np.column_stack([np.log1p(ysmin), np.log1p(ysmax), oh(ysmin), oh(ysmax)])


# ---------------------------------------------------------------------------------------------------------- graphs
def test_world(c):
    """B6 history (canonical rows + dated years, the validation edges dated 2018) and activity of the age10 test graph."""
    m, _ = c.test_age10()
    mE = m[:len(c.E)]
    vp = canon(c.Ev)
    hist = canon(np.vstack([canon(c.E[mE]), vp]))
    years = np.r_[c.Y[mE], np.full(len(vp), 2018, np.int64)]
    rows, yrs, _ = c.test_rows()
    return hist, years, node_activity(rows[m], yrs[m], T=2018)


def val_world(c, age10):
    """Validation-time history (train rows, optionally age10-edited) and activity (T = 2017)."""
    m = c.val_age10() if age10 else np.ones(len(c.E), bool)
    return canon(c.E[m]), c.Y[m], node_activity(c.E[m], c.Y[m], T=2017)


# ---------------------------------------------------------------------------------------------------------- R*
def cands(base, ACT, REFA, a, g1, g2, g3):
    return {'ACTPEN_FUSE': base + g1 * a * ACT, 'ACT_REFIT': base + g2 * a * REFA, 'ACT_REFIT_FUSE': base + g2 * a * REFA + g3 * a * ACT}


def pick_scalars(sv, ACT, REFA, a, sel):
    """g1, g2, g3 and the R* candidate on half B."""
    g1, _ = fuse.pick(sv, ACT, a, sel, fuse.SIGNED)
    g2, _ = fuse.pick(sv, REFA, a, sel, fuse.GAMMAS)
    S2 = sv + g2 * a * REFA
    g3, _ = fuse.pick(S2, ACT, a, sel, fuse.SIGNED)
    cand = cands(sv, ACT, REFA, a, g1, g2, g3)
    best = max(RSTAR_CANDS, key=lambda k: (fuse.nrcrit(sel, sv, cand[k]), -RSTAR_CANDS.index(k)))
    return {'g1': float(g1), 'g2': float(g2), 'g3': float(g3), 'best': best}


def validation_runs(ch_val, twin_vz, npB, nrows, seeds, twins):
    """Per (GCN seed, twin seed) run, every R* scalar picked on half B from the validation-time channels (unedited graph)."""
    sel = fuse.ESet(npB, nrows)
    FA = ch_val['FA64']
    mu, sd = float(FA.mean()), float(FA.std())
    runs = []
    for s in seeds:
        vb, vd = ch_val['tb'][s], ch_val['td'][s]
        a6 = fuse.choose(vb, vd, npB)
        bv = vb + a6 * vd
        csd = float(vd.std())
        for ts in twins:
            zv = twin_vz[ts]
            at = fuse.choose(bv, zv, npB)
            sc = pick_scalars(bv + at * zv, csd * (FA - mu) / sd, ch_val['refa64'][s], a6, sel)
            runs.append({'gcn_seed': s, 'twin_seed': ts, 'a6': float(a6), 'at': float(at), 'csd': csd, 'act_mu': mu, 'act_sd': sd, **sc})
    return runs


def test_composite(ch, run, twin_z):
    """R* + age10 test score of one run (float32 arithmetic, as recorded)."""
    s = run['gcn_seed']
    base = ch['tb'][s] + run['a6'] * ch['td'][s] + run['at'] * twin_z
    ACT = run['csd'] * (ch['FA'] - run['act_mu']) / run['act_sd']
    return cands(base, ACT, ch['refa'][s], run['a6'], run['g1'], run['g2'], run['g3'])[run['best']].astype(np.float64)


def validation_composite(ch, run, twin_term):
    """R* + age10 on half B (validation-time age10 graph; float64), the base of the FLEX channel picks."""
    s = run['gcn_seed']
    base = ch['tb'][s].astype(np.float64) + run['a6'] * ch['td'][s].astype(np.float64) + twin_term
    ACT = run['csd'] * (ch['FA'].astype(np.float64) - run['act_mu']) / run['act_sd']
    return cands(base, ACT, ch['refa'][s].astype(np.float64), run['a6'], run['g1'], run['g2'], run['g3'])[run['best']]


def twin_term(ch_val_unedited, run, sv, npB):
    """The twin's additive term on half B for one run: (bv + at * z) - bv, with a6 / at re-chosen on the unedited channels."""
    s = run['gcn_seed']
    vb = ch_val_unedited['tb'][s].astype(np.float64)
    vd = ch_val_unedited['td'][s]
    bv = vb + fuse.choose(vb, vd, npB) * vd
    at = fuse.choose(bv, sv, npB)
    return (bv + at * sv) - bv


def twin_z(path, ts):
    """Twin z-scores (validation half B, test) standardised on its half-B scores."""
    d = np.load(path)
    sv, st = d[f'validB{ts}'], d[f'test{ts}']
    mu, sd = sv.mean(), max(sv.std(), 1e-8)
    return (sv - mu) / sd, (st - mu) / sd


def b6_channels(models, c, gcn, which):
    """R* channels on one side. which: 'test' (age10 test graph, PT), 'val' / 'val_age10' (half B).
    gcn: {seed: GCN scores of these pairs on the same graph}."""
    if which == 'test':
        hist, years, ys = test_world(c)
        pairs = c.PT
    else:
        hist, years, ys = val_world(c, age10=(which == 'val_age10'))
        pairs = c.PB
    feats = b6_features(hist, years, pairs, c.x)
    ch = models.channels(feats, gcn, ys, pairs)
    if which == 'val':      # the scalar picks read the activity term and REFA in float64 (as fitted)
        u, v = pairs[:, 0], pairs[:, 1]
        ch['FA64'] = -(np.log1p(ys[u]) + np.log1p(ys[v]))
        tl = models.teacher_logit(feats)
        XA = act_cols(ys[u], ys[v])
        ch['refa64'] = {s: models.mA.decision(np.column_stack([models.residual_x(feats, tb, tl), XA])) - ch['td'][s]
                        for s, tb in gcn.items()}
    return ch

