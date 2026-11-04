"""ISS10: an invariant cross-year link-formation channel, trained on past years of the collab graph only.

Environments: one per (label year Y in 2012..fit_max, protocol), built from the train rows (Y = 2018 also uses the
validation edges as its labels, which the OGB rule allows for a model deployed at test time):
  protocol S1a   graph = rows <= Y-1                                           (how OGB builds the validation-time input)
           S2b   graph = rows <= Y-2 + year Y-1 rows between nodes seen <= Y-2 (how OGB thins the test-time input)
  age10 is applied inside each env's history; positives = year-Y pairs between already-seen nodes that were never linked
  before ("cold" positives: the pairs a recency-driven ranker misses), negatives = 100,000 uniform non-edges.
Each env is described by the 31 node-feature-free columns (pair_features.XCOLS), standardised on that env's negatives.

ISS = invariant stability selection: per env, 50 half-sample L1-logistic fits; keep the columns selected in >= 80% of all
fits whose sign agrees in every env. Then an MLP (128-64, dropout 0.1) is trained on the kept columns with a per-env
class-balanced BCE (cold positives on CN = 0 pairs weighted 3x) and a per-env bias that prediction drops.

Channel z: mean over the 10 seed models of the MLP logit, standardised on 100,000 uniform non-edges of the target graph
("anchors", label-free); inputs are standardised on the same anchors. Validation picks use the models fit on years <=
2017 and the validation-time graph; the test channel uses the models fit on years <= 2018 and the test-time graph.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import config
from .data import N, pair_key
from .pair_features import XCOLS, DatedGraph

NNEG = 100000
ENV_SEED = 20261015
PROTOS = ('S2b', 'S1a')
AGE = 10
FIT_MIN = 2012                 # first label year of the environments
NEG_PER_ENV = 50000
VAL_ANCHOR_SEED = 20261016
TEST_ANCHOR_SEED = 20261018


# ---------------------------------------------------------------------------------------------------------- environments
def canon(p):
    """Sorted endpoints, self-loops removed, unique rows."""
    p = np.sort(np.asarray(p, np.int64).reshape(-1, 2), 1)
    return np.unique(p[p[:, 0] != p[:, 1]], axis=0)


def first_year(E, yr):
    f = np.full(N, 9999)
    np.minimum.at(f, E[:, 0], yr)
    np.minimum.at(f, E[:, 1], yr)
    return f


def env_graph(Y, proto, E, yr):
    """Row mask of the history visible before label year Y under the protocol, with age10 computed inside it."""
    if proto == 'S1a':
        m = yr <= Y - 1
    else:
        f = first_year(E[yr <= Y - 2], yr[yr <= Y - 2])
        s2 = f <= Y - 2
        m = (yr <= Y - 2) | ((yr == Y - 1) & s2[E[:, 0]] & s2[E[:, 1]])
    k = pair_key(E)
    uk, inv = np.unique(k, return_inverse=True)
    last = np.full(len(uk), -1)
    np.maximum.at(last, inv[m], yr[m])
    return m & ~(last[inv] <= (Y - 1) - AGE)


def uniform_nonedges(rng, ban, m=NNEG):
    """m uniform node pairs (u != v) whose key is not in `ban`, in draw order without repeats."""
    got = np.zeros(0, np.int64)
    while len(got) < m:
        u, v = rng.integers(0, N, 4 * m), rng.integers(0, N, 4 * m)
        k = pair_key(np.c_[u, v][u != v])
        k = k[~np.isin(k, ban) & ~np.isin(k, got)]
        _, fi = np.unique(k, return_index=True)
        got = np.r_[got, k[np.sort(fi)]]
    return np.c_[got[:m] // N, got[:m] % N]


def build_env(Y, proto, E, yr, W, valid_edges=None):
    """One environment: pairs P (positives first), x-free features X, labels y, cold flags."""
    rng = np.random.default_rng(ENV_SEED + 10 * Y + PROTOS.index(proto))
    m = env_graph(Y, proto, E, yr)
    Pall = canon(E[yr == Y]) if Y <= 2017 else canon(valid_edges)
    lim = Y - 2 if proto == 'S2b' else Y - 1
    seen = np.zeros(N, bool)
    seen[E[yr <= lim].ravel()] = True
    Pall = Pall[seen[Pall[:, 0]] & seen[Pall[:, 1]]]
    hist_keys = np.unique(pair_key(E[yr <= Y - 1]))
    cold = ~np.isin(pair_key(Pall), hist_keys)
    neg = uniform_nonedges(rng, np.unique(np.r_[hist_keys, pair_key(Pall)]))
    P = np.vstack([Pall, neg])
    X = DatedGraph(E[m], yr[m], W[m], Y).xfree(P)
    return {'P': P, 'X': X, 'y': np.r_[np.ones(len(Pall)), np.zeros(NNEG)].astype(np.float32),
            'cold': np.r_[cold, np.ones(NNEG, bool)], 'npos': len(Pall)}


def env_rows(env, Y, seed):
    """(X standardised on the env's negatives, y, CN = 0 flag): cold positives + 50,000 sub-sampled negatives."""
    npos = env['npos']
    X = env['X'].astype(np.float64)
    cold = env['cold'][:npos]
    neg = np.arange(npos, len(X))
    mu, sd = X[neg].mean(0), X[neg].std(0) + 1e-6
    ni = np.random.default_rng(seed * 1000 + Y).choice(neg, min(NEG_PER_ENV, len(neg)), replace=False)
    idx = np.r_[np.flatnonzero(cold), ni]
    y = np.r_[np.ones(int(cold.sum())), np.zeros(len(ni))]
    return ((X[idx] - mu) / sd).astype(np.float32), y.astype(np.float32), X[idx, XCOLS.index('cn')] == 0


# ---------------------------------------------------------------------------------------------------------- model
def make_mlp(d, hid=(128, 64), drop=0.1):
    import torch.nn as nn
    layers, k = [], d
    for h in hid:
        layers += [nn.Linear(k, h), nn.ReLU(), nn.Dropout(drop)]
        k = h
    return nn.Sequential(*layers, nn.Linear(k, 1))


def iss_select(data, cfg):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(cfg['seed'])
    d = data[0][0].shape[1]
    sel, sgn, nfit = np.zeros(d), [], 0
    for X, y, _ in data:
        se = np.zeros(d)
        for _ in range(cfg['iss_boot']):
            i = rng.choice(len(X), len(X) // 2, replace=False)
            c = LogisticRegression(penalty='l1', C=cfg['iss_C'], solver='liblinear', class_weight='balanced',
                                   max_iter=200).fit(X[i], y[i]).coef_.ravel()
            sel += c != 0
            se += np.sign(c)
            nfit += 1
        sgn.append(np.sign(se))
    S = np.array(sgn)
    consistent = (np.abs(S.sum(0)) == (S != 0).sum(0)) & ((S != 0).sum(0) == len(data))
    return (sel / nfit >= cfg['iss_keep']) & consistent


def env_batch(data, rng, bp, bn):
    Xs, ys, es, ws = [], [], [], []
    for k, (X, y, w) in enumerate(data):
        p, q = np.flatnonzero(y > 0.5), np.flatnonzero(y <= 0.5)
        i = np.r_[rng.choice(p, min(bp, len(p)), replace=False), rng.choice(q, min(bn, len(q)), replace=False)]
        Xs.append(X[i])
        ys.append(y[i])
        es.append(np.full(len(i), k))
        ws.append(w[i])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(es), np.concatenate(ws)


def balanced_env_risk(loss, y, env, n_env, w):
    import torch
    out = []
    for k in range(n_env):
        m = env == k
        mp, mn = m & (y > 0.5), m & (y <= 0.5)
        rp = (loss[mp] * w[mp]).sum() / w[mp].sum().clamp_min(1e-12)
        rn = (loss[mn] * w[mn]).sum() / w[mn].sum().clamp_min(1e-12)
        out.append(0.5 * (rp + rn))
    return torch.stack(out)


def train_mlp(data, cfg, dev):
    import torch
    import torch.nn.functional as Fn
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    n_env, d = len(data), data[0][0].shape[1]
    net = make_mlp(d, drop=cfg['drop']).to(dev)
    bias = torch.zeros(n_env, device=dev, requires_grad=True)          # per-env label prior, not used at prediction
    opt = torch.optim.Adam([{'params': net.parameters(), 'weight_decay': cfg['wd']}, {'params': [bias], 'weight_decay': 0.0}],
                           lr=cfg['lr'])
    for _ in range(cfg['steps']):
        Xb, yb, eb, wb = env_batch(data, rng, cfg['bp'], cfg['bn'])
        Xb, yb, wb = (torch.as_tensor(a, device=dev) for a in (Xb, yb, wb))
        eb = torch.as_tensor(eb, device=dev, dtype=torch.int64)
        s = torch.ones(1, device=dev, requires_grad=True)
        logit = s * net(Xb).squeeze(1) + bias[eb]
        loss = Fn.binary_cross_entropy_with_logits(logit, yb, reduction='none')
        obj = balanced_env_risk(loss, yb, eb, n_env, wb).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
    return net.eval()


def cfg_for(seed, fit_max):
    """Training config of one ISS model (configs/collab/collab.yaml 'iss'); its hash names the model directory."""
    return {'fit_max': fit_max, **config()['iss']['shared'], 'seed': seed}


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def train_iss(envs, seed, fit_max, dev):
    """Train one ISS model on the real envs with label years FIT_MIN..fit_max. Returns {'state', 'keep', 'd'}."""
    cfg = cfg_for(seed, fit_max)
    data = []
    for Y in range(FIT_MIN, fit_max + 1):
        for p in PROTOS:
            X, y, cn0 = env_rows(envs[(Y, p)], Y, seed)
            data.append((X, y, np.where((y > 0.5) & cn0, cfg['w_far'], 1.0).astype(np.float32)))
    keep = iss_select(data, cfg)
    net = train_mlp([(X[:, keep], y, w) for X, y, w in data], cfg, dev)
    return {'state': {k: v.cpu() for k, v in net.state_dict().items()}, 'keep': keep, 'd': int(keep.sum())}


# ---------------------------------------------------------------------------------------------------------- channel
def load_models(root, fit_max):
    """ISS model files (<hash>/model.pt + cfg.json) of one fit year, sorted by hash (the averaging order)."""
    out = []
    for d in sorted(Path(root).iterdir()):
        cfg = json.loads((d / 'cfg.json').read_text())
        if cfg['fit_max'] == fit_max:
            out.append(d / 'model.pt')
    return out


def standardise(X, X_anchor):
    mu, sd = X_anchor.mean(0), X_anchor.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def predict(ck, X, dev, chunk=200000):
    import torch
    net = make_mlp(ck['d'])
    net.load_state_dict(ck['state'])
    net.to(dev).eval()
    X = X[:, ck['keep']]
    out = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            out.append(net(torch.as_tensor(X[s:s + chunk], device=dev)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def channel(models, X, XA, dev):
    """z of pairs X: mean ensemble logit, standardised on the anchors XA."""
    import torch
    Xs, XAs = standardise(X, XA), standardise(XA, XA)
    cks = [m if isinstance(m, dict) else torch.load(m, map_location='cpu', weights_only=False) for m in models]
    S = np.mean([np.r_[predict(ck, Xs, dev), predict(ck, XAs, dev)] for ck in cks], 0)
    s, sa = S[:len(X)], S[len(X):]
    return (s - sa.mean()) / (sa.std() + 1e-12)


def val_features(c):
    """x-free features of half B and of the validation anchors on the validation-time age10 graph (rows <= 2017)."""
    m = env_graph(2018, 'S1a', c.E, c.Y)
    A = uniform_nonedges(np.random.default_rng(VAL_ANCHOR_SEED), np.unique(pair_key(c.E[m])))
    X = DatedGraph(c.E[m], c.Y[m], c.W[m], 2018).xfree(np.vstack([c.PB, A]))
    return X[:len(c.PB)], X[len(c.PB):]


def test_features(c):
    """x-free features of the test pairs and of the test anchors on the age10 test-time graph (train + validation rows)."""
    rows, yrs, wts = c.test_rows()
    m, _ = c.test_age10()
    rows, yrs, wts = rows[m], yrs[m], wts[m]
    A = uniform_nonedges(np.random.default_rng(TEST_ANCHOR_SEED), np.unique(pair_key(rows)))
    X = DatedGraph(rows, yrs, wts, 2019).xfree(np.vstack([c.PT, A]))
    return X[:len(c.PT)], X[len(c.PT):]
