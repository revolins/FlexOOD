"""The LPShift cell's named shift: axis, direction and validation folds along it.

LPShift orders train -> validation -> test along one heuristic in one direction, and the cell key names both
(e.g. CN_2_1_0 = test pairs have FEWER common neighbours). On the TRAIN message-passing graph:
  axis value  CN = common neighbours; PA = log1p du + log1p dv; SP = closeness = -d with the walk-class distance
              d = 2 (CN > 0), 3 (a 3-walk), 4 (a 4-walk), 5 (none of these) -- or, with the SP columns on, the exact
              distance class incl. 6 = disconnected.
  sign        +1 when the test split lies at HIGHER axis values than train, -1 otherwise.
  fold        validation positives strictly beyond the FOLD_Q quantile along the sign; if fewer than MIN_STRICT of them
              (discrete axes with a large tie class), the tie class is included.
Only validation positives are used (as for any model selection); test pairs are never read.
"""
import numpy as np

from . import features as F

FOLD_Q = 0.30
MIN_STRICT = 0.10
MIN_FOLD = 100          # threshold_v2: a fold with fewer validation positives falls back to all of validation
FLAT_AUC = 0.60         # threshold_v2: train -> validation AUC along the signed axis below this = no visible shift


def spec(cell):
    """(axis, sign) for an LPShift cell (cell_info dict), else None."""
    if not cell or cell.get('family') != 'lpshift' or cell.get('shift') not in ('CN', 'PA', 'SP'):
        return None
    parts = cell['key'].split('_')         # e.g. ['ogbl-collab', 'SP', '026', '017', '00']; SP values are 0.xx
    h = parts[1]
    v = [float('0.' + p[1:]) if (h == 'SP' and len(p) > 1 and p[0] == '0') else float(p) for p in parts[2:]]
    return h, int(np.sign(v[-1] - v[0]))


def sp_dist(X, sp=False):
    if sp:
        return X[:, F.cols(True).index('spc')].astype(int)
    c = F.BASE_COLS.index
    floor = np.log(F.EPS) + 1e-3
    return np.where(X[:, c('cn')] > 0, 2, np.where(X[:, c('w3pos')] > 0, 3, np.where(X[:, c('RW4')] > floor, 4, 5)))


def axis_value(h, X, sp=False):
    """X = pair_feats rows on the train graph."""
    if h == 'CN':
        return X[:, F.BASE_COLS.index('cn')].astype(np.float64)
    if h == 'PA':
        return X[:, F.BASE_COLS.index('dsum')].astype(np.float64)
    return -sp_dist(X, sp).astype(np.float64)


def threshold(axis, A, val_pos, sp=False):
    """(fold mask over val_pos, threshold, info). axis = spec(cell); val_pos (k, 2) validation positives."""
    h, sign = axis
    X = F.pair_feats(A, val_pos, sp=sp)
    v = axis_value(h, X, sp)
    q = float(np.quantile(v, FOLD_Q if sign < 0 else 1 - FOLD_Q))
    strict = v < q if sign < 0 else v > q
    m = strict if strict.mean() >= MIN_STRICT else (v <= q if sign < 0 else v >= q)
    info = {'axis': h, 'sign': sign, 'threshold': q, 'strict': bool(m is strict), 'fold_share': float(m.mean()), 'n_val': int(len(v))}
    if h == 'SP':
        info['fold_far_share'] = float((sp_dist(X[m], sp) >= 5).mean()) if m.any() else 0.0
    if h == 'PA':
        deg = np.asarray(A.sum(1)).ravel()
        P = np.asarray(val_pos, np.int64)[m]
        info['fold_deg_median'] = [float(np.median(deg[P[:, 0]])), float(np.median(deg[P[:, 1]]))]
    return m, q, info


def threshold_v2(axis, A, val_pos, X_train_pos, min_fold=MIN_FOLD, sp=False):
    """threshold() with two fallbacks to all of validation (mask None):
      flat   AUC(train pseudo-positives vs validation positives) along sign * axis < FLAT_AUC (the named shift is not
             visible between train and validation on the train graph);
      small  fewer than min_fold validation positives in the fold.
    X_train_pos: pair_feats rows of the train pseudo-positives (each on the train graph minus its own fold)."""
    from sklearn.metrics import roc_auc_score
    m, q, info = threshold(axis, A, val_pos, sp)
    vt = axis[1] * axis_value(axis[0], X_train_pos, sp)
    vv = axis[1] * axis_value(axis[0], F.pair_feats(A, val_pos, sp=sp), sp)
    auc = float(roc_auc_score(np.r_[np.zeros(len(vt)), np.ones(len(vv))], np.r_[vt, vv])) if len(set(np.r_[vt, vv])) > 1 else 0.5
    info.update(train_valid_auc=auc, n_fold=int(m.sum()))
    if auc < FLAT_AUC:
        info['fallback'] = 'flat'
        return None, q, info
    if m.sum() < min_fold:
        info['fallback'] = 'small'
        return None, q, info
    info['fallback'] = None
    return m, q, info


def fold_degree_pairs(A, val_pos, mask):
    deg = np.asarray(A.sum(1)).ravel()
    P = np.asarray(val_pos, np.int64)[mask]
    return np.stack([deg[P[:, 0]], deg[P[:, 1]]], 1).astype(int)
