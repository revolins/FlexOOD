"""Real ogbl-collab rows: FLEX (GCN + R* + age10 + ISS10 + AD + CN_REC2), the age10 baseline rows and the NCN / Refined-GAE
backbone rows (alone and with FLEX).

    python -m flex_www.collab.run --mode weights   --weights review_weights/ogbl-collab --inputs review_data/ogbl-collab
    python -m flex_www.collab.run --mode train     --weights review_weights/ogbl-collab --inputs review_data/ogbl-collab
    python -m flex_www.collab.run --mode baselines --weights <camera-ready dir>/ogbl-collab
    python -m flex_www.collab.run --mode backbones --weights <camera-ready dir>/ogbl-collab \
                                  --iss review_weights/ogbl-collab/iss --inputs review_data/ogbl-collab

Pipeline (10 GCN seeds x 3 twin seeds = 30 runs; a GCN seed's score is the mean over its 3 runs)
  1 GCN        the trained GCN of each seed scores half B (validation-time graph, plain and age10) and the test pairs
               (age10 test-time graph)                                                         gcn_scores
  2 R*+age10   B6 residual, twin, activity and REFA channels; per-run scalars picked on half B   rstar_age10
  3 ISS10      invariant cross-year channel z (10 models; fit <= 2017 for picks, <= 2018 for test) iss
  4 AD         AA-DC channel; CN_REC2 recent common-neighbour channel                          channels
  5 fusion     per run, on half B with the R*+age10 validation composite SR:
                 g    = pick(SR, ISS)            S1   = SR + g a6 csd z_ISS             (row ISS10)
                 g_ad = pick(S1, AD)             S_AD = S1 + g_ad a6 csd z_AD           (row ISS10+AD)
                 g_cn = pick(S_AD, CN_REC2)      S_CN = S_AD + g_cn a6 csd z_CN         (row ISS10+AD+CN_REC2 = the paper row)
               the same weights are applied to the test-time channels; Hits@50 / Hits@20 on the official test split
Nothing is fitted or selected on test.

weights   GCN checkpoints (<weights>/gcn/seed<s>_{model,predictor}.pt) and the 20 ISS models (<weights>/iss/<id>/) are
          loaded; everything else is recomputed from the dataset and the frozen inputs (--inputs: rstar_models.npz,
          twin_scores.npz). The ISS channel runs on a GPU (CPU float32 inference differs in the last bits).
train     as weights, but the 20 ISS models are retrained first (14 environments rebuilt from the train / validation
          rows). Still loaded, not retrained: the GCN checkpoints (flex_www.train with the recorded collab config), the R*
          models and the DiGress twin scores (frozen inputs).
baselines age10 rows: each GCN baseline's checkpoints (<weights>/baselines/<method>/seed<s>_*.pt) and gMPNN_ref
          (<weights>/gMPNN_ref/) score the age10 test-time graph; plus AA-DC on the same graph.
backbones NCN and RGAE (<weights>/<backbone>/seed<s>_*.pt): the backbone's age10 row, and backbone+FLEX = the backbone in
          the GCN slot of R* + age10 + ISS10 (steps 2-3, every scalar and g re-picked on half B for that backbone).
The baseline / backbone checkpoints are not part of the review bundle (scripts/stage_collab_inputs.py --baseline_weights).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import channels as CH
from . import fuse
from . import iss as ISS
from . import rstar_age10 as RS
from .data import load_collab
from .gcn_scores import score_checkpoints

SEEDS = tuple(range(1, 11))
TWINS = (1, 2, 3)
KS = (50, 20)
BASELINES = ('ERM', 'IRM', 'VRex', 'GroupDRO', 'CORAL', 'DANN', 'SRGNN', 'VGAE')
BACKBONES = ('NCN', 'RGAE')
GCN_JOBS = [('test_age10', 'PT'), ('val', 'PB'), ('val_age10', 'PB')]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def ckpts(root, seeds=SEEDS):
    root = Path(root)
    return {s: (root / f'seed{s}_model.pt', root / f'seed{s}_predictor.pt') for s in seeds}


# ---------------------------------------------------------------------------------------------------------- stages
def shared_channels(c, iss_root, inputs, device, iss_models=None, heuristics=True):
    """Backbone-free inputs: R* models, twin z, ISS z (both sides) and, with heuristics, AD and CN_REC2 (both sides)."""
    t0 = time.time()
    I = {'models': RS.RStarModels(Path(inputs) / 'rstar_models.npz'),
         'twin': {ts: RS.twin_z(Path(inputs) / 'twin_scores.npz', ts) for ts in TWINS}}
    if iss_models is None:
        iss_models = {f: ISS.load_models(iss_root, f) for f in (2017, 2018)}
    Xv, XAv = ISS.val_features(c)
    Xt, XAt = ISS.test_features(c)
    I['iss'] = (ISS.channel(iss_models[2017], Xv, XAv, device), ISS.channel(iss_models[2018], Xt, XAt, device).astype(np.float64))
    log('ISS channel', f'{time.time() - t0:.0f}s')
    if heuristics:
        ad = CH.AADC(c)
        I['ad'] = (ad.val_age10(), ad.test_age10())
        log('AA-DC channel', f'{time.time() - t0:.0f}s', 'anchor scale', ad.scale)
        I['cn'] = (CH.cn_rec2_val(c), CH.cn_rec2_test(c))
    return I


def rstar_channels(c, I, scores):
    """R* channels on the three sides from the slot model's scores {(graph, seed): scores}."""
    return {w: RS.b6_channels(I['models'], c, {s: scores[(gw, s)] for s in SEEDS}, w)
            for w, gw in (('test', 'test_age10'), ('val', 'val'), ('val_age10', 'val_age10'))}


def picks(c, I, ch, heuristics=True):
    """Per-run R* scalars and the FLEX channel weights (g; with heuristics also g_ad, g_cn), all on validation half B."""
    tw = I['twin']
    runs = RS.validation_runs(ch['val'], {ts: tw[ts][0] for ts in TWINS}, c.npB, len(c.PB), SEEDS, TWINS)
    sel = fuse.ESet(c.npB, len(c.PB))
    zv = I['iss'][0]
    units = {}
    if heuristics:
        ad = I['ad'][0]
        units['AD'] = (float(ad.mean()), max(float(ad.std()), 1e-12))
        zad = (ad - units['AD'][0]) / units['AD'][1]
        zcn, units['CN_REC2'] = fuse.unit(np.log1p(I['cn'][0]))
    for r in runs:
        a6, csd = r['a6'], r['csd']
        SR = RS.validation_composite(ch['val_age10'], r, RS.twin_term(ch['val'], r, tw[r['twin_seed']][0], c.npB))
        r['g'], r['fallback'] = fuse.pick(SR, csd * zv, a6, sel)
        if heuristics:
            S1 = SR + r['g'] * a6 * csd * zv
            r['g_ad'], r['fallback_ad'] = fuse.pick(S1, csd * zad, a6, sel)
            k = a6 * csd
            S_AD = SR + r['g'] * k * zv + r['g_ad'] * k * zad
            r['g_cn'], r['fallback_cn'] = fuse.pick(S_AD, csd * zcn, a6, sel)
    return runs, units


def test_rows(I, ch, runs, units, heuristics=True):
    """{row: [test score vector per run]}: R*+age10, ISS10 (and ISS10+AD, ISS10+AD+CN_REC2 with heuristics)."""
    tw, z = I['twin'], I['iss'][1]
    rows = {'RSTAR_AGE10': [], 'ISS10': []}
    if heuristics:
        zad = (I['ad'][1] - units['AD'][0]) / units['AD'][1]
        zcn = (np.log1p(I['cn'][1]) - units['CN_REC2'][0]) / units['CN_REC2'][1]
        rows.update({'ISS10+AD': [], 'ISS10+AD+CN_REC2': []})
    for r in runs:
        SR = RS.test_composite(ch['test'], r, tw[r['twin_seed']][1])
        S1 = SR + r['g'] * r['a6'] * r['csd'] * z
        rows['RSTAR_AGE10'].append(SR)
        rows['ISS10'].append(S1)
        if heuristics:
            S_AD = S1 + r['g_ad'] * r['a6'] * r['csd'] * zad
            rows['ISS10+AD'].append(S_AD)
            rows['ISS10+AD+CN_REC2'].append(S_AD + r['g_cn'] * (r['a6'] * r['csd']) * zcn)
    return rows


def hits_table(rows, run_seeds, npos):
    """{row: {'H50' | 'H20': {'mean', 'std', 'per_seed'}}}; per seed = mean over its runs and positives (std: ddof 1)."""
    run_seeds = np.asarray(run_seeds)
    out = {}
    for name, L in rows.items():
        out[name] = {}
        for K in KS:
            H = np.stack([fuse.test_hits(S, npos, K) for S in L])
            per = [float(H[run_seeds == s].mean()) for s in sorted(set(run_seeds.tolist()))]
            out[name][f'H{K}'] = {'mean': float(np.mean(per)), 'std': float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                                  'per_seed': per}
    return out


def fused_table(c, I, ch, heuristics=True):
    runs, units = picks(c, I, ch, heuristics)
    rows = test_rows(I, ch, runs, units, heuristics)
    return {'rows': hits_table(rows, [r['gcn_seed'] for r in runs], c.npos), 'runs': runs, 'units': units}


# ---------------------------------------------------------------------------------------------------------- modes
def flex(c, weights, inputs, device, iss_models=None):
    I = shared_channels(c, Path(weights) / 'iss', inputs, device, iss_models)
    g = score_checkpoints(c, ckpts(Path(weights) / 'gcn'), GCN_JOBS, device)
    return fused_table(c, I, rstar_channels(c, I, g))


def backbones(c, weights, iss_root, inputs, device, methods=BACKBONES):
    """{backbone: age10 row}, {backbone+FLEX: R*+age10+ISS10 with the backbone in the GCN slot}."""
    from .backbones import score_backbone
    I = shared_channels(c, iss_root, inputs, device, heuristics=False)
    table, runs = {}, {}
    for m in methods:
        sc, logits = score_backbone(c, m, ckpts(Path(weights) / m), device)
        log('backbone scored', m)
        table[m] = hits_table({m: [logits[s] for s in SEEDS]}, list(SEEDS), c.npos)[m]
        f = fused_table(c, I, rstar_channels(c, I, sc), heuristics=False)
        table[f'{m}+RSTAR_AGE10'] = f['rows']['RSTAR_AGE10']
        table[f'{m}+FLEX'] = f['rows']['ISS10']
        runs[m] = f['runs']
    return {'rows': table, 'runs': runs}


def train_iss_models(c, out, device):
    """Rebuild the 14 environments and retrain the 20 ISS models (seeds 1-10 x fit years 2017 / 2018) under <out>/iss."""
    envs = {}
    for Y in range(ISS.FIT_MIN, 2019):
        for p in ISS.PROTOS:
            envs[(Y, p)] = ISS.build_env(Y, p, c.E, c.Y, c.W, valid_edges=c.Ev)
            log('env', Y, p, 'positives', envs[(Y, p)]['npos'])
    models = {2017: [], 2018: []}
    for fit_max in (2017, 2018):
        for seed in SEEDS:
            cfg = ISS.cfg_for(seed, fit_max)
            h = ISS.cfg_hash(cfg)
            d = Path(out) / 'iss' / h
            d.mkdir(parents=True, exist_ok=True)
            ck = ISS.train_iss(envs, seed, fit_max, device)
            torch.save(ck, d / 'model.pt')
            (d / 'cfg.json').write_text(json.dumps({**cfg, 'hash': h}, indent=1))
            models[fit_max].append((h, ck))
            log('ISS trained', fit_max, seed, h, 'keeps', int(ck['d']))
    return {f: [ck for _, ck in sorted(v, key=lambda t: t[0])] for f, v in models.items()}


def baselines(c, weights, device, methods=BASELINES):
    """age10 rows: each GCN baseline's 10 checkpoints and gMPNN_ref on the age10 test-time graph, and AA-DC there."""
    from .backbones import score_gmpnn
    rows = {}
    for m in methods:
        g = score_checkpoints(c, ckpts(Path(weights) / 'baselines' / m), [('test_age10', 'PT')], device,
                              ood_method=None if m == 'ERM' else m)
        rows[m] = [g[('test_age10', s)] for s in SEEDS]
        log('baseline scored', m)
    g = score_gmpnn(c, ckpts(Path(weights) / 'gMPNN_ref'), device)
    rows['gMPNN_ref'] = [g[s] for s in SEEDS]
    table = hits_table(rows, list(SEEDS), c.npos)
    ad = CH.AADC(c).test_age10()
    table['AA-DC'] = {f'H{K}': {'mean': float(fuse.test_hits(ad, c.npos, K).mean()), 'std': 0.0} for K in KS}
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('weights', 'train', 'baselines', 'backbones'), required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--inputs', default='review_data/ogbl-collab')
    ap.add_argument('--iss', default='review_weights/ogbl-collab/iss', help='ISS models (backbones mode)')
    ap.add_argument('--out', default='output/collab')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data_root', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = load_collab(a.data_root)
    if a.mode == 'baselines':
        res = baselines(c, a.weights, dev)
    elif a.mode == 'backbones':
        res = backbones(c, a.weights, a.iss, a.inputs, dev)
    else:
        iss_models = train_iss_models(c, out, dev) if a.mode == 'train' else None
        res = flex(c, a.weights, a.inputs, dev, iss_models)
    (out / f'collab_{a.mode}.json').write_text(json.dumps(res, indent=1, default=float))
    for k, v in (res['rows'] if 'rows' in res else res).items():
        log(f'{k:18s} H50 {100 * v["H50"]["mean"]:.2f} +/- {100 * v["H50"]["std"]:.2f}   H20 {100 * v["H20"]["mean"]:.2f}')


if __name__ == '__main__':
    main()
