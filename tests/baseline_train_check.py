"""Baselines, training fidelity: retrain seed 1 of a (method, cell) config with flex_www.baselines and compare the
per-epoch (valid, test) of the cell metric with the original run log ('Run: 01, <metric>Epoch: N, ... Valid: v%, Test: t%'),
2-decimal percent = the 4-decimal value. Reports the first epoch that differs (GPU kernels can be non-deterministic).

    FLEX_REF=<reference tree> python tests/baseline_train_check.py --method IRM --cell cora_PA_0_50_100 [--epochs N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402
from train_check import log_curve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def check(method, cell, device, epochs=None, log=None):
    import torch
    from flex_www import baselines as B
    from flex_www.baselines.run import load_config
    from flex_www.train import fit, prepare
    root = R.data_root()
    cfg, spec = load_config(REPO / 'configs/baselines' / method / f'{cell}.yaml', data_root=str(root),
                            warm_start_dir=str(R.REF / 'baseline_models'))
    spec.runs, spec.seed = 1, 1
    if epochs:
        spec.epochs = epochs
    h = B.hooks(method, cfg.get('method_opts', {}))
    curve, losses = {}, {}
    inner = h.on_eval

    def on_eval(ctx, n, ep, res, P):
        curve[ep] = tuple(float(x) for x in res[ctx.split.metric][1:3])
        if inner:
            inner(ctx, n, ep, res, P)
    h.on_eval = on_eval
    t0 = time.time()
    ctx = prepare(spec, 1, 0, torch.device(device), h)
    fit(ctx, h, log=lambda *a: None)
    ref = log_curve(log or (R.REF / cfg['reference']['run_log']), ctx.split.metric)
    eps = sorted(set(curve) & set(ref))
    bad = [e for e in eps if max(abs(a - b) for a, b in zip(curve[e], ref[e])) > 1e-6]
    return {'method': method, 'cell': cell, 'epochs_compared': len(eps), 'epochs_trained': max(curve) if curve else 0,
            'epochs_logged_run1': max(ref) if ref else 0, 'first_diff_epoch': bad[0] if bad else None, 'n_diff': len(bad),
            'first_diff': {bad[0]: (curve[bad[0]], ref[bad[0]])} if bad else None,
            'last': {e: (curve[e], ref[e]) for e in eps[-2:]}, 'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--log', help='reference log (default: the config\'s run log)')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    print(json.dumps(check(a.method, a.cell, a.device, a.epochs, a.log)), flush=True)
