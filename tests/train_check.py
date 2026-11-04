"""Training fidelity: retrain seed 1 of a row's encoder with flex_www and compare the per-epoch headline metric
(valid, test) with the original trainer's log (Run 01 lines, 2-decimal percent = the 4-decimal value).

    FLEX_REF=<reference tree> python tests/train_check.py --cell citeseer_PA_100_50_0 --log <training log> [--epochs N]
AUG rows train with the replay loss (as the original row did). GPU kernels can be non-deterministic, so the report gives
the first epoch that differs instead of asserting.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def log_curve(path, metric, run=1):
    pat = re.compile(rf'Run: {run:02d}, {re.escape(metric)}Epoch: (\d+),.*Valid: ([\d.]+)%, Test: ([\d.]+)%')
    out = {}
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            out[int(m.group(1))] = (round(float(m.group(2)) / 100, 4), round(float(m.group(3)) / 100, 4))
    return out


def check(cell, log, device, epochs=None):
    import torch
    from flex_www.data import cell_info
    from flex_www.flex.replay import Replay
    from flex_www.flex.run import AG_ROWS, AUG_ROWS, SP_ROWS, load_config
    from flex_www.train import Hooks, fit, prepare
    root = R.data_root()
    cfg = load_config(REPO / 'configs/flex' / f'{cell}.yaml', data_root=str(root), warm_start_dir=str(R.REF / 'baseline_models'))
    cfg.spec.runs = 1
    if epochs:
        cfg.spec.epochs = epochs
    hooks = Hooks()
    if cfg.row in AUG_ROWS:
        rp = Replay(cell_info(cell), pa_only=cfg.row in AG_ROWS, sp=cfg.row in SP_ROWS)
        hooks.extra_loss = lambda ctx, ep: rp.loss_factory(ctx.model, ctx.predictor, ctx.graph.data, ctx.split.valid_pos)
    curve = {}

    def on_eval(ctx, n, ep, res, P):
        curve[ep] = tuple(float(x) for x in res[ctx.split.metric][1:3])
    hooks.on_eval = on_eval
    ctx = prepare(cfg.spec, 1, 0, torch.device(device), hooks)
    fit(ctx, hooks, log=lambda *a: None)
    ref = log_curve(log, ctx.split.metric)
    eps = sorted(set(curve) & set(ref))
    bad = [e for e in eps if max(abs(a - b) for a, b in zip(curve[e], ref[e])) > 1e-6]
    return {'cell': cell, 'epochs_compared': len(eps), 'first_diff_epoch': bad[0] if bad else None, 'n_diff': len(bad),
            'last': {e: (curve[e], ref[e]) for e in eps[-2:]}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    print(json.dumps(check(a.cell, a.log, a.device, a.epochs)), flush=True)
