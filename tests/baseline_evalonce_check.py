"""Baselines, eval-once: load a run's saved encoder + predictor (strict) with flex_www.baselines, evaluate once, compare
the cell metric with the original run log at the saved epoch (the first strict maximum of the trainer's stop metric on
validation, per run).

    FLEX_REF=<reference tree> python tests/baseline_evalonce_check.py --method IRM --cell cora_PA_0_50_100 [--seed 1 --seed 2]
A strict-load key mismatch (the original trainer loaded non-strictly) is reported, then the load is retried non-strictly
so the values are still compared.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402
from evalonce_check import from_log  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def check(method, cell, device, seeds=(1, 2)):
    import torch
    from flex_www import baselines as B
    from flex_www.baselines.run import checkpoint_paths, evaluate_ctx, load_config
    from flex_www.train import Hooks, prepare
    root = R.data_root()
    cfg, spec = load_config(REPO / 'configs/baselines' / method / f'{cell}.yaml', data_root=str(root))
    ck = R.REF / cfg['reference']['checkpoint_dir']
    log = R.REF / cfg['reference']['run_log']
    opts = cfg.get('method_opts', {})
    h, mod = B.hooks(method, opts), B.module(method)
    t0, out, bs = time.time(), [], None
    dev = torch.device(device)
    for run_idx, s in enumerate(spec.seeds()):
        if s not in seeds:
            # the batch-size halving carries over between seeds; it depends only on the split sizes, which are fixed
            # off Amazon, so skipped seeds change nothing
            continue
        ctx = prepare(spec, s, run_idx, dev, Hooks(build_models=h.build_models), bs, warm_start=False)
        bs = ctx.batch_sizes
        if mod is not None and hasattr(mod, 'before_load'):
            mod.before_load(ctx, opts)
        mp, pp = checkpoint_paths(ck, spec, ctx.split.dataset, run_idx, method)
        mismatch = {}
        for name, m_, p_ in (('model', ctx.model, mp), ('predictor', ctx.predictor, pp)):
            st = torch.load(p_, map_location=dev)
            try:
                m_.load_state_dict(st, strict=True)
            except RuntimeError as e:
                mismatch[name] = str(e).split('\n', 1)[-1][:500]
                m_.load_state_dict(st, strict=False)
        res, _ = evaluate_ctx(ctx, h)
        metric, stop = ctx.split.metric, ctx.split.stop_metric
        ref = from_log(log, metric, stop)[run_idx]
        got = tuple(float(x) for x in res[metric][1:3])
        row = {'seed': s, 'got': got, 'ref': ref, 'maxdiff': max(abs(a - b) for a, b in zip(got, ref)),
               'strict_mismatch': mismatch or None, 'checkpoint': mp.name}
        print(json.dumps(row), flush=True)
        out.append(row)
    return {'method': method, 'cell': cell, 'maxdiff': max(r['maxdiff'] for r in out),
            'strict_ok': all(r['strict_mismatch'] is None for r in out), 'runtime_s': round(time.time() - t0, 1), 'seeds': out}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--seed', type=int, action='append')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    r = check(a.method, a.cell, a.device, tuple(a.seed or (1, 2)))
    print(json.dumps({k: v for k, v in r.items() if k != 'seeds'}), flush=True)
