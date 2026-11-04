"""Eval-once: load a row's saved encoder + predictor per seed, evaluate once, compare with what the original run recorded
for the saved epoch.

    FLEX_REF=<reference tree> python tests/evalonce_check.py --cell ogbl-collab_CN_2_1_0 [--log <training log>] [--device cuda:0]

Reference values: rows whose checkpoint is the epoch FLEX fused (non-AUG rows whose trainer stop metric is the headline
metric) -> the run JSON's base_per_seed; otherwise (AUG-family rows: base_* is the best-stop-metric epoch, not the fused
one; Amazon: the trainer stops / saves on Hits@20, the cell reports Hits@50) -> --log, the original trainer's per-epoch
log, at the first strict maximum of the stop metric's validation per run. Rows with a capture dump also compare the
prediction tensors with the dumped ones (exact).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path


sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def from_log(path, metric, sel_metric=None):
    """Per run, `metric` (valid, test) at the epoch the trainer saved: the first strict maximum of `sel_metric`'s
    validation, from the trainer's 'Run: rr, <metric>Epoch: ee, ... Valid: v%, Test: t%' lines."""
    sel_metric = sel_metric or metric
    pat = re.compile(r'Run: (\d+), (\S+?)Epoch: (\d+),.*Valid: ([\d.]+)%, Test: ([\d.]+)%')
    vals, best = {}, {}
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        r, k, ep = int(m.group(1)), m.group(2), int(m.group(3))
        v, t = round(float(m.group(4)) / 100, 4), round(float(m.group(5)) / 100, 4)
        vals[(r, k, ep)] = (v, t)
        if k == sel_metric and (r not in best or v > best[r][0]):
            best[r] = (v, ep)
    return [vals[(r, metric, best[r][1])] for r in sorted(best)]


def check(cell, device, log=None):
    import torch
    import yaml
    from flex_www.evaluate import eval_once
    from flex_www.flex.run import AUG_ROWS, load_config, load_captures
    from flex_www.train import prepare
    root = R.data_root()
    c = yaml.safe_load((REPO / 'configs/flex' / f'{cell}.yaml').read_text())
    cfg = load_config(REPO / 'configs/flex' / f'{cell}.yaml', data_root=str(root))
    run_json = json.loads((R.REF / c['reference']['run_json']).read_text())
    metric = c['metric']
    aug = cfg.row in AUG_ROWS
    stop = 'MRR' if cell == 'dblp' else 'Hits@20'          # the trainer's stop metric (Split.stop_metric)
    use_log = aug or stop != metric
    ref = from_log(log, metric, stop) if use_log else [tuple(x) for x in run_json['base_per_seed'][metric]]
    caps = None
    if not use_log and c['reference']['caps']:
        meta, seeds_c, sel, *_ = load_captures(R.REF / c['reference']['caps'])
        caps = sel
    ck = R.REF / c['reference']['checkpoint_dir']
    t0 = time.time()
    out, bs = [], None
    for run_idx, s in enumerate(cfg.spec.seeds()):
        ctx = prepare(cfg.spec, s, run_idx, torch.device(device), None, bs, warm_start=False)
        bs = ctx.batch_sizes
        res, P = eval_once(ctx.model, ctx.predictor, ctx.graph, ctx.split, ctx.edges, ctx.batch_sizes,
                           ck / cfg.spec.checkpoint_name(ctx.split.dataset, run_idx, 'model'),
                           ck / cfg.spec.checkpoint_name(ctx.split.dataset, run_idx, 'predictor'))
        got = tuple(float(x) for x in res[metric][1:3])
        row = {'seed': s, 'got': got, 'ref': ref[run_idx], 'maxdiff': max(abs(a - b) for a, b in zip(got, ref[run_idx]))}
        if caps is not None:
            Pc = caps(s, metric)
            row['pred_maxdiff'] = max(float((a.float() - b.float()).abs().max()) for a, b in zip(P[1:5], Pc[1:5]))
        out.append(row)
        print(json.dumps(row), flush=True)
    return {'cell': cell, 'variant': cfg.variant, 'reference': f'log ({stop}-selected epoch)' if use_log else 'run_json base_per_seed',
            'maxdiff': max(r['maxdiff'] for r in out),
            'pred_maxdiff': max(r['pred_maxdiff'] for r in out) if caps is not None else None,
            'runtime_s': round(time.time() - t0, 1), 'seeds': out}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--log')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out')
    a = ap.parse_args()
    r = check(a.cell, a.device, a.log)
    print(json.dumps({k: v for k, v in r.items() if k != 'seeds'}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
