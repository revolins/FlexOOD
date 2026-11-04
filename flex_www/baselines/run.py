"""Baseline driver.

    python -m flex_www.baselines.run --config configs/baselines/<method>/<cell>.yaml --mode train   [--out DIR]
    python -m flex_www.baselines.run --config configs/baselines/<method>/<cell>.yaml --mode weights --ckpt DIR

train    train every seed of the config (seeds 1..runs) and report, per seed, the cell metric's (valid, test) at the
         first evaluation with the best validation value, skipping evaluations whose scores are tied (the paper's
         collapse watcher, `collapsed`), plus mean / std (population) over seeds; --out also writes the checkpoints
         (the best-validation epoch of the trainer's stop metric).
weights  load each seed's saved encoder + predictor (strict), evaluate once, report the cell metric at that epoch.
         Checkpoints are named as the paper's runs wrote them: base_<dataset>_drp<dropout>_lr<lr>_run<r>_<tag>{model,
         predictor}.pt with run r = seed r + 1.
AA-DC    configs with method AA-DC run the training-free heuristic (validation-only calibration; Amazon: one split per
         seed) and report valid / test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import hooks as method_hooks
from . import module as method_module
from ..evaluate import evaluate
from ..train import Hooks, RunSpec, prepare, run


def load_config(path, data_root=None, warm_start_dir=None):
    """(config dict, RunSpec or None for AA-DC)."""
    c = yaml.safe_load(Path(path).read_text())
    if c['method'] == 'AA-DC':
        return c, None
    spec = RunSpec.from_config({**c['train'], 'cell': c['cell'], 'data_root': data_root, 'warm_start_dir': warm_start_dir})
    return c, spec


def checkpoint_paths(ckpt_dir, spec, dataset, run_idx, method):
    """(model, predictor) files of one run: the backbone-tagged names the trainer wrote ('unGCN' / 'NCN' / 'RGAE', 'GCN'
    for gMPNN_ref) or, failing those, the method-tagged copies (..._lc<coeff>_ood<method>_run<r>_{model,predictor}.pt)."""
    d = Path(ckpt_dir)
    m, p = d / spec.checkpoint_name(dataset, run_idx, 'model'), d / spec.checkpoint_name(dataset, run_idx, 'predictor')
    if m.exists() and p.exists():
        return m, p
    stem = f"base_{dataset}_drp{spec.hp['dropout']}_lr{spec.hp['lr']}"
    for tag in ('GCN',):
        m, p = d / f'{stem}_run{run_idx}_{tag}model.pt', d / f'{stem}_run{run_idx}_{tag}predictor.pt'
        if m.exists() and p.exists():
            return m, p
    m = sorted(d.glob(f'{stem}_lc*_ood{method}_run{run_idx}_model.pt'))
    p = sorted(d.glob(f'{stem}_lc*_ood{method}_run{run_idx}_predictor.pt'))
    if len(m) == 1 and len(p) == 1:
        return m[0], p[0]
    raise FileNotFoundError(f'no checkpoint for run {run_idx} in {d}')


def load_strict(module, path, device):
    """Strict load; returns the key mismatch (the paper's trainer loaded non-strictly, so a mismatch would be silent)."""
    state = torch.load(path, map_location=device)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'{path}: missing {missing}, unexpected {unexpected}')
    return {'missing': list(missing), 'unexpected': list(unexpected)}


def evaluate_ctx(ctx, h):
    if h.evaluate:
        return h.evaluate(ctx)
    return evaluate(ctx.model, ctx.predictor, ctx.graph, ctx.split, ctx.edges, ctx.batch_sizes, ctx.spec.use_valedges_as_input)


def weights_mode(cfg, spec, ckpt_dir, device, seeds=None):
    method, opts = cfg['method'], cfg.get('method_opts', {})
    h, mod = method_hooks(method, opts), method_module(method)
    out, bs = [], None
    for run_idx, s in enumerate(spec.seeds()):
        if seeds and s not in seeds:
            continue
        ctx = prepare(spec, s, run_idx, device, Hooks(build_models=h.build_models), bs, warm_start=False)
        bs = ctx.batch_sizes
        if mod is not None and hasattr(mod, 'before_load'):
            mod.before_load(ctx, opts)
        mp, pp = checkpoint_paths(ckpt_dir, spec, ctx.split.dataset, run_idx, method)
        load_strict(ctx.model, mp, device)
        load_strict(ctx.predictor, pp, device)
        res, _ = evaluate_ctx(ctx, h)
        row = {'seed': s, 'metric': ctx.split.metric, 'valid': float(res[ctx.split.metric][1]),
               'test': float(res[ctx.split.metric][2]), 'stop_metric': ctx.split.stop_metric,
               'stop_valid': float(res[ctx.split.stop_metric][1]), 'checkpoint': mp.name}
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def collapsed(auc, prev):
    """Collapse watcher of the paper's results export: an evaluation whose scores are tied is not selectable.
    auc / prev = (train, valid, test) AUC in % (2 decimals, as logged) of this / the previous epoch. Exact tie: valid and
    test AUC both 50.00; tie-break spike: train AUC within 1 of 50 and the previous epoch tied or valid AUC >= 99.9 or
    <= 0.1 (a constant scorer can put every positive just above or below its tied negatives)."""
    tie = lambda a: a is not None and a[1] == 50.0 and a[2] == 50.0  # noqa: E731
    if auc is None:
        return False
    tr, va, _ = auc
    return tie(auc) or (abs(tr - 50.0) <= 1.0 and (tie(prev) or va >= 99.9 or va <= 0.1))


def select(history, metric, eval_steps):
    """(valid, test, epoch, n_flagged) of `metric` at the first best-validation evaluation that the collapse watcher
    does not flag (the reported selection; without flagged epochs it equals the trainer's own)."""
    rows = history[metric]
    pct = lambda t: tuple(round(100 * float(x), 2) for x in t)  # noqa: E731
    auc = {(i + 1) * eval_steps: pct(a) for i, a in enumerate(history.get('AUC', []))}
    ok = [i for i in range(len(rows)) if not collapsed(auc.get((i + 1) * eval_steps), auc.get((i + 1) * eval_steps - 1))]
    i = max(ok, key=lambda j: (rows[j][1], -j))
    return float(rows[i][1]), float(rows[i][2]), (i + 1) * eval_steps, len(rows) - len(ok)


def train_mode(cfg, spec, device, out_dir=None, log=print):
    if out_dir:
        spec.checkpoint_dir = str(Path(out_dir) / 'ckpt')
    h = method_hooks(cfg['method'], cfg.get('method_opts', {}))
    res = run(spec, device, h, log=log)
    metric, rows = cfg['metric'], []
    for s, r in res.seeds.items():
        v, t, ep, flagged = select(r.history, metric, spec.eval_steps)
        rows.append({'seed': s, 'metric': metric, 'valid': v, 'test': t, 'epoch': ep, 'watcher_flagged': flagged,
                     'trainer_selection': list(r.best[metric][:2]), 'best_epoch_stop_metric': r.best_epoch})
    return rows


def aadc_mode(cfg, data_root=None):
    from ..flex.aadc_runtime import run_cell
    seeds = cfg.get('method_opts', {}).get('seeds')
    rows = [run_cell(cfg['cell'], seed=s, root=data_root) for s in seeds] if seeds else [run_cell(cfg['cell'], root=data_root)]
    return [{'seed': s, 'metric': r['metric'], 'valid': r['valid'] / 100, 'test': r['test'] / 100,
             'anchor_scale': r['anchor_scale'], 'gate_thresholds': r['gate_thresholds']} for s, r in zip(seeds or [None], rows)]


def summarise(rows):
    t = np.array([r['test'] for r in rows])
    v = np.array([r['valid'] for r in rows])
    return {'valid_mean': 100 * float(v.mean()), 'valid_std': 100 * float(v.std()),
            'test_mean': 100 * float(t.mean()), 'test_std': 100 * float(t.std()), 'seeds': len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', default='train', choices=['train', 'weights'])
    ap.add_argument('--ckpt', help='weights mode: directory of the saved checkpoints')
    ap.add_argument('--out', help='output directory (results JSON; train mode also writes checkpoints)')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', help='data bundle root (default $FLEX_DATA_ROOT or <repo>/data)')
    ap.add_argument('--warm_start_dir', help='Amazon cross-domain: directory of the source-domain ERM checkpoints')
    ap.add_argument('--runs', type=int, help='override the number of seeds (1 = only --seed)')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--epochs', type=int, help='override the epoch budget (smoke tests)')
    a = ap.parse_args()
    cfg, spec = load_config(a.config, a.data_root, a.warm_start_dir)
    t0 = time.time()
    if spec is None:
        rows = aadc_mode(cfg, a.data_root)
    else:
        for k in ('runs', 'seed', 'epochs'):
            if getattr(a, k) is not None:
                setattr(spec, k, getattr(a, k))
        dev = torch.device(a.device)
        if a.mode == 'weights':
            assert a.ckpt, '--mode weights needs --ckpt'
            rows = weights_mode(cfg, spec, a.ckpt, dev)
        else:
            assert not spec.x_d or a.warm_start_dir, 'cross-domain cells warm-start from the source-domain ERM checkpoints: --warm_start_dir'
            rows = train_mode(cfg, spec, dev, a.out)
    result = {'method': cfg['method'], 'cell': cfg['cell'], 'mode': 'aadc' if spec is None else a.mode, 'per_seed': rows,
              **summarise(rows), 'paper': cfg.get('paper'), 'runtime_s': round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in result.items() if k != 'per_seed'}), flush=True)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"{cfg['method']}__{cfg['cell']}__{result['mode']}.json").write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
