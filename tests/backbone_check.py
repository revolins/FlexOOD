"""Eval-once for the NCN / Refined-GAE backbone rows: load each run's saved encoder + predictor and compare the headline
metric with the original trainer's log at the saved epoch (first strict maximum of the stop metric's validation).

    FLEX_REF=<ref> python tests/backbone_check.py --backbone NCN --cell dblp [--runs 10] [--device cuda:0]
Reads <ref>/ood_final_backbone/<backbone>/<cell>/{model_flags.txt, final.txt, ckpt/}.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402
from evalonce_check import from_log  # noqa: E402


def namespace(final_txt):
    """argparse Namespace printed at the top of the trainer's log -> dict."""
    line = next(ln for ln in Path(final_txt).read_text().splitlines() if ln.startswith('Namespace('))
    body = line[len('Namespace('):-1]
    out = {}
    for m in re.finditer(r"(\w+)=('(?:[^']*)'|[^,]+)(?:, |$)", body):
        try:
            out[m.group(1)] = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            out[m.group(1)] = m.group(2)
    return out


def check(backbone, cell, device, runs=10):
    import torch
    from flex_www.evaluate import eval_once
    from flex_www.train import RunSpec, prepare
    root = R.data_root()
    d = R.REF / 'ood_final_backbone' / backbone / cell
    a = namespace(d / 'final.txt')
    hp = {k: a[k] for k in ('hidden_channels', 'num_layers', 'num_layers_predictor', 'dropout', 'edge_drop', 'lr', 'l2')}
    hp.update({k: v for k, v in a.items() if k.startswith(('ncn_', 'rgae_'))})
    spec = RunSpec(cell=cell, hp=hp, backbone=backbone, epochs=a['epochs'], eval_steps=a['eval_steps'], runs=runs,
                   batch_size=a['batch_size'], val_batch_size=a['val_batch_size'], test_batch_size=a['test_batch_size'],
                   x_d=bool(a.get('x_d')), data_root=str(root))
    out, bs = [], None
    for run_idx, s in enumerate(spec.seeds()):
        ctx = prepare(spec, s, run_idx, torch.device(device), None, bs, warm_start=False)
        bs = ctx.batch_sizes
        metric, stop = ctx.split.metric, ctx.split.stop_metric
        ref = from_log(d / 'final.txt', metric, stop)
        res, _ = eval_once(ctx.model, ctx.predictor, ctx.graph, ctx.split, ctx.edges, ctx.batch_sizes,
                           d / 'ckpt' / spec.checkpoint_name(ctx.split.dataset, run_idx, 'model'),
                           d / 'ckpt' / spec.checkpoint_name(ctx.split.dataset, run_idx, 'predictor'))
        got = tuple(float(x) for x in res[metric][1:3])
        out.append({'seed': s, 'got': got, 'ref': ref[run_idx], 'maxdiff': max(abs(x - y) for x, y in zip(got, ref[run_idx]))})
        print(json.dumps(out[-1]), flush=True)
    return {'backbone': backbone, 'cell': cell, 'metric': metric, 'maxdiff': max(r['maxdiff'] for r in out), 'seeds': len(out)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    print(json.dumps(check(a.backbone, a.cell, a.device, a.runs)), flush=True)
