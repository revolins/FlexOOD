"""Weights mode end-to-end on rows whose original run saved checkpoints: eval once per seed, refit channels, rerun the
picks; compare picks, the encoder-only base_per_seed and the fused per-seed headline values with the run's JSON.

    FLEX_REF=<reference tree> python tests/weights_check.py --cell citeseer_PA_100_50_0 [--device cuda:0] [--out r.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def check(cell, device, ckpt=None, ref_json=None):
    import torch
    import yaml
    from flex_www.flex.run import load_config, weights_mode
    root = R.data_root()
    cfg_path = REPO / 'configs/flex' / f'{cell}.yaml'
    c = yaml.safe_load(cfg_path.read_text())
    cfg = load_config(cfg_path, data_root=str(root))
    ref = json.loads((R.REF / (ref_json or c['reference']['run_json'])).read_text())
    t0 = time.time()
    rec = weights_mode(cfg, torch.device(device), R.REF / (ckpt or c['reference']['checkpoint_dir']), ref, check_picks=False)
    m = rec['metric']
    return {'cell': cell, 'variant': cfg.variant, 'picks_match': rec['picks_match'], 'g': rec['picks']['g'],
            'b': rec['picks'].get('b', 1.0), 'base_maxdiff': rec['base_maxdiff'], 'per_seed_maxdiff': rec['per_seed_maxdiff'],
            'test_mean': rec['test_mean'], 'ref_test_mean': sum(t for _, t in ref['per_seed'][m]) / len(ref['per_seed'][m]),
            'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt')
    ap.add_argument('--ref_json')
    ap.add_argument('--out')
    a = ap.parse_args()
    out = []
    for cell in a.cell:
        r = check(cell, a.device, a.ckpt, a.ref_json)
        print(json.dumps(r), flush=True)
        out.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
