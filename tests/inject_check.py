"""INJ rows (NCN / Refined-GAE + FLEX on DBLP): run flex_www.flex.run in train or weights mode and compare with the
original run's JSON (picks, injection choice and per-candidate validation, encoder and fused per-seed headline MRR).

    FLEX_REF=<ref> python tests/inject_check.py --backbone NCN --mode weights|train [--device cuda:0] [--out r.json]
weights mode reads <ref>/ood_final_backbone/<backbone>/dblp/ckpt (the backbone finals: same command, same seeds; the
INJ runs themselves saved no weights).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def check(backbone, mode, device, out_dir=None):
    import torch
    import yaml
    from flex_www.flex.run import load_config, train_mode, weights_mode
    root = R.data_root()
    path = REPO / 'configs/backbones' / f'{backbone}_FLEX' / 'dblp.yaml'
    c = yaml.safe_load(path.read_text())
    cfg = load_config(path, data_root=str(root))
    ref = json.loads((R.REF / c['reference']['run_json']).read_text())
    t0 = time.time()
    if mode == 'weights':
        rec = weights_mode(cfg, torch.device(device), R.REF / 'ood_final_backbone' / backbone / 'dblp' / 'ckpt', ref, check_picks=False)
    else:
        rec = train_mode(cfg, torch.device(device), out_dir or (REPO / '.check_out' / f'INJ_{backbone}'))
    m = rec['metric']
    d = lambda a, b: max(abs(x - y) for u, v in zip(a, b) for x, y in zip(u, v))  # noqa: E731
    inj = {k: (v['g'] if isinstance(v['g'], list) else list(v['g']), v['val'], v.get('N')) for k, v in rec['inj']['by_frac'].items()}
    rinj = {k: (v['g'], v['val'], v.get('N')) for k, v in ref['inj']['by_frac'].items()}
    return {'backbone': backbone, 'mode': mode, 'g': rec['picks']['g'], 'ref_g': ref['picks']['g'],
            'picks_identical': json.loads(json.dumps(rec['picks'])) == ref['picks'] if mode == 'train' else None,
            'inj_choice': rec['inj']['choice'], 'ref_choice': ref['inj']['choice'], 'inj_by_frac_equal': inj == rinj,
            'inj_by_frac': inj, 'ref_inj_by_frac': rinj,
            'base_maxdiff': d(rec['base_per_seed'][m], ref['base_per_seed'][m]),
            'per_seed_maxdiff': d(rec['per_seed'][m], ref['per_seed'][m]),
            'test_mean': sum(t for _, t in rec['per_seed'][m]) / len(rec['per_seed'][m]),
            'ref_test_mean': sum(t for _, t in ref['per_seed'][m]) / len(ref['per_seed'][m]), 'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True, choices=('NCN', 'RGAE'))
    ap.add_argument('--mode', required=True, choices=('weights', 'train'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out')
    a = ap.parse_args()
    r = check(a.backbone, a.mode, a.device)
    print(json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
