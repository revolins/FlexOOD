"""Channels: fit the FLEX channels with flex_www on a cell's train graph and compare every channel array with the ones
the original run dumped (caps C<i>_<j>_<name>). CPU only.

    FLEX_REF=<reference tree> python tests/channels_check.py --cell cora_CN_0_1_2 [--out r.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def check(cell):
    import torch
    import yaml
    from flex_www.data import build_graph, load
    from flex_www.flex.run import _aadc_channel, fit_channels, load_config, load_captures
    root = R.data_root()
    c = yaml.safe_load((REPO / 'configs/flex' / f'{cell}.yaml').read_text())
    cfg = load_config(REPO / 'configs/flex' / f'{cell}.yaml', data_root=str(root))
    meta, seeds, sel, chans_ref, masks, mf = load_captures(R.REF / c['reference']['caps'])
    t0 = time.time()
    split = load(cell, root=str(root))
    g = build_graph(split, torch.device('cpu'), cfg.spec.val_edges_test_input)
    e = [split.train_pos, split.valid_pos, split.valid_neg, split.test_pos, split.test_neg]
    aadc = _aadc_channel(cfg, lambda *a: None) if cfg.flags[0] else None
    _, _, _, chans, _ = fit_channels(cfg, seeds, {s: g for s in seeds}, {s: e for s in seeds}, lambda *a: None, aadc)
    diff = {}
    for j in range(4):
        for k, v in chans_ref[seeds[0]][j].items():
            if k == 'iss':
                for s in seeds:
                    diff[f'iss_{s}'] = max(diff.get(f'iss_{s}', 0.0), float(np.abs(chans[s][j]['iss'][s] - v[s]).max()))
            else:
                diff[k] = max(diff.get(k, 0.0), float(np.abs(chans[seeds[0]][j][k] - v).max()))
    return {'cell': cell, 'variant': cfg.variant, 'channels': sorted(diff), 'maxdiff': max(diff.values()),
            'per_channel': diff, 'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    out = []
    for cell in a.cell:
        r = check(cell)
        print(json.dumps(r), flush=True)
        out.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
