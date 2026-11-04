"""Reproduce the reference AA-DC outputs (<ref>/output/aadc/<cell>.json) with flex_www.flex.aadc_runtime.run_cell.

    FLEX_REF=<reference tree> python tests/aadc_check.py --cell cora_CN_2_1_0 [--cell ...] [--out results.json]
Amazon cells must run in a fresh process (their split is drawn from the untouched torch RNG), one cell per process.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402


def check(cell):
    from flex_www.flex.aadc_runtime import run_cell
    ref = json.loads((R.REF / 'output/aadc' / f'{cell}.json').read_text())
    t0 = time.time()
    res = run_cell(cell)
    d = {k: abs(float(res[k]) - float(ref[k])) for k in ('valid', 'test', 'anchor_scale')}
    g = res['gate_thresholds'], ref['gate_thresholds']
    gd = 0.0 if g[0] is None and g[1] is None else max(abs(float(g[0][k]) - float(g[1][k])) for k in g[1]) if (g[0] and g[1]) else float('inf')
    return {'cell': cell, 'test': res['test'], 'ref_test': ref['test'], 'valid': res['valid'], 'ref_valid': ref['valid'],
            'diff': d, 'gate_diff': gd, 'runtime_s': round(time.time() - t0, 1)}


def check_10split(cell):
    """Amazon: the paper's AA-DC row runs on each seed's own split (seeds 1-10); compare with <cell>_10split.json."""
    import statistics as st
    from flex_www.flex.aadc_runtime import run_cell
    ref = json.loads((R.REF / 'output/aadc' / f'{cell}_10split.json').read_text())
    t0 = time.time()
    got = [run_cell(cell, seed=s) for s in ref['seeds']]
    per = [[r['valid'], r['test']] for r in got]
    d = max(abs(a - b) for x, y in zip(per, ref['per_seed']) for a, b in zip(x, y))
    return {'cell': cell + '_10split', 'test': st.mean(r['test'] for r in got), 'ref_test': ref['test'],
            'test_std': st.pstdev([r['test'] for r in got]), 'ref_test_std': ref['test_std'], 'per_seed_maxdiff': d,
            'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    R.data_root()
    out = []
    for c in a.cell:
        r = check_10split(c[:-len('_10split')]) if c.endswith('_10split') else check(c)
        print(json.dumps(r), flush=True)
        out.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
