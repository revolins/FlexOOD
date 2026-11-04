"""Re-run the FLEX picks and report from the original runs' capture dumps (caps_<stamp>.npz) and compare with their JSON.

    FLEX_REF=<reference tree> python tests/caps_check.py [--cell cora_CN_0_1_2 ...] [--out results.json]
CPU only. Exact comparison (after a JSON round trip) of `picks`, `per_seed` (every metric) and `seeds`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402


def caps_rows():
    return {k: e for k, e in R.manifest().items() if e.get('has_caps')}


def check(cell, e):
    from flex_www.flex.run import load_captures, parse_variant, select
    src = R.ref_path(e['source'])
    ref = json.loads(src.read_text())
    caps = src.parent / f"caps_{src.stem}.npz"
    t0 = time.time()
    meta, seeds, sel, chans, masks, mf = load_captures(caps)
    row, aadc, gap, base = parse_variant(e['variant'])
    assert row == meta['row'] and aadc == bool(meta['aadc_stage']), (row, meta['row'])
    rec = select(row, cell, (aadc, gap, base), seeds, sel, chans, mf, meta['metric'], meta['metrics'], masks, log=lambda *a: None)
    j = lambda v: json.loads(json.dumps(v, default=str))  # noqa: E731
    bad = [k for k in ('picks', 'per_seed', 'seeds') if j(rec.get(k)) != j(ref.get(k))]
    m = meta['metric']
    dmax = max(abs(a - b) for x, y in zip(rec['per_seed'][m], ref['per_seed'][m]) for a, b in zip(x, y))
    return {'cell': cell, 'variant': e['variant'], 'identical': not bad, 'diff_keys': bad, 'g': rec['picks']['g'],
            'b': rec['picks'].get('b'), 'ref_g': ref['picks']['g'], 'headline_maxdiff': dmax,
            'test_mean': sum(t for _, t in rec['per_seed'][m]) / len(seeds), 'runtime_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = caps_rows()
    out = []
    for c in (a.cell or list(rows)):
        r = check(c, rows[c])
        print(json.dumps(r), flush=True)
        out.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
