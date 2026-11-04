"""Stage the frozen data inputs of the real ogbl-collab FLEX row into review_data/ogbl-collab/ (+ a SHA-256 manifest).

    python scripts/stage_collab_inputs.py --src <root of the original experiment tree> [--out review_data/ogbl-collab]
    python scripts/stage_collab_inputs.py --src <root> --baseline_weights <camera-ready dir>/ogbl-collab

The second form copies the checkpoints of the other ogbl-collab rows (10 seeds each) into <dir>/<method>/seed<s>_*.pt:
the GCN+age10 baselines (8 methods) under <dir>/baselines/, and NCN, RGAE (Refined-GAE) and gMPNN_ref under <dir>/.
These weights are NOT part of the review bundle (only the FLEX headline rows ship weights): the target must lie outside
review_weights/ (default: ../FLEX_WWW_camera_ready/ogbl-collab next to the repository).

Frozen inputs (everything else is recomputed from the OGB dataset by flex_www.collab):
  rstar_models.npz   the fitted R* models as plain arrays: B6 residual model m6, its feature teacher, the activity-aware
                     refit mA (each = standardiser mean / scale + logistic coef / intercept). Fitted once on a 2015 history
                     with a separately trained 2015 GCN; refitting them is outside this release.
  twin_scores.npz    the frozen DiGress "twin" scorer's raw scores on validation half B and on the test pairs, twin seeds 1-3
                     (validB<ts>, test<ts>). DiGress itself is not part of this release.
The source files are read only; their sklearn pickles are converted to arrays here (needs scikit-learn to unpickle).
"""
import argparse
import hashlib
import os
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
B6_PICKLE = 'output/shift_gap/collab/gen_oracle/val/b6_model.pkl'
TWIN = 'output/digress_scorer_collab_v1/scores_TWIN_seed{}.npz'
FORBIDDEN = []            # filled in main(): the source root path + $FLEX_FORBID (comma-separated strings)


def forbidden_strings(src):
    extra = [w for w in os.environ.get('FLEX_FORBID', '').split(',') if w]
    return [str(Path(src).resolve()).encode()] + [w.encode() for w in extra]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def linear_arrays(prefix, model, scaler):
    return {f'{prefix}_mean': scaler.mean_, f'{prefix}_scale': scaler.scale_, f'{prefix}_coef': model.coef_,
            f'{prefix}_intercept': model.intercept_}


def stage(src, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(src / B6_PICKLE, 'rb') as f:
        B = pickle.load(f)
    m6, s6, key, mode = B['model']
    assert (key, mode) == ('semantic_rrwp', 'ts'), (key, mode)
    t, ts = B['teacher']
    arrays = {**linear_arrays('m6', m6, s6), **linear_arrays('teacher', t, ts), **linear_arrays('mA', B['mA'], B['sA'])}
    np.savez(out / 'rstar_models.npz', **arrays)
    tw = {}
    for s in (1, 2, 3):
        d = np.load(src / TWIN.format(s))
        tw[f'validB{s}'], tw[f'test{s}'] = d['validB'], d['test']
    np.savez(out / 'twin_scores.npz', **tw)


BASELINES = ('ERM', 'IRM', 'VRex', 'GroupDRO', 'CORAL', 'DANN', 'SRGNN', 'VGAE')
HARNESS_ROOT = 'ood_final_rstar'


def baseline_checkpoints(src, method):
    """{seed: (model, predictor)} of the recorded 10-seed final run of a method (seed s = run s - 1)."""
    recs = [json.loads(line) for line in open(src / HARNESS_ROOT / '_manifest.jsonl')]
    recs = [r for r in recs if r.get('method') == method and r.get('cell') == 'ogbl-collab_rstar' and r.get('status') == 'ok'
            and r.get('cmd')]
    cmd = recs[-1]['cmd']
    d = src / cmd[cmd.index('--checkpoint_dir') + 1]
    out = {}
    for s in range(1, 11):
        if method == 'ERM':
            m, p = f'_run{s - 1}_unGCNmodel.pt', f'_run{s - 1}_unGCNpredictor.pt'
            pick = lambda pat: [h for h in d.glob(f'*{pat}')]  # noqa: E731
        else:
            m, p = f'_ood{method}_run{s - 1}_model.pt', f'_ood{method}_run{s - 1}_predictor.pt'
            pick = lambda pat: [h for h in d.glob(f'*{pat}') if '_ood' in h.name]  # noqa: E731
        hm, hp = pick(m), pick(p)
        assert len(hm) == len(hp) == 1, (method, s, hm, hp)
        out[s] = (hm[0], hp[0])
    return out


BACKBONE_CKPT = {'NCN': 'ood_final_backbone/NCN/ogbl-collab/seed{s}_ckpt', 'RGAE': 'ood_final_backbone/RGAE/ogbl-collab/seed{s}_ckpt'}
GMPNN_CKPT = 'ood_final_backbone/gMPNN_ref/ogbl-collab/ckpt'
CAMERA_READY = REPO.parent / 'FLEX_WWW_camera_ready/ogbl-collab'


def backbone_checkpoints(src, method):
    """{seed: (model, predictor)}: NCN / RGAE (one final run per seed) and gMPNN_ref (one 10-run final, run r = seed r+1)."""
    out = {}
    for s in range(1, 11):
        if method == 'gMPNN_ref':
            d, tag = src / GMPNN_CKPT, f'_oodgMPNN_ref_run{s - 1}_'
            hm, hp = sorted(d.glob(f'*{tag}model.pt')), sorted(d.glob(f'*{tag}predictor.pt'))
        else:
            d = src / BACKBONE_CKPT[method].format(s=s)
            hm, hp = sorted(d.glob('*model.pt')), sorted(d.glob('*predictor.pt'))
        assert len(hm) == len(hp) == 1, (method, s, hm, hp)
        out[s] = (hm[0], hp[0])
    return out


def stage_baselines(src, dst):
    import shutil
    assert 'review_weights' not in dst.resolve().parts, 'baseline / backbone weights never go under review_weights/'
    jobs = [(dst / 'baselines' / m, baseline_checkpoints(src, m)) for m in BASELINES]
    jobs += [(dst / m, backbone_checkpoints(src, m)) for m in ('NCN', 'RGAE', 'gMPNN_ref')]
    for d, ck in jobs:
        d.mkdir(parents=True, exist_ok=True)
        for s, (m, p) in ck.items():
            for f, name in ((m, 'model'), (p, 'predictor')):
                if not (d / f'seed{s}_{name}.pt').exists():
                    shutil.copyfile(f, d / f'seed{s}_{name}.pt')


def scan(out):
    """Anonymity scan: raw bytes (covers text) and `strings` output (binary) of every staged file."""
    bad = []
    for p in sorted(out.rglob('*')):
        if not p.is_file():
            continue
        raw = p.read_bytes()
        found = [w.decode() for w in FORBIDDEN if w in raw]
        try:
            st = subprocess.run(['strings', '-a', str(p)], capture_output=True, check=True).stdout
            found += [w.decode() for w in FORBIDDEN if w in st and w.decode() not in found]
        except (OSError, subprocess.CalledProcessError):
            pass
        if found:
            bad.append((str(p.relative_to(out)), found))
    return bad


def manifest(out):
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    lines = [f'{sha256(p)}  {p.relative_to(out)}' for p in files]
    (out / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
    return sum(p.stat().st_size for p in files), len(files)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default=str(REPO / 'review_data/ogbl-collab'))
    ap.add_argument('--baseline_weights', nargs='?', const=str(CAMERA_READY), default=None,
                    help='stage the non-review checkpoints here (default when given without a value: %(const)s)')
    a = ap.parse_args()
    FORBIDDEN[:] = forbidden_strings(a.src)
    out = Path(a.out)
    if a.baseline_weights:
        out = Path(a.baseline_weights)
        stage_baselines(Path(a.src), out)
    else:
        stage(Path(a.src), out)
    bad = scan(out)
    size, nfiles = manifest(out)
    print(json.dumps({'files': nfiles, 'bytes': size, 'anonymity_hits': bad}, indent=1))
    assert not bad, f'staged files contain identifying strings: {bad}'
