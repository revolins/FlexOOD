"""Copy the LPShift splits of the review headline cells into review_data/lpshift/ (the layout flex_www.utils.data_root()
expects), write a SHA-256 manifest and scan every staged file for identifying strings.

    python scripts/stage_review_lpshift.py --src <tree with dataset/<cell>_hfix_seed1Dataset/>
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

CELLS = ('pubmed_CN_0_1_2', 'cora_SP_026_017_00', 'citeseer_PA_100_50_0')
BANNED = []            # filled in main(): the source root path + $FLEX_FORBID (comma-separated strings)


def forbidden_strings(src):
    extra = [w for w in os.environ.get('FLEX_FORBID', '').split(',') if w]
    return [str(Path(src).resolve()).encode()] + [w.encode() for w in extra]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'review_data'))
    a = p.parse_args()
    BANNED[:] = forbidden_strings(a.src)
    out = Path(a.out)
    staged = []
    for cell in CELLS:
        name = f'{cell}_hfix_seed1'
        src = Path(a.src) / 'dataset' / f'{name}Dataset'
        dst = out / 'lpshift' / f'{name}Dataset'
        dst.mkdir(parents=True, exist_ok=True)
        for f in (f'{name}Dataset.pt', f'{name}Dataset_split.pt', 'heart_valid_samples.npy', 'heart_test_samples.npy'):
            shutil.copyfile(src / f, dst / f)            # follows symlinks: the bundle holds real files
            staged.append(dst / f)
    for f in staged:
        raw = f.read_bytes()
        hits = [b.decode() for b in BANNED if b in raw]
        assert not hits, f'{f}: identifying string(s) {hits}'
    with open(out / 'lpshift' / 'MANIFEST.sha256', 'w') as fh:
        for f in staged:
            fh.write(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out / "lpshift")}\n')
    print(f'{len(staged)} files, {sum(f.stat().st_size for f in staged) / 1e6:.1f} MB, anonymity scan clean')


if __name__ == '__main__':
    main()
