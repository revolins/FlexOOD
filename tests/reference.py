"""Locations of the reference artifacts the equivalence tests compare against.

$FLEX_REF points at the root of the original experiment tree (the run JSONs, capture dumps, checkpoints and AA-DC
outputs listed in the headline manifest). `data_root()` builds (once) a directory of symlinks in the layout
flex_www.utils.data_root() expects, so the package reads exactly the files the original runs read.
"""
import json
import os
from pathlib import Path

REF = Path(os.environ['FLEX_REF']) if 'FLEX_REF' in os.environ else None
MANIFEST = 'logs_flexport/release/headline_manifest.json'


def available():
    return REF is not None and (REF / MANIFEST).exists()


def manifest():
    return json.loads((REF / MANIFEST).read_text())


def ref_path(p):
    """Manifest paths are absolute paths of the original tree; re-anchor them on $FLEX_REF."""
    p = str(p)
    i = p.find('/output/')
    return REF / p[i + 1:] if i >= 0 else Path(p)


def data_root(dst=None):
    dst = Path(dst or os.environ.get('FLEX_DATA_ROOT', Path(__file__).resolve().parents[1] / '.reference_data'))
    (dst / 'lpshift').mkdir(parents=True, exist_ok=True)
    (dst / 'dblp').mkdir(parents=True, exist_ok=True)
    links = {dst / 'Amazon': REF / 'data/Amazon', dst / 'ogb': REF / 'dataset',
             dst / 'dblp/harness_k1_neg50.npz': REF / 'output/dblp_fewshot_v1/harness_k1_neg50.npz',
             dst / 'dblp/node_types.npy': REF / 'data/DBLP/raw/node_types.npy'}
    for d in (REF / 'dataset').glob('*_hfix_seed1Dataset'):
        links[dst / 'lpshift' / d.name] = d
    for k, v in links.items():
        if not k.exists() and not k.is_symlink():
            k.symlink_to(v)
    os.environ['FLEX_DATA_ROOT'] = str(dst)
    return dst
