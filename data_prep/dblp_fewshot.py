"""Build the leak-free few-shot DBLP split (author -> paper link prediction under an author shift).

Input: the raw PyG DBLP files (adjM.npz, node_types.npy, features_{0,1}.npz, features_2.npy, train_val_test_idx.npz).
Output, in the layout flex_www.data.dblp reads:
    <out>/dblp/harness_k1_neg50.npz   x, edge_index, num_nodes, train/val/test positives, val/test negatives
    <out>/dblp/node_types.npy         copied from the raw folder (0 = author, 1 = paper, 2 = term, 3 = venue)

Protocol:
  * the official author split (train / val / test author lists) is kept;
  * train authors: every author-paper (AP) edge is visible and is a training positive;
  * val / test authors: K = 1 paper per author is revealed into the visible graph (seeded, authors in sorted order,
    val then test); the remaining papers are the query positives. Authors with a single paper give no query;
  * author features are rebuilt from VISIBLE papers only: x_a = L2-normalised sum of the paper title BoW (features_1)
    over a's visible papers. The raw author BoW (features_0) is built from all of an author's papers, held-out ones
    included, so it is never used as model input;
  * every query positive gets 50 typed author -> paper negatives (never an AP edge, revealed ones included);
  * paper-term and paper-venue edges stay visible.

    python -m data_prep.dblp_fewshot --raw <PyG DBLP>/raw --out data
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_undirected

REVEAL_SEED = 20261003
NEG_SEED = 20261003
K_REVEAL = 1
NEGATIVES = 50


def sha256_array(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, x.dtype.str)).encode() + x.tobytes()).hexdigest()


def canon(edges):
    """Unique undirected pairs as sorted rows [min, max]; authors (ids 0..A-1) come first, so rows are (author, paper)."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.unique(np.sort(edges, axis=1), axis=0)


def sparse_npz(path):
    q = np.load(path)
    return sp.csr_matrix((q['data'], q['indices'], q['indptr']), shape=tuple(q['shape']))


def load_raw(raw):
    """Read the raw DBLP files into edge lists, the padded node-feature matrix and the official author split."""
    raw = Path(raw)
    z = np.load(raw / 'adjM.npz')
    adj = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape'])).tocsr()
    node_types = np.load(raw / 'node_types.npy')
    row, col = adj.nonzero()
    all_edges = np.column_stack([row, col]).astype(np.int64)
    ap_mask = ((node_types[row] == 0) & (node_types[col] == 1)) | ((node_types[row] == 1) & (node_types[col] == 0))

    x0 = sparse_npz(raw / 'features_0.npz').toarray().astype(np.float32)
    x1 = sparse_npz(raw / 'features_1.npz').toarray().astype(np.float32)
    x2 = np.load(raw / 'features_2.npy').astype(np.float32)
    width = max(x0.shape[1], x1.shape[1], x2.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, width - x.shape[1]))).astype(np.float32)  # noqa: E731
    features = np.vstack([pad(x0), pad(x1), pad(x2)])
    n_missing = len(node_types) - len(features)   # venues have no feature file: zero rows
    if n_missing > 0:
        features = np.vstack([features, np.zeros((n_missing, width), dtype=np.float32)])

    split = np.load(raw / 'train_val_test_idx.npz')
    return {
        'n_nodes': int(len(node_types)),
        'node_types': node_types,
        'all_edges': all_edges,
        'ap_edges': canon(all_edges[ap_mask]),
        'features': features,
        'author_bow': x0,
        'paper_bow': sparse_npz(raw / 'features_1.npz').astype(np.float64),
        'papers': np.flatnonzero(node_types == 1).astype(np.int64),
        'official': {k: np.asarray(split[k], dtype=np.int64) for k in ('train_idx', 'val_idx', 'test_idx')},
    }


def make_typed_negatives(positive, papers, all_ap, n_neg, seed):
    """n_neg (author, paper) pairs per positive, keeping the positive's author; rejection-sampled against every AP edge."""
    forbidden = {tuple(p) for p in all_ap.tolist()}
    rng = np.random.default_rng(seed)
    out = np.empty((len(positive), n_neg, 2), dtype=np.int64)
    for i, (u, _) in enumerate(positive):
        made = 0
        while made < n_neg:
            v = int(papers[rng.integers(len(papers))])
            if (int(u), v) not in forbidden:
                out[i, made] = (int(u), v)
                made += 1
    return out


def author_features(n_authors, visible_ap, first_paper, paper_bow):
    """x_a = L2-normalised sum of title BoW over a's visible papers (zero row if none is visible)."""
    e = np.asarray(visible_ap, np.int64)
    B = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1] - first_paper)), shape=(n_authors, paper_bow.shape[0]))
    X = np.asarray((B @ paper_bow).todense())
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(nrm, 1e-12)).astype(np.float32)


def build_split(info, negatives=NEGATIVES):
    """Reveal one paper per val / test author, rebuild author features from the visible history, draw typed negatives."""
    ap, off = info['ap_edges'], info['official']
    n_authors = int((info['node_types'] == 0).sum())
    first_paper = int(info['papers'][0])
    assert np.array_equal(np.flatnonzero(info['node_types'] == 0), np.arange(n_authors)) and first_paper == n_authors, \
        'authors must occupy ids 0..A-1 and papers start right after'

    by_author = {}
    for u, p in ap.tolist():
        by_author.setdefault(u, []).append(p)
    rng = np.random.default_rng(REVEAL_SEED)
    revealed, pos = {'val': [], 'test': []}, {'val': [], 'test': []}
    for name, idx in (('val', off['val_idx']), ('test', off['test_idx'])):
        for a in np.sort(idx).tolist():
            ps = sorted(by_author.get(a, []))
            if not ps:
                continue
            r = rng.choice(len(ps), size=min(K_REVEAL, len(ps)), replace=False)
            rs = set(int(ps[i]) for i in r)
            revealed[name].extend((a, p) for p in sorted(rs))
            if len(ps) < K_REVEAL + 1:
                continue                      # single-paper author: revealed, but no query left
            pos[name].extend((a, p) for p in ps if p not in rs)
    rev = {k: canon(np.asarray(v, np.int64)) for k, v in revealed.items()}
    split = {
        'train_pos': ap[np.isin(ap[:, 0], off['train_idx'])],
        'val_pos': canon(np.asarray(pos['val'], np.int64)),
        'test_pos': canon(np.asarray(pos['test'], np.int64)),
        'revealed': np.vstack([rev['val'], rev['test']]),
    }
    split['history'] = canon(np.vstack([split['train_pos'], rev['val'], rev['test']]))

    features = info['features'].copy()
    xa = author_features(n_authors, split['history'], first_paper, info['paper_bow'])
    features[:n_authors] = 0.0
    features[:n_authors, :xa.shape[1]] = xa
    assert not np.allclose(features[:n_authors, :info['author_bow'].shape[1]], info['author_bow']), \
        'author features still equal the leaky raw author BoW'
    split['features'] = features

    split['val_neg'] = make_typed_negatives(split['val_pos'], info['papers'], ap, negatives, NEG_SEED + 10)
    split['test_neg'] = make_typed_negatives(split['test_pos'], info['papers'], ap, negatives, NEG_SEED + 20)
    return split


def check_split(info, split):
    """Leak checks: no held-out AP edge is visible, val/test author features see only the revealed paper,
    negatives are typed and never an AP edge."""
    key = lambda e: set(map(tuple, np.asarray(e).reshape(-1, 2).tolist()))  # noqa: E731
    hist, ap_set = key(split['history']), key(info['ap_edges'])
    heldout, rev = key(split['val_pos']) | key(split['test_pos']), key(split['revealed'])
    vt_authors = set(info['official']['val_idx'].tolist()) | set(info['official']['test_idx'].tolist())
    assert not (hist & heldout), 'held-out positive in the visible history'
    assert {e for e in hist if e[0] in vt_authors} == rev
    assert (rev | heldout) == {e for e in ap_set if e[0] in vt_authors} and not (rev & heldout)
    n_authors = int((info['node_types'] == 0).sum())
    first_paper, bow = int(info['papers'][0]), info['paper_bow']
    xa = split['features'][:n_authors, :bow.shape[1]].astype(np.float64)
    rev_by = dict(rev)
    for a in sorted(vt_authors):
        if a in rev_by:
            v = np.asarray(bow[rev_by[a] - first_paper].todense()).ravel()
            assert np.allclose(xa[a], v / max(np.linalg.norm(v), 1e-12), atol=1e-6), f'x_a mismatch for author {a}'
        else:
            assert not xa[a].any()
    assert not split['features'][:n_authors, bow.shape[1]:].any()
    papers = set(info['papers'].tolist())
    for nm in ('val_neg', 'test_neg'):
        neg = split[nm].reshape(-1, 2)
        assert not (key(neg) & ap_set), f'{nm} contains an AP edge'
        assert all(p in papers for p in neg[:, 1].tolist())
        assert np.array_equal(split[nm][:, 0, 0], split[nm.replace('neg', 'pos')][:, 0])
    assert key(split['train_pos']) <= hist


def visible_edge_index(info, history):
    """Message-passing graph: every non-AP edge plus the visible (history) AP edges, undirected."""
    n, ae = info['n_nodes'], info['all_edges']
    key = lambda e: np.minimum(e[:, 0], e[:, 1]).astype(np.int64) * n + np.maximum(e[:, 0], e[:, 1]).astype(np.int64)  # noqa: E731
    is_ap = np.isin(key(ae), key(info['ap_edges']))
    keep = (~is_ap) | np.isin(key(ae), key(history))
    return to_undirected(torch.from_numpy(ae[keep].T).long(), num_nodes=n).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--raw', required=True, help='folder with the raw PyG DBLP files')
    ap.add_argument('--out', required=True, help='data root; writes <out>/dblp/')
    ap.add_argument('--negatives', type=int, default=NEGATIVES)
    args = ap.parse_args()

    info = load_raw(args.raw)
    split = build_split(info, args.negatives)
    check_split(info, split)
    arrays = {
        'x': split['features'].astype(np.float32),
        'edge_index': visible_edge_index(info, split['history']),
        'num_nodes': np.array([info['n_nodes']]),
        'train_pos': split['train_pos'],
        'val_pos': split['val_pos'],
        'test_pos': split['test_pos'],
        # [P, neg, 2] -> [P * neg, 2] row-major, so reshape(P, neg) recovers each query's own negatives
        'val_neg': split['val_neg'].reshape(-1, 2),
        'test_neg': split['test_neg'].reshape(-1, 2),
    }
    out = Path(args.out) / 'dblp'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'harness_k1_neg{args.negatives}.npz', **arrays)
    shutil.copyfile(Path(args.raw) / 'node_types.npy', out / 'node_types.npy')
    summary = {k: list(v.shape) for k, v in arrays.items()}
    summary['history_edges'] = int(len(split['history']))
    summary['sha256'] = {k: sha256_array(v) for k, v in arrays.items()}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
