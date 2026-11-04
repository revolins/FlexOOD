"""Environment labels: flex_www.baselines.envs vs the original trainer's environment block (its heuristic scorers and
det_environ, imported from $FLEX_REF), including the CPU RNG state afterwards (the scorers' DataLoaders draw from it)
and the environment-count carry-over across two consecutive seeds.

    FLEX_REF=<reference tree> python tests/envs_check.py --cell cora_CN_0_1_2 [--cell ...]
CPU only. Imports the reference modules read-only (no bytecode is written).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402


def reference_block(edge_index, pos_train_edge, num_nodes, dataset, num_envs):
    """The original gnn_baseline.py environment block, verbatim but for the variable plumbing."""
    import scipy.sparse as ssp
    import torch
    from gnn_baseline_utils import det_environ
    from utils.flex_utils import CN, PA, SP
    edge_index_cpu = edge_index.cpu()
    edge_weight = torch.ones(edge_index_cpu.size(1), dtype=int)
    A = ssp.csr_matrix((edge_weight, (edge_index_cpu[0], edge_index_cpu[1])), shape=(num_nodes, num_nodes))
    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    split_train_edge = pos_train_edge.transpose(1, 0)
    if 'SP' in dataset:
        pos_train_pred, _ = SP(A, split_train_edge, True)
    if 'CN' in dataset:
        pos_train_pred, _ = CN(A, split_train_edge)
    if 'PA' in dataset:
        pos_train_pred, _ = PA(A, split_train_edge)
    if not any(tag in dataset for tag in ('SP', 'CN', 'PA')):
        pos_train_pred, _ = CN(A, split_train_edge)
    if torch.unique(pos_train_pred).numel() > num_envs:
        env_idx = det_environ(pos_train_pred, torch.unique(pos_train_pred), num_envs)
    else:
        env_idx = pos_train_pred.long()
    unique_values, inverse_indices = torch.unique(env_idx, return_inverse=True)
    env_idx = torch.tensor([0, 1, 2, 3, 4][:len(unique_values)])[inverse_indices]
    return env_idx, int(torch.max(env_idx)) + 1


def check(cell, num_envs=3):
    import io
    import contextlib
    import torch
    from flex_www.baselines.envs import env_labels, train_pair_scores
    from flex_www.data import build_graph, load
    from flex_www.utils import init_seed
    R.data_root()
    out = {'cell': cell, 'seeds': []}
    ne_new, ne_ref = num_envs, num_envs
    for seed in (1, 2):                       # two seeds: the environment count carries over
        init_seed(seed)
        split = load(cell, seed=seed)
        g = build_graph(split, torch.device('cpu'))
        ei, tp = g.edge_index, split.train_pos
        state0 = torch.get_rng_state()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env_new, _, ne_new_next = env_labels(train_pair_scores(ei, tp, split.num_nodes, split.dataset), ne_new)
        state_new = torch.get_rng_state()
        torch.set_rng_state(state0)
        cwd = os.getcwd()
        os.chdir(R.REF)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env_ref, ne_ref_next = reference_block(ei, tp, split.num_nodes, split.dataset, ne_ref)
        finally:
            os.chdir(cwd)
        state_ref = torch.get_rng_state()
        out['seeds'].append({'seed': seed, 'num_envs_in': [ne_new, ne_ref], 'num_envs_out': [ne_new_next, ne_ref_next],
                             'labels_equal': bool(torch.equal(env_new, env_ref)), 'rng_equal': bool(torch.equal(state_new, state_ref)),
                             'counts': torch.unique(env_new, return_counts=True)[1].tolist()})
        ne_new, ne_ref = ne_new_next, ne_ref_next
    out['identical'] = all(s['labels_equal'] and s['rng_equal'] and s['num_envs_out'][0] == s['num_envs_out'][1] for s in out['seeds'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', required=True)
    a = ap.parse_args()
    sys.path.append(str(R.REF))
    for c in a.cell:
        print(json.dumps(check(c)), flush=True)
