"""Seeding and data-root helpers shared by every module."""
import os
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def init_seed(seed):
    """Seed Python, NumPy and torch (CPU + CUDA) exactly as every run in the paper did, once per seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_root():
    """Root of the downloaded data bundle: $FLEX_DATA_ROOT, else <repo>/data.

    Expected layout:
      lpshift/<cell>_hfix_seed1Dataset/   LPShift graph, split and HeaRT negatives (one directory per cell)
      Amazon/                             PyG Amazon root (Photo/, Computers/)
      ogb/                                OGB root (ogbl_collab/)
      dblp/harness_k1_neg50.npz           leak-free few-shot DBLP split
      dblp/node_types.npy                 DBLP node types (1 = paper)
    """
    return Path(os.environ.get('FLEX_DATA_ROOT', REPO / 'data'))


def ogb_link_dataset(name, root):
    """(dataset, edge split) of an OGB link dataset under torch >= 2.6. ogb 1.3.6 reloads its own cache (the processed
    PyG graph and the legacy pickled numpy split files) with torch.load's new default weights_only=True, which rejects
    them. Full unpickling is re-enabled only while OGB loads its own download, then the default is restored."""
    from ogb.linkproppred import PygLinkPropPredDataset
    key, old = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')
    os.environ[key] = '1'
    try:
        dataset = PygLinkPropPredDataset(name=name, root=str(root))
        return dataset, dataset.get_edge_split()
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old
