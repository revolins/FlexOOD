# Data builders

These scripts rebuild the released data from the public downloads. Run them from the repository root with the
`flex` environment and `OMP_NUM_THREADS=9`. They write the layout that `flex_www.utils.data_root()` reads, so
`--out data` (or `--root data`) fills the default data root. Each builder reproduces the released files exactly.
The `.npy` and `.npz` files are byte-identical. The torch pickles are equal tensor by tensor on every field.

## LPShift splits: `lpshift_splits.py`

Builds the graph and the positives of LPShift cells, such as `cora_CN_0_1_2` or `ogbl-collab_SP_026_017_00`
(`--cells all` builds the 24 cells of the paper). Every edge is scored with one heuristic: common neighbours (CN),
preferential attachment (PA), or inverse shortest-path length with the edge itself removed (SP). The score then
puts the edge in train, valid or test. The cell name gives the two thresholds and the direction of the shift (see
the module docstring).

- **Inputs:** a PyG / OGB root (`--raw`) holding the Planetoid raw files `cora/raw/`, `citeseer/raw/` and `pubmed/raw/`
  (`torch_geometric.datasets.Planetoid`) and the OGB download `ogbl_collab/` (`ogb.linkproppred`). Both libraries
  download and process these on first use.
- **Outputs:** `<out>/lpshift/<cell>_hfix_seed1Dataset/<cell>_hfix_seed1Dataset.pt` holds the training graph (`x`,
  `edge_index`, `adj_t`). The `_split.pt` file next to it holds the train / valid / test positives with their random
  negatives.

```bash
python -m data_prep.lpshift_splits --raw <pyg_root> --out data --cells all
```

The released cells come from three revisions of the original generator. The revisions use the same protocol, but
they stored slightly different extras: the random negatives or a placeholder, and an unused `z` node mask. For
ogbl-collab PA, the scored edge list also differs. `FORMATS` in the script records the revision of each cell
family, so the output matches field for field.

Runtime (one CPU process, 9 threads): about 10 s per Planetoid cell, and 1-2 min and up to 2.5 GB of memory per
ogbl-collab cell. Add about 1 min the first time PyG / OGB process the raw files.

## Corrected HeaRT negatives: `lpshift_heart.py`

Writes `heart_valid_samples.npy` and `heart_test_samples.npy`, of shape [P, 250, 2]. Each validation and test
positive gets its own 250 hard negatives (HeaRT protocol), ranked by common neighbours. The correction masks both
halves of the (*, v) corruption: existing edges, the endpoints, and nodes unseen in training. The generator this
replaces masked only the source half. Run this after `lpshift_splits.py`, on the same cells. It re-seeds and replays
the generator's random draws, so the zero-score fallback continues the same NumPy stream.

- **Inputs:** `<root>/lpshift/<cell>_hfix_seed1Dataset/` (graph and split).
- **Outputs:** the two `.npy` files in the same folder.
- **Runtime:** 10-45 s per Planetoid cell. An ogbl-collab cell takes 0.5-6 min, or up to 15 min for the PA cells
  (their valid / test sets are 10x larger). On ogbl-collab the common-neighbour rows are computed on demand.

```bash
python -m data_prep.lpshift_heart --root data --cells all
```

## Leak-free few-shot DBLP: `dblp_fewshot.py`

Builds the author-shift DBLP split. Train authors keep every author-paper edge. Validation and test authors reveal
one paper each, and their remaining papers are the queries. Author features are rebuilt from visible papers only,
and every query has 50 typed author → paper negatives.

- **Inputs:** the raw PyG DBLP folder (`--raw`, `torch_geometric.datasets.DBLP`'s `raw/`), which holds `adjM.npz`,
  `node_types.npy`, `features_0.npz`, `features_1.npz`, `features_2.npy` and `train_val_test_idx.npz`.
- **Outputs:** `<out>/dblp/harness_k1_neg50.npz` (`x`, `edge_index`, `num_nodes`, `train_pos`, `val_pos`, `test_pos`,
  `val_neg`, `test_neg`) and `<out>/dblp/node_types.npy`.

```bash
python -m data_prep.dblp_fewshot --raw <pyg_root>/DBLP/raw --out data
```

Runtime: about 20 s.
