# FLEX: Subgraph Generation for Generalizing to Out-of-Distribution Links

This repository reproduces the benchmark results of the paper, from released weights or by retraining.

## What is here

| Path | Contents |
|---|---|
| `flex_www/` | The package: data loaders (`data/`), GCN / NCN / Refined-GAE models (`models/`), the training loop (`train.py`), evaluation (`evaluate.py`), FLEX (`flex/`), real ogbl-collab FLEX (`collab/`), and the 16 baselines (`baselines/`). |
| `configs/` | One YAML per table entry, generated from the recorded command of the paper's run. Never hand-edited. |
| `data_prep/` | Builders for the LPShift splits, the corrected HeaRT negatives and the leak-free DBLP split. |
| `tests/` | Equivalence tests: the ported code against the outputs of the paper's runs. |
| `review_weights/`, `review_data/` | Weights and data for the four headline results (see below). |

## Setup

```bash
conda env create -f environment.yml      # Python 3.10, torch 2.7.1 (CUDA 12.6), PyG 2.6.1, ogb 1.3.6
conda activate flex
export OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 OPENBLAS_NUM_THREADS=9
```

Data is read from `$FLEX_DATA_ROOT` (default `./data`), with this layout:

```
lpshift/<cell>_hfix_seed1Dataset/    LPShift graph, split and HeaRT negatives
ogb/                                 OGB root (ogbl-collab is downloaded on first use)
Amazon/                              PyG Amazon root (downloaded on first use)
dblp/harness_k1_neg50.npz            leak-free few-shot DBLP split (data_prep/)
dblp/node_types.npy
```

The thread settings matter. Some channels fit logistic models with lbfgs, and the result depends on the thread count in the last bits. Every number below was produced with 9 threads.

## Reproducing Results from Weights

The review bundle holds weights for four individual FLEX+GCN results. Each is reported as test mean ± std over 10 seeds; std is the population std (ddof = 0), as in the paper.

| Result | Paper | Command |
|---|---|---|
| ogbl-collab (real temporal shift), Hits@50 | 71.74 ± 0.38 | `python -m flex_www.collab.run --mode weights --weights review_weights/ogbl-collab --inputs review_data/ogbl-collab` |
| PubMed CN↑, Hits@20 | 98.64 ± 0.15 | `FLEX_DATA_ROOT=review_data python -m flex_www.flex.run --config configs/flex/pubmed_CN_0_1_2.yaml --mode weights --ckpt review_weights/pubmed_CN_0_1_2/RSTAR_ISS --ref review_weights/pubmed_CN_0_1_2/RSTAR_ISS/run_record.json` |
| Cora SP↓, Hits@20 | 51.23 ± 2.08 | same, with `cora_SP_026_017_00` and `RSTAR_ISS_SPX_AD` |
| CiteSeer PA↓, Hits@20 | 51.77 ± 2.16 | same, with `citeseer_PA_100_50_0` and `RSTAR_ISS_AD_B` |

Each run loads the encoder and predictor of every seed and evaluates them once. It then refits the FLEX channels, which are deterministic given the training graph, and re-runs the validation picks. It checks that the picks equal the recorded ones and reports the test metric. With `--ref`, it also compares every per-seed value with the recorded run. All four reproduce the paper exactly: every per-seed difference is 0.

Runtime is a few minutes per LPShift cell and about 5 min for ogbl-collab (including the first-time OGB download) on one 24 GB A5000. Evaluation needs a few GB of free GPU memory (PubMed and ogbl-collab need the most). The collab ISS channel must run on a GPU: CPU float32 inference differs in the last bits. Results go to `output/`.

## Retrain

Every table entry has a config.

```bash
# FLEX+GCN, any of the 27 LPShift / DBLP / Amazon cells
python -m flex_www.flex.run --config configs/flex/<cell>.yaml --mode train --out output/flex

# a baseline (16 methods: ERM, IRM, VREx, GroupDRO, DANN, CORAL, SRGNN, EERM, CIGA, GALA, GIGraph, VGAE, SIGVAE, CFLP,
# gMPNN, AA-DC)
python -m flex_www.baselines.run --config configs/baselines/<method>/<cell>.yaml --mode train

# NCN / Refined-GAE backbones and their +FLEX rows (DBLP, Amazon)
python -m flex_www.baselines.run --config configs/backbones/<NCN|RGAE>/<cell>.yaml --mode train
python -m flex_www.flex.run      --config configs/backbones/<NCN|RGAE>_FLEX/<cell>.yaml --mode train

# real ogbl-collab: retrains the 20 ISS models; the GCN checkpoints, R* models and DiGress twin scores stay fixed
python -m flex_www.collab.run --mode train --weights review_weights/ogbl-collab --inputs review_data/ogbl-collab
```

Cross-domain Amazon cells (`Photo_to_Computers`, `Computers_to_Photo`) warm-start from a source-domain GCN; pass `--warm_start_dir`.

### Expected agreement with the paper

| What | Agreement |
|---|---|
| FLEX+GCN, 27 non-collab cells | The released code reproduces paper results exactly from trained weights, the training loop matches the original epoch by epoch on the cells checked (GCN and replay-augmented training), on the same GPU type (A5000). |
| Baselines | Training is bit-identical epoch by epoch for every method except EERM. EERM's graph editor samples with CUDA `torch.multinomial`, which is not deterministic, so its retrained numbers vary between runs (a 3-seed retrain gave 62.29 ± 2.58 vs. 64.04 ± 2.80). Ten DBLP baselines were trained on a different GPU type, and their training curves differ from an A5000 run in the floating-point noise. |
| Real ogbl-collab, train mode | 71.72 ± 0.40 vs. 71.74. The ISS feature selection in the original runs used sklearn's unseeded default (`random_state=None`). |

Other GPU types can change results at float precision.

## Tests

```bash
export FLEX_REF=<directory holding the paper's run outputs>   # needed only by the equivalence tests
pytest tests/ -q                       # CPU tests (~3 h, mostly the six ogbl-collab fusion replays)
pytest tests/ -q -k "not ogbl-collab"  # CPU tests, skipping every ogbl-collab test (minutes)
pytest tests/ -q --gpu --slow          # everything
```

## What FLEX is, in code

`flex_www/flex/` implements FLEX on top of a trained backbone (GCN, NCN or Refined-GAE):

| Module | What it implements |
|---|---|
| `features.py` | Pair features of the training graph (common-neighbour, Adamic-Adar, walk counts, degrees, shortest-path classes) and the shift-axis environments. |
| `channels.py` | The fused channels: ACT (activity), REL (relation model fitted on the training graph), ISS (invariant stability selection over shift-axis environments, then an MLP), and the optional AA-DC channel. |
| `replay.py` | AUG variants: replay auxiliary loss on edited real subgraphs, and direction-aware epoch selection. |
| `fusion.py` | Validation picking of the channel weights, the guard / gated stage (`_G`) and the joint base stage (`_B`). |
| `inject.py` | INJ: proposer links injected into the input graph (used by the backbone+FLEX rows on DBLP). |
| `direction.py` | The validation direction fold for each LPShift shift (CN / PA / SP, up or down). |

Nothing is fitted or selected on test. The paper's FLEX row reports the best variant per cell selected on test, which the paper discloses. Each cell's variant is the `variant` field of its config.

## License

MIT (see `LICENSE`). `flex_www/flex/aadc.py` contains functions copied from the AA-DC reference implementation (MIT; `third_party/AA-DC_LICENSE`).
