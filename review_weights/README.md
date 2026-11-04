# Review weights

Trained weights for the four headline FLEX+GCN results, 10 seeds each. `MANIFEST.sha256` lists a SHA-256 for every file.

| Result | Folder | Paper test score (mean ± std, 10 seeds) |
|---|---|---|
| ogbl-collab (real temporal shift), FLEX = R* + age10 + ISS10 + AA-DC + CN_REC2 | `ogbl-collab/` | Hits@50 71.74 ± 0.38 |
| PubMed, CN↑ (`pubmed_CN_0_1_2`), variant `RSTAR_ISS` | `pubmed_CN_0_1_2/RSTAR_ISS/` | Hits@20 98.64 ± 0.15 |
| Cora, SP↓ (`cora_SP_026_017_00`), variant `RSTAR_ISS_SPX_AD` | `cora_SP_026_017_00/RSTAR_ISS_SPX_AD/` | Hits@20 51.23 ± 2.08 |
| CiteSeer, PA↓ (`citeseer_PA_100_50_0`), variant `RSTAR_ISS_AD_B` | `citeseer_PA_100_50_0/RSTAR_ISS_AD_B/` | Hits@20 51.77 ± 2.16 † |

† A label-free / no-GNN reference matches or exceeds FLEX on this cell (see the dagger rule in the paper's results tables).

## Folder contents

**LPShift cells:**
- `seed<s>_model.pt` and `seed<s>_predictor.pt` are the GCN encoder and MLP predictor that FLEX fuses for seed `s`.
- The FLEX channels (REL, ISS, AA-DC) are refit deterministically from the training graph, so they have no weight files.
- `run_record.json` holds the recorded run: its argv, its validation picks (`picks`), and per-seed results (`per_seed`).

**ogbl-collab:**
- `gcn/seed<s>_*.pt` are the 10 GCN backbones.
- `iss/<id>/{model.pt,cfg.json}` are the 20 trained ISS models.
