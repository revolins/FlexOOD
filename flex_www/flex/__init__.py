"""FLEX: label-free structural channels fused with a GCN link predictor, weights picked on validation only.

    features   year-free pair features, pseudo-labels from the train graph, shift-axis environments, REL / ISS models
    direction  the LPShift cell's named shift axis and direction; validation folds along it
    channels   per-pair channel scores (ACT, REL, ISS, feature cosine, AA-DC) standardised on uniform non-edges
    replay     the AUG encoder: replay of train-graph 1-hop patches as an auxiliary loss + combined epoch selection
    fusion     z(logit p) + sum_k g_k z_k with the staged validation picks and the adoption rule
    aadc       AA-DC scoring (verbatim third-party code) and its validation-only calibration (aadc_runtime)
    run        the row driver (train mode / weights mode)
"""
