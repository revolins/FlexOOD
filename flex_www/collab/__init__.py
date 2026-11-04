"""FLEX on the real ogbl-collab temporal shift (train <= 2017, validation 2018, test 2019): GCN + R* + age10 + ISS10 + AD +
CN_REC2, and the GCN+age10 baseline rows.

    data           collab arrays, the test / validation-half-B pair sets, the age10 graph edit
    config         the recorded configuration (configs/collab/collab.yaml)
    gcn_scores     trained GCN checkpoints scoring pairs on the validation-time and (age10) test-time graphs
    pair_features  label-free pair descriptors (B6 inputs of R*, the 31 x-free ISS columns)
    rstar_age10    R* re-ranking (frozen B6 residual + twin + activity terms) and its validation-picked scalars
    iss            ISS10: environments over past years, invariant stability selection + MLP, the z channel
    channels       AA-DC (AD) and the 2-year common-neighbour channel (CN_REC2)
    fuse           validation picks (net-repair criterion, Hits@50 fallback) and Hits@K
    run            the CLI: weights / train / baselines

What each stage adds (test Hits@50, mean over the 10 GCN seeds of the FLEX backbone): GCN+age10 62.46 -> R*+age10 70.02
-> + ISS10 71.58 -> + AD 71.69 -> + CN_REC2 71.74 (the paper row).
"""
