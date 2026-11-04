"""Real ogbl-collab: weights mode reproduces the paper's per-seed test numbers of the FLEX rows and the GCN+age10 baselines.

    FLEX_REF=<reference tree> OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 pytest tests/test_collab.py --gpu

Compared with the original locked reads under $FLEX_REF (per GCN seed, |diff| <= 1e-6; observed: 0):
  ISS10, ISS10+AD, ISS10+AD+CN_REC2   output/heur_chan/read/read.json            (paper row: ISS10+AD+CN_REC2, 71.74 +/- 0.38)
  R*+age10                            output/shift_gap/collab/bundle_age10/read.json (ERM_exact_RSTAR)
  GCN+age10 baselines                 output/shift_gap/collab/age10_methods/read.json ('e|H50' / 'e|H20')
  AA-DC, gMPNN (age10 graph)          output/shift_gap/collab/bundle_age10/read.json (AADC_age10, gMPNN_ref_age10)
  NCN / RGAE, alone and +FLEX         output/shift_gap/collab/bundle_age10/read.json (<B>_age10, <B>_RSTAR, <B>_FLEX)
and every validation pick (g, g_ad, g_cn and the R* scalars) with the recorded picks.
Baseline / backbone checkpoints are not in the review bundle: $FLEX_COLLAB_CKPTS points at their staged copy
(scripts/stage_collab_inputs.py --baseline_weights; default ../FLEX_WWW_camera_ready/ogbl-collab).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import reference as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CKPTS = Path(os.environ.get('FLEX_COLLAB_CKPTS', REPO.parent / 'FLEX_WWW_camera_ready/ogbl-collab'))
BUNDLE = 'output/shift_gap/collab/bundle_age10/'
TOL = 1e-6
pytestmark = pytest.mark.skipif(not R.available(), reason="needs FLEX_REF")


def ref(p):
    return json.loads((R.REF / p).read_text())


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


@pytest.fixture(scope='module')
def flex_result():
    import torch
    from flex_www.collab.data import load_collab
    from flex_www.collab.run import flex
    c = load_collab(R.data_root())
    return flex(c, REPO / 'review_weights/ogbl-collab', REPO / 'review_data/ogbl-collab', torch.device('cuda:0'))


@pytest.fixture(scope='module')
def baseline_result():
    import torch
    from flex_www.collab.data import load_collab
    from flex_www.collab.run import baselines
    c = load_collab(R.data_root())
    return baselines(c, CKPTS, torch.device('cuda:0'))


@pytest.fixture(scope='module')
def backbone_result():
    import torch
    from flex_www.collab.data import load_collab
    from flex_www.collab.run import backbones
    c = load_collab(R.data_root())
    return backbones(c, CKPTS, REPO / 'review_weights/ogbl-collab/iss', REPO / 'review_data/ogbl-collab', torch.device('cuda:0'))


@pytest.mark.gpu
@pytest.mark.parametrize('row', ['ISS10', 'ISS10+AD', 'ISS10+AD+CN_REC2'])
@pytest.mark.parametrize('K', ['H50', 'H20'])
def test_flex_rows(flex_result, row, K):
    want = ref('output/heur_chan/read/read.json')['rows'][row][K]['per_seed']
    assert maxdiff(flex_result['rows'][row][K]['per_seed'], want) <= TOL


@pytest.mark.gpu
def test_rstar_age10_row(flex_result):
    want = ref('output/shift_gap/collab/bundle_age10/read.json')['rows']['ERM_exact_RSTAR']
    for K in ('H50', 'H20'):
        assert maxdiff(flex_result['rows']['RSTAR_AGE10'][K]['per_seed'], want[K]['per_seed']) <= TOL


@pytest.mark.gpu
def test_picks(flex_result):
    runs = flex_result['runs']
    prep = ref('output/shift_gap/collab/bundle_age10/prep_ERM_exact.json')['runs']
    keys = ('gcn_seed', 'twin_seed', 'a6', 'at', 'csd', 'act_mu', 'act_sd', 'g1', 'g2', 'g3', 'best')
    assert [{k: r[k] for k in keys} for r in runs] == [{k: r[k] for k in keys} for r in prep]
    bundle = ref('output/shift_gap/collab/bundle_age10/picks.json')['methods']['ERM_exact']
    assert [(r['g'], r['g_ad']) for r in runs] == [(q['g'], q['g_ad']) for q in bundle]
    assert [r['g_cn'] for r in runs] == [q['g_cn'] for q in ref('output/heur_chan/read/picks.json')['runs']]


@pytest.mark.gpu
@pytest.mark.parametrize('method', ['ERM', 'IRM', 'VRex', 'GroupDRO', 'CORAL', 'DANN', 'SRGNN', 'VGAE'])
def test_gcn_age10_baselines(baseline_result, method):
    want = ref('output/shift_gap/collab/age10_methods/read.json')['methods'][method]
    for K in ('H50', 'H20'):
        assert maxdiff(baseline_result[method][K]['per_seed'], want[f'e|{K}']['per_seed']) <= TOL


@pytest.mark.gpu
def test_aadc_age10_row(baseline_result):
    want = ref('output/shift_gap/collab/bundle_age10/read.json')['rows']['AADC_age10']
    for K in ('H50', 'H20'):
        assert abs(baseline_result['AA-DC'][K]['mean'] - want[K]['mean']) <= TOL


@pytest.mark.gpu
def test_gmpnn_age10_row(baseline_result):
    want = ref(BUNDLE + 'read.json')['rows']['gMPNN_ref_age10']
    for K in ('H50', 'H20'):
        assert maxdiff(baseline_result['gMPNN_ref'][K]['per_seed'], want[K]['per_seed']) <= TOL


@pytest.mark.gpu
@pytest.mark.parametrize('bb', ['NCN', 'RGAE'])
@pytest.mark.parametrize('row,ref_row', [('{}', '{}_age10'), ('{}+RSTAR_AGE10', '{}_RSTAR'), ('{}+FLEX', '{}_FLEX')])
def test_backbone_rows(backbone_result, bb, row, ref_row):
    want = ref(BUNDLE + 'read.json')['rows'][ref_row.format(bb)]
    for K in ('H50', 'H20'):
        assert maxdiff(backbone_result['rows'][row.format(bb)][K]['per_seed'], want[K]['per_seed']) <= TOL


@pytest.mark.gpu
@pytest.mark.parametrize('bb', ['NCN', 'RGAE'])
def test_backbone_picks(backbone_result, bb):
    runs = backbone_result['runs'][bb]
    keys = ('gcn_seed', 'twin_seed', 'a6', 'at', 'csd', 'act_mu', 'act_sd', 'g1', 'g2', 'g3', 'best')
    prep = ref(BUNDLE + f'prep_{bb}.json')['runs']
    assert [{k: r[k] for k in keys} for r in runs] == [{k: r[k] for k in keys} for r in prep]
    assert [r['g'] for r in runs] == [q['g'] for q in ref(BUNDLE + 'picks.json')['methods'][bb]]


@pytest.mark.slow
@pytest.mark.parametrize('Y,proto', [(2014, 'S1a'), (2018, 'S2b')])
def test_iss_environment(Y, proto):
    """Train mode rebuilds the ISS environments exactly (pairs, features, cold flags) from the dataset."""
    import numpy as np
    from flex_www.collab import iss
    from flex_www.collab.data import load_collab
    c = load_collab(R.data_root())
    e = iss.build_env(Y, proto, c.E, c.Y, c.W, valid_edges=c.Ev)
    Z = np.load(R.REF / f'output/flex_inv/s0/envs/env_Y{Y}_{proto}.npz')
    assert np.array_equal(e['P'], Z['P']) and np.array_equal(e['X'], Z['X']) and np.array_equal(e['cold'], Z['cold'])
