"""Equivalence of the ported code with the artifacts of the paper's original runs.

Needs $FLEX_REF (root of the original experiment tree: run JSONs, capture dumps, checkpoints, AA-DC outputs, data);
every test is skipped without it. CPU tests run by default; GPU tests need a CUDA device and `--gpu` (pytest option
defined in conftest.py), slow CPU tests (ogbl-collab AA-DC, ~15 min each) need `--slow`.

  captures   for every headline row with a capture dump, the picks (every stage, gains, gate info) and the per-seed values
             of every metric recomputed from the dumped prediction lists and channels equal the run JSON (exact)
  channels   ACT / REL / ISS / feature cosine / AA-DC refit on the train graph equal the dumped channel arrays (<= 1e-6)
  aadc       the AA-DC baseline (calibration + valid / test) equals the reference outputs (exact)
  eval_once  loading a saved encoder + predictor and evaluating once reproduces the recorded metric (<= 1e-6)
  weights    weights mode end-to-end: same picks, encoder and fused per-seed headline values as the run JSON (<= 1e-6)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import reference as R  # noqa: E402

pytestmark = pytest.mark.skipif(not R.available(), reason='set FLEX_REF to the reference experiment tree')
TOL = 1e-6

CAPS_CELLS = ['cora_CN_0_1_2', 'cora_PA_100_50_0', 'citeseer_PA_100_50_0', 'pubmed_PA_0_50_100', 'pubmed_SP_00_017_026',
              'pubmed_PA_100_50_0', 'ogbl-collab_CN_0_1_2', 'ogbl-collab_PA_0_50_100', 'ogbl-collab_SP_00_017_026',
              'ogbl-collab_CN_2_1_0', 'ogbl-collab_PA_100_50_0', 'ogbl-collab_SP_026_017_00']
CHANNEL_CELLS = ['cora_CN_0_1_2', 'pubmed_PA_0_50_100', 'ogbl-collab_CN_2_1_0']
AADC_FAST = [f'{b}_{s}' for b in ('cora', 'citeseer', 'pubmed') for s in
             ('CN_0_1_2', 'CN_2_1_0', 'PA_0_50_100', 'PA_100_50_0', 'SP_00_017_026', 'SP_026_017_00')] + ['dblp']
AADC_SLOW = [f'ogbl-collab_{s}' for s in ('CN_0_1_2', 'CN_2_1_0', 'PA_0_50_100', 'PA_100_50_0', 'SP_00_017_026', 'SP_026_017_00')]
EVAL_ONCE = [('Photo_to_Computers', 'outputs/baseline_GCN_Computers_lr0.0009383054498574604_20261023_020914.txt'),
             ('ogbl-collab_CN_2_1_0', None),
             ('pubmed_PA_0_50_100', 'outputs/baseline_GCN_pubmed_PA_0_50_100_hfix_lr0.002714722134505889_20261021_153933.txt')]
WEIGHTS = ['citeseer_PA_100_50_0', 'cora_PA_100_50_0']
# Amazon rows: the original trainer saved its checkpoint at the best-Hits@20 epoch while FLEX fused the best-Hits@50 epoch,
# so weights mode from those files reproduces the picks but not the per-seed values (see test_weights_mode_amazon).


@pytest.mark.parametrize('cell', CAPS_CELLS)
def test_captures(cell):
    import caps_check
    r = caps_check.check(cell, caps_check.caps_rows()[cell])
    assert r['identical'], r


@pytest.mark.parametrize('cell', CHANNEL_CELLS)
def test_channels(cell):
    import channels_check
    r = channels_check.check(cell)
    assert r['maxdiff'] <= TOL, r


def _fresh(script, *args):
    """Run a check script in a fresh interpreter (the Amazon AA-DC split is drawn from the untouched torch RNG)."""
    out = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith('{')]


@pytest.mark.parametrize('cell', AADC_FAST + ['Photo_to_Computers_10split', 'Computers_to_Photo_10split'])
def test_aadc(cell):
    r = _fresh('aadc_check.py', '--cell', cell)[-1]
    if cell.endswith('_10split'):
        assert r['per_seed_maxdiff'] <= TOL, r
    else:
        assert max(r['diff'].values()) <= TOL and r['gate_diff'] <= TOL, r


@pytest.mark.slow
@pytest.mark.parametrize('cell', AADC_SLOW)
def test_aadc_collab(cell):
    r = _fresh('aadc_check.py', '--cell', cell)[-1]
    assert max(r['diff'].values()) <= TOL and r['gate_diff'] <= TOL, r


@pytest.mark.gpu
@pytest.mark.parametrize('cell,log', EVAL_ONCE)
def test_eval_once(cell, log):
    import evalonce_check
    r = evalonce_check.check(cell, 'cuda:0', str(R.REF / log) if log else None)
    assert r['maxdiff'] <= TOL, r
    if r['pred_maxdiff'] is not None:
        assert r['pred_maxdiff'] <= TOL, r


@pytest.mark.gpu
@pytest.mark.parametrize('cell', WEIGHTS)
def test_weights_mode(cell):
    import weights_check
    r = weights_check.check(cell, 'cuda:0')
    assert r['picks_match'] and r['base_maxdiff'] <= TOL and r['per_seed_maxdiff'] <= TOL, r


@pytest.mark.gpu
@pytest.mark.xfail(reason='original Amazon checkpoints are the best-Hits@20 epoch, FLEX fused the best-Hits@50 epoch', strict=True)
def test_weights_mode_amazon():
    import weights_check
    r = weights_check.check('Photo_to_Computers', 'cuda:0')
    assert r['picks_match'] and r['per_seed_maxdiff'] <= TOL, r
