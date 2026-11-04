"""Equivalence of the ported baselines (flex_www.baselines) with the paper's original runs.

Needs $FLEX_REF (see reference.py); GPU tests need `--gpu`, the training-fidelity tests also `--slow`.

  envs        environment labels, RNG state afterwards and the environment-count carry-over equal the original block
  configs     every config's recorded command matches its coefficient in the finals table (regenerated, compared)
  eval_once   strict-loading each method's saved checkpoint (cora / citeseer PA-up, seeds 1-2) and evaluating once gives
              the logged test value of the selection metric at the saved epoch (<= 1e-6)
  aadc        the AA-DC baseline through the baselines CLI equals output/aadc/<cell>.json
  train       seed 1 retrained with the port reproduces the original per-epoch (valid, test) curve
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import reference as R  # noqa: E402

pytestmark = pytest.mark.skipif(not R.available(), reason='set FLEX_REF to the reference experiment tree')
TOL = 1e-6
METHODS = ['ERM', 'IRM', 'VRex', 'GroupDRO', 'DANN', 'CORAL', 'SRGNN', 'EERM', 'CIGA', 'GALA', 'GIGraph', 'VGAE', 'SIGVAE',
           'CFLP', 'gMPNN_ref']
ENV_CELLS = ['cora_CN_0_1_2', 'cora_CN_2_1_0', 'citeseer_SP_00_017_026', 'cora_PA_0_50_100', 'dblp', 'Photo_to_Computers']
EVAL_CELLS = ['cora_PA_0_50_100', 'citeseer_PA_0_50_100']
# GI-Graph trains ~45 min per seed on cora; its fidelity test covers the first epochs (two pool regenerations)
TRAIN_EPOCHS = {'GIGraph': 25}
# methods whose seed-1 curve cannot be bit-identical to the original log
TRAIN_XFAIL = {'EERM': 'the graph editor samples edges with torch.multinomial over K x N^2 pair probabilities on CUDA, '
                       'which is not run-to-run deterministic (two runs of the port from the same seed already sample '
                       'different edges in the first batch); curves differ from epoch 1'}


def _fresh(script, *args):
    out = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith('{')]


@pytest.mark.parametrize('cell', ENV_CELLS)
def test_envs(cell):
    r = _fresh('envs_check.py', '--cell', cell)[-1]
    assert r['identical'], r


def test_configs_regenerate(tmp_path):
    subprocess.run([sys.executable, str(HERE.parent / 'scripts/make_baseline_configs.py'), '--ref', str(R.REF),
                    '--out', str(tmp_path)], check=True, capture_output=True)
    shipped = HERE.parent / 'configs'
    for p in (tmp_path / 'baselines').rglob('*.yaml'):
        assert p.read_text() == (shipped / p.relative_to(tmp_path)).read_text(), p


@pytest.mark.parametrize('cell', ['cora_CN_2_1_0', 'dblp'])
def test_aadc_cli(cell, tmp_path):
    subprocess.run([sys.executable, '-m', 'flex_www.baselines.run', '--config', f'configs/baselines/AA-DC/{cell}.yaml',
                    '--out', str(tmp_path)], check=True, capture_output=True, cwd=HERE.parent,
                   env={**os.environ, 'FLEX_DATA_ROOT': str(R.data_root())})
    got = json.loads((tmp_path / f'AA-DC__{cell}__aadc.json').read_text())
    ref = json.loads((R.REF / 'output/aadc' / f'{cell}.json').read_text())
    assert abs(100 * got['per_seed'][0]['test'] - ref['test']) <= TOL and abs(100 * got['per_seed'][0]['valid'] - ref['valid']) <= TOL


@pytest.mark.gpu
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('cell', EVAL_CELLS)
def test_eval_once(method, cell):
    import baseline_evalonce_check
    r = baseline_evalonce_check.check(method, cell, 'cuda:0')
    assert r['maxdiff'] <= TOL and r['strict_ok'], r


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('method', METHODS)
def test_train_fidelity(method):
    if method in TRAIN_XFAIL:
        pytest.xfail(TRAIN_XFAIL[method])
    import baseline_train_check
    r = baseline_train_check.check(method, 'cora_PA_0_50_100', 'cuda:0', TRAIN_EPOCHS.get(method))
    assert r['first_diff_epoch'] is None and r['epochs_compared'] > 0, r


def test_collapse_watcher():
    """The reported selection skips tied (collapsed) evaluations: replaying VRex pubmed PA-down's original log through
    flex_www.baselines.run.select gives the finals table's per-seed test values (run 5 reselected 318 -> 246)."""
    import re
    import yaml
    from flex_www.baselines.run import select
    cfg = yaml.safe_load((HERE.parent / 'configs/baselines/VRex/pubmed_PA_100_50_0.yaml').read_text())
    pat = re.compile(r'Run: (\d+), (Hits@20|AUC)Epoch: (\d+),.*Train: ([\d.]+)%, Valid: ([\d.]+)%, Test: ([\d.]+)%')
    hist = {}
    for line in (R.REF / cfg['reference']['run_log']).read_text(errors='replace').splitlines():
        m = pat.search(line)
        if m:
            run, key, ep = int(m.group(1)), m.group(2), int(m.group(3))
            rows = hist.setdefault(run, {}).setdefault(key, {})
            rows.setdefault(ep, tuple(float(m.group(i)) / 100 for i in (4, 5, 6)))
    got = []
    for run in sorted(hist):
        h = {k: [v[e] for e in sorted(v)] for k, v in hist[run].items()}
        got.append(round(100 * select(h, 'Hits@20', 1)[1], 2))
    assert got == cfg['reference']['finals']['seed_test'], (got, cfg['reference']['finals']['seed_test'])
