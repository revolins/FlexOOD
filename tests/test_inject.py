"""INJ arm (NCN / Refined-GAE + FLEX on DBLP) against the original runs. Needs $FLEX_REF and a CUDA device (--gpu).

  weights  the backbone finals' checkpoints (same command and seeds as the INJ runs, which saved no weights) -> encoder
           base_per_seed, every injection candidate's validation pick, the choice and the fused per-seed MRR (exact)
  train    the full row in train mode (10 seeds) -> the same quantities (exact; the originals ran on 24 GB A5000s)
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import reference as R  # noqa: E402

pytestmark = [pytest.mark.skipif(not R.available(), reason='set FLEX_REF to the reference experiment tree'), pytest.mark.gpu]
TOL = 1e-6


@pytest.mark.parametrize('backbone', ['NCN', 'RGAE'])
@pytest.mark.parametrize('mode', ['weights', 'train'])
def test_inject(backbone, mode):
    import inject_check
    r = inject_check.check(backbone, mode, 'cuda:0')
    assert r['g'] == r['ref_g'] and r['inj_choice'] == r['ref_choice'] and r['inj_by_frac_equal'], r
    assert r['base_maxdiff'] <= TOL and r['per_seed_maxdiff'] <= TOL, r
    if mode == 'train':
        assert r['picks_identical'], r
