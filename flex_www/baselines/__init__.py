"""OOD / generative link-prediction baselines on the shared GCN backbone, one module per method.

Each module exposes `hooks(opts) -> flex_www.train.Hooks` (opts = the config's `method_opts`), and optionally
`before_load(ctx, opts)` for state a trained checkpoint needs before a strict load (EERM's graph editor, CFLP's
treatment matrix). ERM is the core trainer itself (no hooks). Environment labels (IRM, VRex, GroupDRO, DANN, CORAL,
SRGNN, GALA, GI-Graph) come from envs.py. AA-DC (a training-free heuristic) runs through flex_www.flex.aadc_runtime.

    python -m flex_www.baselines.run --config configs/baselines/<method>/<cell>.yaml [--mode train|weights --ckpt DIR]
"""
import importlib

MODULES = {'ERM': None, 'IRM': 'irm', 'VRex': 'vrex', 'GroupDRO': 'groupdro', 'DANN': 'dann', 'CORAL': 'coral',
           'SRGNN': 'srgnn', 'EERM': 'eerm', 'CIGA': 'ciga', 'GALA': 'gala', 'GIGraph': 'gigraph', 'VGAE': 'vgae',
           'SIGVAE': 'sigvae', 'CFLP': 'cflp', 'gMPNN_ref': 'gmpnn_ref'}


def module(method):
    """The method's module (None for ERM)."""
    name = MODULES[method]
    return None if name is None else importlib.import_module(f'{__name__}.{name}')


def hooks(method, opts):
    from ..train import Hooks
    m = module(method)
    return Hooks() if m is None else m.hooks(opts)
