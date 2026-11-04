"""FLEX row driver.

    python -m flex_www.flex.run --config configs/flex/<cell>.yaml --mode train   [--out DIR] [--device cuda:0]
    python -m flex_www.flex.run --config configs/flex/<cell>.yaml --mode weights --ckpt DIR [--ref RUN.json]

train    the full row: train the encoder for every seed (AUG rows: replay aux loss + combined epoch selection), fit the
         channels on the train graph, run the validation picks, report every metric at its selection epoch; writes
         <out>/<cell>.json and the checkpoints FLEX fuses (fused_seed<s>_* on AUG rows, else base_*_run<r>_*).
weights  load each seed's checkpoint, evaluate once, refit the channels (deterministic), rerun the validation picks,
         assert they equal the config's recorded picks, report the headline metric. With --ref (the JSON of the run
         that wrote the weights) the encoder-only and fused per-seed values are checked against it.

Rows (variant = row + suffixes; `_AD` adds the AA-DC stage, `_G` the guard + gated AD stage, `_B` the joint base stage):
  RSTAR_ISS        ERM GCN, channels ACT / REL / ISS, plain picks
  RSTAR_ISS_VALIN  same, test pairs scored on train + validation edges (PA-up cells)
  RSTAR_ISS_DIR    picks on the validation fold along the shift
  RSTAR_ISS_SP     SP columns; picks on full validation + the fold (PA cells: no fold)
  RSTAR_ISS_SPX    SP columns + feature-cosine stage; gated picks
  RSTAR_ISS_AUG    replay-trained encoder (combined selection), plain picks
  RSTAR_ISS_AG / _AGX / _AGX_VALIN   replay encoder (PA-down edits only), SP columns, gated picks (+ X stage)
  RSTAR_ISS_INJ    RSTAR_ISS + the injection arm (flex_www.flex.inject; the NCN / Refined-GAE + FLEX DBLP rows)
"""
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import direction as DIR
from .aadc_runtime import AADCChannel, run_cell
from .channels import Channels, chans_on, eval_pairs, graph_of, source_domain_pseudo
from .fusion import MetricFn, fuse, run_picks, stage_ad, stage_base
from .replay import Replay, fold_metric, selection_fold
from ..data import cell_info
from ..evaluate import eval_once
from ..train import Hooks, RunSpec, fit, prepare

AG_ROWS = ('RSTAR_ISS_AG', 'RSTAR_ISS_AGX', 'RSTAR_ISS_AGX_VALIN')
AUG_ROWS = ('RSTAR_ISS_AUG',) + AG_ROWS
GATED_ROWS = AG_ROWS + ('RSTAR_ISS_SPX',)
X_ROWS = ('RSTAR_ISS_SPX', 'RSTAR_ISS_AGX', 'RSTAR_ISS_AGX_VALIN')
SP_ROWS = ('RSTAR_ISS_SP',) + GATED_ROWS
INJ_ROW = 'RSTAR_ISS_INJ'
ROWS = ('RSTAR_ISS', 'RSTAR_ISS_VALIN', 'RSTAR_ISS_DIR', 'RSTAR_ISS_SP') + AUG_ROWS + ('RSTAR_ISS_SPX', INJ_ROW)


def parse_variant(variant):
    """'RSTAR_ISS_AUG_AD_B' -> (row, aadc_stage, gap_rules, base_stage)."""
    v = variant
    base = v.endswith('_B')
    gap = base or v.endswith('_G')
    if base or v.endswith('_G'):
        v = v[:-2]
    aadc = v.endswith('_AD')
    if aadc:
        v = v[:-3]
    assert v in ROWS, variant
    return v, aadc, gap, base


@dataclass
class RowConfig:
    cell: str
    variant: str
    spec: RunSpec
    expected: dict = field(default_factory=dict)       # recorded picks {'g': [...], 'b': ...}
    aadc_calibration: dict = None                       # {'gate_thresholds', 'anchor_scale'} (else computed)

    @property
    def row(self):
        return parse_variant(self.variant)[0]

    @property
    def flags(self):
        return parse_variant(self.variant)[1:]


def load_config(path, data_root=None, warm_start_dir=None):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    spec = RunSpec.from_config({**c['train'], 'cell': c['cell'], 'data_root': data_root, 'warm_start_dir': warm_start_dir})
    return RowConfig(cell=c['cell'], variant=c['variant'], spec=spec, expected=c.get('expected', {}),
                     aadc_calibration=c.get('aadc_calibration'))


def _cpu_state(mod):
    return {k: t.detach().cpu().clone() for k, t in mod.state_dict().items()}


# ------------------------------------------------------------------------------------------------ capture (train mode)
class Capture:
    """Sees every evaluation: per seed and metric the prediction list at the FIRST strict maximum of validation; on AUG
    rows also the combined-selection epoch (validation + fold metric) with its encoder / predictor state."""

    def __init__(self, row, cell, metric, mf, sp=False):
        self.row, self.cell, self.metric, self.mf, self.sp = row, cell, metric, mf, sp
        self.best, self.comb, self.fold, self.ctx = {}, {}, {}, {}

    def on_eval(self, ctx, n_eval, epoch, results, preds):
        s = ctx.seed
        self.ctx[s] = ctx
        B = self.best.setdefault(s, {})
        st = None
        res = {k: tuple(v) for k, v in results.items()}
        for m, r in res.items():
            if m not in B or r[1] > B[m]['v']:
                B[m] = {'v': r[1], 'ep': n_eval, 'P': preds, 'res': res}
                if self.row == INJ_ROW:          # INJ re-evaluates every selection state on the injected graph (memory only)
                    if st is None:
                        st = {'model': _cpu_state(ctx.model), 'predictor': _cpu_state(ctx.predictor)}
                    B[m]['state'] = st
        if self.row in AUG_ROWS and self.metric in res:
            if s not in self.fold:
                self.fold[s] = selection_fold(self.cell, graph_of(ctx.graph.data.adj_t), ctx.edges[1].cpu().numpy(), self.sp)
            v = res[self.metric][1] + fold_metric(self.mf, preds, self.fold[s], self.metric)
            if s not in self.comb or v > self.comb[s]['v']:
                self.comb[s] = {'v': v, 'ep': n_eval, 'P': preds, 'res': res,
                                'state': {'model': _cpu_state(ctx.model), 'predictor': _cpu_state(ctx.predictor)}}

    def sel(self, s, m):
        return self.comb[s]['P'] if self.row in AUG_ROWS else self.best[s][m]['P']

    def base(self, s, m):
        return self.comb[s]['res'][m] if self.row in AUG_ROWS else self.best[s][m]['res'][m]


# ------------------------------------------------------------------------------------------------ channels + selection
def fit_channels(cfg, seeds, graphs, edges, log, aadc=None):
    """Channel dicts per seed. graphs / edges: seed -> Graph / evaluation edge list (the last seed's are used unless the
    seeds have their own splits, i.e. Amazon). Reproduces the order of every random draw of the paper's runs: the NumPy
    global state is the one right after init_seed(last seed) (liblinear seeds itself from it), and on per-seed splits the
    source-domain split of each seed is drawn from the torch RNG left by the previous ISS fit."""
    row, (aadc_stage, _, _) = cfg.row, cfg.flags
    cell = cell_info(cfg.cell)
    sp = row in SP_ROWS
    last = seeds[-1]
    np.random.seed(last)
    domain = cell['family'] == 'crossdomain'

    def make(s_list, s_key):
        g, e = graphs[s_key], edges[s_key]
        A = graph_of(g.data.adj_t)
        x = g.data.x.detach().cpu().numpy()
        src = (lambda: source_domain_pseudo(cell['base'], cfg.spec.data_root)) if domain else None
        CH = Channels(cell, A, x, e[0].detach().cpu().numpy().reshape(-1, 2), s_list, log, sp=sp, source_pseudo=src)
        A_full = graph_of(g.data.full_adj_t) if g.val_input_for_test else None
        return CH, A, A_full, eval_pairs(e)
    CH, A, A_full, pairs = make(seeds, last)
    if aadc_stage:
        CH.aadc = aadc
    ch0 = chans_on(CH, A, A_full, pairs)
    chans = {s: ch0 for s in seeds}
    same = lambda t, u: t.shape == u.shape and bool(torch.equal(t.detach().cpu(), u.detach().cpu()))  # noqa: E731
    per_seed_split = any(not all(same(edges[s][i], edges[last][i]) for i in range(5)) for s in seeds)
    if per_seed_split:
        assert row not in ('RSTAR_ISS_DIR', 'RSTAR_ISS_SP') + GATED_ROWS and not aadc_stage, f'{row} assumes one split'
        log('per-seed splits: channels fitted on each seed\'s own train graph and evaluation pairs')
        for s in seeds:
            CH_s, A_s, Af_s, p_s = make([s], s)
            chans[s] = chans_on(CH_s, A_s, Af_s, p_s)
        CH = CH_s
    return CH, A, pairs, chans, per_seed_split


def compute_masks(cfg, A, pairs, CH):
    """Validation folds the row's picks use: 'dir' (DIR), 'tv2' (SP / gated rows: threshold_v2 with the SP columns)."""
    row, cell = cfg.row, cell_info(cfg.cell)
    masks = {}
    if row == 'RSTAR_ISS_DIR':
        masks['dir'] = (DIR.threshold(DIR.spec(cell), A, pairs[0], False)[0], None)
    if row in SP_ROWS:
        m, _, info = DIR.threshold_v2(DIR.spec(cell), A, pairs[0], CH.Xp, sp=True)
        masks['tv2'] = (m, info)
    return masks


def select(row, cell_key, flags, seeds, sel, chans, mf, metric, metrics, masks, log=print, inject=None):
    """The row's validation picks and its report: rec with 'picks' and 'per_seed' {metric: [(valid, test)]}.
    inject (INJ row): callable(g, pinfo, ch0) -> (final_ch, final_sel, info, g), run after the plain picks."""
    aadc_stage, gap_rules, base_stage = flags
    gap_rules = gap_rules or base_stage
    cell = cell_info(cell_key)
    epP = {s: sel(s, metric) for s in seeds}
    fold = gate = None
    rec = {}
    final_sel, final_ch = sel, chans
    if row == 'RSTAR_ISS_DIR':
        fold = masks['dir'][0]
    if row == 'RSTAR_ISS_SP':
        fold = masks['tv2'][0]
        if DIR.spec(cell)[0] == 'PA':        # PA cells carry no fold
            fold = None
        gA, pA = run_picks(mf, seeds, epP, chans, metric, log)
        if fold is None:
            g, pinfo = gA, dict(pA, reused='features_only')
        else:
            g, pinfo = run_picks(mf, seeds, epP, chans, metric, log, fold=fold, combined=True)
    elif row in GATED_ROWS:
        gate = masks['tv2'][0]
        if DIR.spec(cell)[0] == 'PA':
            gate = None
        xs = row in X_ROWS
        gU, pU = run_picks(mf, seeds, epP, chans, metric, log, xstage=xs, guard=gap_rules)
        if gate is None:
            g, pinfo = gU, dict(pU, reused='ungated')
        else:
            g, pinfo = run_picks(mf, seeds, epP, chans, metric, log, gate=gate, xstage=xs, guard=gap_rules)
    else:
        g, pinfo = run_picks(mf, seeds, epP, chans, metric, log, fold=fold)
    rec['picks'] = {'g': list(g), **pinfo}
    if fold is not None:
        rec['picks']['val_full_at_g'] = float(np.mean([mf.value(fuse(epP[s], chans[s], s, g), metric, 1) for s in seeds]))
    if row == INJ_ROW:
        final_ch, final_sel, rec['inj'], g = inject(g, pinfo, chans[seeds[0]])
        rec['picks']['g'] = list(g)
    rej = list(pinfo.get('guard_rejected', []))
    sgate = gate if gate is not None else fold
    if aadc_stage:
        g, iad, gb, VA, r_ = stage_ad(mf, seeds, final_sel, final_ch, metric, g, log, gate=sgate, rules=gap_rules)
        rej += r_
        rec['picks'].update(stage_ad=iad, g_no_ad=list(gb), g=list(g), val_mean_ad=float(np.mean(VA[g])))
    bsel = 1.0
    if base_stage:
        g, bsel, ib, gnb, b0, r_ = stage_base(mf, seeds, final_sel, final_ch, metric, g, log, gate=sgate)
        rej += r_
        rec['picks'].update(stage_base=ib, g_no_base=list(gnb), g=list(g), b=bsel)
    if gap_rules:
        rec['picks']['guard_rejected'] = sorted(set(rej))
    rec['per_seed'] = {m: [tuple(float(x) for x in mf.full(fuse(final_sel(s, m), final_ch[s], s, g, bsel))[m][1:3]) for s in seeds]
                       for m in metrics}
    rec['seeds'] = list(seeds)
    return rec


def _aadc_channel(cfg, log):
    c = cfg.aadc_calibration
    if c is None:
        log('calibrating AA-DC on validation (no stored calibration)')
        c = run_cell(cfg.cell, root=cfg.spec.data_root)
    return AADCChannel(c['gate_thresholds'], c['anchor_scale'])


def _injector(cfg, mf, CH, A, pairs, ctx, seeds, metrics, metric, best, log):
    from .inject import run_injection
    cell = cell_info(cfg.cell)
    return lambda g, pinfo, ch0: run_injection(mf, cell, CH, A, ctx, pairs, seeds, metrics, metric, best, g, pinfo, ch0, log)


def _mf(split):
    return MetricFn(split.layout, split.per_query)


# ------------------------------------------------------------------------------------------------ modes
def train_mode(cfg, device, out_dir, log=print):
    spec, row = cfg.spec, cfg.row
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / 'ckpt'
    spec.checkpoint_dir = str(ckpt_dir)
    cell = cell_info(cfg.cell)
    t0 = time.time()
    capture, replay = None, None
    hooks = Hooks()
    if row in AUG_ROWS:
        replay = Replay(cell, pa_only=row in AG_ROWS, sp=row in SP_ROWS)
        hooks.extra_loss = lambda ctx, epoch: replay.loss_factory(ctx.model, ctx.predictor, ctx.graph.data, ctx.split.valid_pos)
    graphs, edges, results = {}, {}, {}
    bs = None
    for run_idx, s in enumerate(spec.seeds()):
        ctx = prepare(spec, s, run_idx, device, hooks, bs)
        bs = ctx.batch_sizes
        if capture is None:
            capture = Capture(row, cell, ctx.split.metric, _mf(ctx.split), sp=row in SP_ROWS)
            hooks.on_eval = capture.on_eval
        results[s] = fit(ctx, hooks, log)
        graphs[s], edges[s] = ctx.graph, [t.detach().cpu() for t in ctx.edges]
        if row in AUG_ROWS:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(capture.comb[s]['state']['model'], ckpt_dir / f'fused_seed{s}_model.pt')
            torch.save(capture.comb[s]['state']['predictor'], ckpt_dir / f'fused_seed{s}_predictor.pt')
            capture.comb[s]['state'] = None
    seeds = sorted(capture.best)
    metric, mf = capture.metric, capture.mf
    metrics = [m for m in capture.best[seeds[0]] if m in capture.best[seeds[0]][metric]['res']]
    rec = {'cell': cfg.cell, 'variant': cfg.variant, 'metric': metric, 'mode': 'train'}
    aadc = _aadc_channel(cfg, log) if cfg.flags[0] else None
    CH, A, pairs, chans, pss = fit_channels(cfg, seeds, graphs, edges, log, aadc)
    worst = 0.0                                  # self-test: g = 0 through the fused path reproduces the captured metrics
    for s in seeds:
        for m in metrics:
            got = mf.full(fuse(capture.sel(s, m), chans[s], s, (0.0, 0.0, 0.0)))[m]
            ref = capture.base(s, m)
            worst = max(worst, abs(float(got[1]) - float(ref[1])), abs(float(got[2]) - float(ref[2])))
    assert worst <= 1.5e-4, f'fused path at g = 0 does not reproduce the captured metrics ({worst})'
    masks = compute_masks(cfg, A, pairs, CH)
    inject = _injector(cfg, mf, CH, A, pairs, capture.ctx[seeds[-1]], seeds, metrics, metric, capture.best, log) if row == INJ_ROW else None
    rec.update(select(row, cfg.cell, cfg.flags, seeds, capture.sel, chans, mf, metric, metrics, masks, log, inject))
    rec['base_per_seed'] = {m: [tuple(float(x) for x in capture.base(s, m)[1:3]) for s in seeds] for m in metrics}
    rec.update(selftest_g0_maxdiff=worst, per_seed_split=pss, env_rule=CH.env_rule, iss_keep=CH.iss_keep,
               iss_fallback=CH.iss_fallback, replay=getattr(replay, 'info', None), runtime_s=time.time() - t0,
               fused_ckpt_eval_index={s: c['ep'] for s, c in capture.comb.items()} if row in AUG_ROWS else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cfg.cell}.json').write_text(json.dumps(rec, indent=1, default=str))
    return rec


def checkpoint_paths(cfg, ckpt_dir, seed, run_idx, dataset):
    d = Path(ckpt_dir)
    if (d / f'seed{seed}_model.pt').exists():                # released weights: one fused state per seed
        return d / f'seed{seed}_model.pt', d / f'seed{seed}_predictor.pt'
    if (d / f'fused_seed{seed}_model.pt').exists() or cfg.row in AUG_ROWS:
        return d / f'fused_seed{seed}_model.pt', d / f'fused_seed{seed}_predictor.pt'
    return d / cfg.spec.checkpoint_name(dataset, run_idx, 'model'), d / cfg.spec.checkpoint_name(dataset, run_idx, 'predictor')


def weights_mode(cfg, device, ckpt_dir, ref=None, log=print, check_picks=True):
    """Eval once per seed from the row's checkpoints, refit channels, rerun the picks. ref: dict of the run JSON that wrote
    the weights (optional). Returns rec (headline metric only)."""
    spec, row = cfg.spec, cfg.row
    graphs, edges, P, base, states = {}, {}, {}, {}, {}
    bs, mf, metric, last_ctx = None, None, None, None
    seeds = spec.seeds()
    for run_idx, s in enumerate(seeds):
        ctx = prepare(spec, s, run_idx, device, None, bs, warm_start=False)
        bs = ctx.batch_sizes
        mpath, ppath = checkpoint_paths(cfg, ckpt_dir, s, run_idx, ctx.split.dataset)
        res, preds = eval_once(ctx.model, ctx.predictor, ctx.graph, ctx.split, ctx.edges, ctx.batch_sizes, mpath, ppath,
                               spec.use_valedges_as_input)
        mf, metric = _mf(ctx.split), ctx.split.metric
        P[s], base[s] = preds, tuple(float(x) for x in res[metric][1:3])
        states[s] = {'model': _cpu_state(ctx.model), 'predictor': _cpu_state(ctx.predictor)}
        last_ctx = ctx
        graphs[s], edges[s] = ctx.graph, [t.detach().cpu() for t in ctx.edges]
        log(f'seed {s}: encoder {metric} valid {base[s][0]:.4f} test {base[s][1]:.4f}')
    rec = {'cell': cfg.cell, 'variant': cfg.variant, 'metric': metric, 'mode': 'weights',
           'base_per_seed': {metric: [base[s] for s in seeds]}}
    if ref is not None:
        rb = ref['base_per_seed'][metric]
        rec['base_maxdiff'] = max(max(abs(a - b) for a, b in zip(base[s], rb[i])) for i, s in enumerate(seeds))
        log('encoder vs reference base_per_seed: max |diff|', rec['base_maxdiff'])
    aadc = _aadc_channel(cfg, log) if cfg.flags[0] else None
    CH, A, pairs, chans, pss = fit_channels(cfg, seeds, graphs, edges, log, aadc)
    masks = compute_masks(cfg, A, pairs, CH)
    inject = None
    if row == INJ_ROW:                 # the saved checkpoint is the headline metric's selection state of each seed
        best = {s: {metric: {'P': P[s], 'state': states[s], 'ep': 1}} for s in seeds}
        inject = _injector(cfg, mf, CH, A, pairs, last_ctx, seeds, [metric], metric, best, log)
    rec.update(select(row, cfg.cell, cfg.flags, seeds, lambda s, m: P[s], chans, mf, metric, [metric], masks, log, inject))
    rec.update(per_seed_split=pss, env_rule=CH.env_rule, iss_keep=CH.iss_keep)
    exp = cfg.expected or {}
    if exp.get('g') is not None:
        rec['picks_match'] = (rec['picks']['g'] == list(exp['g'])) and (rec['picks'].get('b', 1.0) == (exp.get('b') or 1.0))
        log('picks', rec['picks']['g'], 'b', rec['picks'].get('b', 1.0), 'recorded', exp['g'], exp.get('b'),
            'MATCH' if rec['picks_match'] else 'MISMATCH')
        if check_picks:
            assert rec['picks_match'], f"picks {rec['picks']['g']} != recorded {exp['g']}"
    if ref is not None:
        rt = ref['per_seed'][metric]
        rec['per_seed_maxdiff'] = max(max(abs(a - b) for a, b in zip(rec['per_seed'][metric][i], rt[i])) for i in range(len(seeds)))
        log('fused vs reference per_seed: max |diff|', rec['per_seed_maxdiff'])
    t = [v[1] for v in rec['per_seed'][metric]]
    rec['test_mean'], rec['test_std'] = float(np.mean(t)), float(np.std(t))
    log(f"{cfg.cell} {cfg.variant}: test {metric} {100 * rec['test_mean']:.2f} +- {100 * rec['test_std']:.2f}")
    return rec


# ------------------------------------------------------------------------------------------------ dumped captures
def load_captures(path):
    """Prediction lists, channels and masks dumped by the original runs (caps_<stamp>.npz) ->
    (meta, seeds, sel, chans, masks, mf). The original Cap layout: P<i>_<j> prediction tensors, C<i>_<j>_<name> channels
    (iss per seed), M_<name> masks; meta maps seed / metric to those indices."""
    Z = np.load(path, allow_pickle=False)
    meta = json.loads(str(Z['meta']))
    Ps = [[torch.from_numpy(Z[f'P{i}_{j}']) for j in range(n)] for i, n in enumerate(meta['P_len'])]
    Cs = [[dict({k: Z[f'C{i}_{j}_{k}'] for k in ks}, iss={s: Z[f'C{i}_{j}_iss_{s}'] for s in iss}) for j, ks in enumerate(keys)]
          for i, (keys, iss) in enumerate(zip(meta['C_keys'], meta['C_iss']))]
    seeds = [int(s) for s in meta['seeds']]
    S = {int(s): {m: Ps[i] for m, i in d.items()} for s, d in meta['final_sel'].items()}
    chans = {int(s): Cs[i] for s, i in meta['final_ch'].items()}
    masks = {nm: (None if v['none'] else Z[f'M_{nm}'], v['info']) for nm, v in meta['masks'].items()}
    kind = meta['call']['kind']
    mf = MetricFn('heart' if kind == 'hard' else ('per_query' if meta['pqn'] else 'pooled'), int(meta['pqn']))
    return meta, seeds, (lambda s, m: S[s][m]), chans, masks, mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=('train', 'weights'), required=True)
    ap.add_argument('--ckpt', help='weights mode: directory of the row checkpoints')
    ap.add_argument('--ref', help='weights mode: JSON of the run that wrote the weights')
    ap.add_argument('--warm_start_dir', help='Amazon train mode: source-domain checkpoints')
    ap.add_argument('--data_root')
    ap.add_argument('--out', default='output/flex')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    cfg = load_config(a.config, a.data_root, a.warm_start_dir)
    dev = torch.device(a.device)
    if a.mode == 'train':
        rec = train_mode(cfg, dev, Path(a.out) / cfg.variant / cfg.cell)
    else:
        ref = json.loads(Path(a.ref).read_text()) if a.ref else None
        rec = weights_mode(cfg, dev, a.ckpt, ref)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'{cfg.cell}_weights.json').write_text(json.dumps(rec, indent=1, default=str))
    m = rec['metric']
    t = np.array([v[1] for v in rec['per_seed'][m]])
    print(f"{cfg.cell} {cfg.variant} test {m}: {100 * t.mean():.2f} +- {100 * t.std():.2f}", flush=True)


if __name__ == '__main__':
    main()
