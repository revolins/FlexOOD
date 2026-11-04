"""Generate configs/baselines/<method>/<cell>.yaml (and configs/backbones/) from the recorded commands of the paper's runs
(never edit the YAML by hand).

    python scripts/make_baseline_configs.py --ref <reference tree> --paper <paper_numbers.csv>

Maintainer script: it needs the original experiment tree and the paper's per-cell numbers table, neither of which ships
with this repository. The generated configs are already in configs/; reviewers do not need to run it.

One config per (method, cell) of the paper numbers table that has a number, except the real ogbl-collab cell (its
baselines run through the collab pipeline). Source of every run, in order:
  * the PA-up Planetoid cells (cora / citeseer / pubmed PA_0_50_100) report the validation-edges-as-test-input reruns:
    ood_final_valin/_manifest.jsonl;
  * otherwise the run named by results/finals/finals_10seed.csv (`run_log`), whose exact command is in the
    `_manifest.jsonl` of its campaign directory (ood_final_hfix / ood_final_v3);
  * results/finals/overrides.json replaces fields of a run (VGAE pubmed SP-up: patience 200).
Checks per run: the command's coefficient flag / value equals the finals table's; every flag of the command equals the
argparse Namespace printed at the top of its run log; the method options are taken from that Namespace (so defaults the
command did not spell out are recorded too). Backbones: ood_final_backbone/<NCN|RGAE>/<cell>/ (Namespace of final.txt,
checked against model_flags.txt) and the FLEX-on-backbone row scripts/paper/build_tables.py:backbone_flex() selects.
"""
import argparse
import ast
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PAPER_METHOD = {'GCN (ERM)': 'ERM', 'VREx': 'VRex', 'gMPNN': 'gMPNN_ref'}
SKIP_METHODS = ('GCN+FLEX', 'NCN', 'NCN+FLEX', 'Refined-GAE', 'Refined-GAE+FLEX')
VALIN_CELLS = ('cora_PA_0_50_100', 'citeseer_PA_0_50_100', 'pubmed_PA_0_50_100')
AMAZON = ('Photo_to_Computers', 'Computers_to_Photo')
TRAIN_INT = ('epochs', 'eval_steps', 'kill_cnt', 'runs', 'seed', 'batch_size', 'val_batch_size', 'test_batch_size')
HP = ('hidden_channels', 'num_layers', 'num_layers_predictor', 'dropout', 'edge_drop', 'lr', 'l2')
# command flags that do not reach the numbers (logging, device, memory cap, saving) or are implied by the cell
INERT = ('device', 'log_steps', 'gpu_mem_cap_gb', 'save', 'checkpoint_dir', 'score_model', 'gnn_model', 'use_rand',
         'dblp_full_batch', 'dataset')
# Namespace values every ported run must have (paths of the trainer that were not ported)
MUST = dict(gnn_model='GCN', score_model='mlp_score', vae_no_sample=False, cflp_t_off=False, cflp_disc='lin',
            intervention_support='', train_rand_edges=0.0, dense_edges='', node_pe_path='', load_sigvae=False,
            use_valedges_as_input=False, eval_only=False, remove_edge_aggre=False, gigraph_use_diffusion=True,
            dblp_legacy_leaky=False, leaky_full_adj=False, gen_tune=False)


def method_opts(m, ns):
    """Method options (flex_www.baselines.<method>.hooks) from the run's Namespace."""
    g = lambda *ks: {k.split(':')[0]: ns[k.split(':')[-1]] for k in ks}  # noqa: E731  'opt:namespace_key'
    envs = g('loss_coeff', 'num_envs') if m in ('IRM', 'VRex', 'GroupDRO', 'DANN', 'CORAL', 'SRGNN', 'GALA', 'GIGraph') else {}
    table = {
        'ERM': lambda: {},
        'IRM': lambda: {**envs, **g('anneal_iters:irm_anneal_iters')},
        'VRex': lambda: {**envs, **g('anneal_iters:vrex_anneal_iters')},
        'GroupDRO': lambda: envs, 'DANN': lambda: envs, 'CORAL': lambda: envs,
        'SRGNN': lambda: {**envs, **g('cmd_k:srgnn_cmd_k')},
        'EERM': lambda: g('loss_coeff', 'K'),
        'CIGA': lambda: g('ratio:ciga_ratio', 'contrast_coe:ciga_contrast_coe', 'contrast_t:ciga_contrast_t',
                          'spu_coe:ciga_spu_coe', 'sampling:ciga_contrast_sampling',
                          'contrast_max_pairs:ciga_contrast_max_pairs'),
        'GALA': lambda: {**envs, **g('ratio:gala_ratio', 'contrast_coe:gala_contrast_coe', 'contrast_t:gala_contrast_t',
                                     'spu_coe:gala_spu_coe', 'contrast_max_pairs:gala_contrast_max_pairs')},
        'GIGraph': lambda: {**envs, **{k[len('gigraph_'):]: v for k, v in ns.items()
                                       if k.startswith('gigraph_') and k != 'gigraph_score_coe'}},
        'VGAE': lambda: g('loss_coeff'),
        'SIGVAE': lambda: g('loss_coeff', 'K:sigvae_K', 'J:sigvae_J', 'noise_dim:sigvae_noise_dim'),
        'CFLP': lambda: g('loss_coeff', 'beta:cflp_beta', 'gamma:cflp_gamma', 'num_np:cflp_num_np',
                          'ft_frac:cflp_ft_frac', 'dgi_epochs:cflp_dgi_epochs', 'dgi_hidden:cflp_dgi_hidden'),
        'gMPNN_ref': lambda: {},
    }
    return table[m]()


def parse_argv(argv):
    out, i = {}, 0
    while i < len(argv):
        k = argv[i]
        assert k.startswith('--'), argv
        flag = i + 1 == len(argv) or argv[i + 1].startswith('--')
        out[k[2:]] = True if flag else argv[i + 1]
        i += 1 if flag else 2
    return out


def namespace(log):
    """argparse Namespace printed at the top of a gnn_baseline.py log -> dict."""
    line = next(ln for ln in Path(log).read_text(errors='replace').splitlines() if ln.startswith('Namespace('))
    out = {}
    for m in re.finditer(r"(\w+)=('(?:[^']*)'|[^,]+)(?:, |\)$)", line[len('Namespace('):]):
        try:
            out[m.group(1)] = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            out[m.group(1)] = m.group(2)
    return out


def trainer_defaults(ref):
    """argparse defaults of the original trainer (gnn_baseline.py), read from its source with the ast module."""
    d = {}
    for n in ast.walk(ast.parse((ref / 'gnn_baseline.py').read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument':
            kw = {k.arg: k.value for k in n.keywords}
            dest = kw['dest'].value if 'dest' in kw else n.args[0].value.lstrip('-')
            if 'default' in kw:
                d.setdefault(dest, ast.literal_eval(kw['default']))
            elif 'action' in kw and kw['action'].value == 'store_true':
                d.setdefault(dest, False)
            else:
                d.setdefault(dest, None)
    return d


def not_in_command(a, ns, defaults):
    """Namespace values that neither the command nor the trainer defaults explain (a manifest command older than the
    run that wrote the log, or a default changed since)."""
    skip = set(INERT) | {'load_checkpoint_dir'}
    return {k: v for k, v in ns.items() if k not in a and k not in skip and k in defaults and v != defaults[k]}


def same(flag_value, ns_value):
    if flag_value is True:
        return ns_value is True
    if isinstance(ns_value, bool):
        return str(ns_value) == flag_value
    if isinstance(ns_value, (int, float)):
        return float(flag_value) == float(ns_value)
    return str(ns_value) == flag_value


def check_cmd(a, ns, where, backbone='GCN'):
    bad = {k: (v, ns.get(k)) for k, v in a.items() if k not in ('device', 'checkpoint_dir') and not same(v, ns.get(k))}
    assert not bad, f'{where}: command vs Namespace {bad}'
    must = {**MUST, 'gnn_model': backbone}
    wrong = {k: (ns.get(k), v) for k, v in must.items() if k in ns and ns[k] != v}
    assert not wrong, f'{where}: unported trainer path {wrong}'


def train_section(m, ns, cell):
    t = {k: int(ns[k]) for k in TRAIN_INT}
    if m == 'gMPNN_ref':
        t['hp'] = {'dropout': float(ns['dropout']), 'lr': float(ns['lr']), 'l2': float(ns['l2']),
                   'gmpnn_hidden_channels': int(ns['gmpnn_hidden_channels']),
                   'gmpnn_predictor_layers': int(ns['gmpnn_predictor_layers']), 'gmpnn_batch_size': int(ns['gmpnn_batch_size'])}
    else:
        t['hp'] = {k: (int(ns[k]) if k in ('hidden_channels', 'num_layers', 'num_layers_predictor') else float(ns[k])) for k in HP}
    t['backbone'] = 'GCN'
    t['val_edges_test_input'] = bool(int(ns.get('val_edges_test_input', 0)))
    t['x_d'] = bool(ns.get('x_d', False))
    t['ood_method'] = None if m == 'ERM' else m
    assert t['x_d'] == (cell in AMAZON), cell
    if cell == 'dblp':
        assert ns.get('dblp_full_batch'), 'only the full-batch DBLP trainer is ported'
    return t


def manifests(ref):
    out = {}
    for camp in ('ood_final_hfix', 'ood_final_v3', 'ood_final_valin'):
        for line in (ref / camp / '_manifest.jsonl').read_text().splitlines():
            x = json.loads(line)
            if x['status'] == 'ok':
                key = (camp, x['method'], x['cell'])
                assert key not in out, key
                out[key] = x
    return out


def run_source(ref, m, cell, finals, man, overrides):
    """(campaign, manifest entry, run log, override or None, finals row or None)."""
    ov = next((o for o in overrides if o['method'] == m and o['cell'] == cell), None)
    if cell in VALIN_CELLS:
        e = man[('ood_final_valin', m, cell)]
        ck = parse_argv(e['cmd'][2:])['checkpoint_dir']
        return 'ood_final_valin', e, ref / (ck[:-len('_ckpt')] + '.txt'), ov, None
    f = finals[(m, cell)]
    camp = f['run_log'].split('/')[0]
    if ov is not None:
        camp = 'ood_final_hfix'
        e = man[(camp, m, cell)]
        log = ref / Path(e['cmd'][e['cmd'].index('--checkpoint_dir') + 1][:-len('_ckpt')] + '.txt')
        return camp, e, log, ov, f
    e = man[(camp, m, cell)]
    a = parse_argv(e['cmd'][2:])
    if f['checkpoint_dir'] and 'checkpoint_dir' in a:
        assert a['checkpoint_dir'] == f['checkpoint_dir'], (m, cell)
    return camp, e, ref / f['run_log'], ov, f


def baseline_config(ref, m, cell, paper, finals, man, overrides, defaults):
    camp, e, log, ov, f = run_source(ref, m, cell, finals, man, overrides)
    a = parse_argv(e['cmd'][2:])
    assert a.get('ood_method', None) == (None if m == 'ERM' else m), (m, cell)
    ns = namespace(log)
    check_cmd(a, ns, f'{m}/{cell}')
    extra = not_in_command(a, ns, defaults)
    if f is not None and f['coeff_flag']:
        k = f['coeff_flag'][2:]
        assert float(a[k]) == float(f['coeff_value']), (m, cell, k, a[k], f['coeff_value'])
    known = set(a) - set(INERT)
    t = train_section(m, ns, cell)
    reference = {'campaign': camp, 'run_log': str(log.relative_to(ref)), 'command': e['cmd'][1:]}
    if extra:
        reference['log_values_not_in_command'] = extra
    cap = re.search(r'MEMORY CAP: [\d.]+ GiB of ([\d.]+) GiB', log.read_text(errors='replace')[:400000])
    if cap:      # GPU the run trained on (23.5 GiB = RTX A5000); bit-identical retraining needs the same card type
        reference['gpu_total_gib'] = float(cap.group(1))
    if ns.get('save') and (ref / ns['checkpoint_dir']).is_dir():      # the log's own Namespace: the run that saved
        reference['checkpoint_dir'] = ns['checkpoint_dir']
    if ov is not None:
        t['kill_cnt'] = int(ov['kill_cnt'])
        reference['override'] = {'kill_cnt': ov['kill_cnt'], 'note': ov['note'], 'seed_logs': ov['seed_logs']}
        assert int(ov['max_epochs']) == t['epochs']
        for sl in ov['seed_logs']:          # the reruns: same command, runs=1, longer budget; patience / cap applied post hoc
            ns2 = namespace(ref / sl)
            diff = {k: (ns.get(k), ns2.get(k)) for k in ns if k not in ('runs', 'seed', 'kill_cnt', 'epochs', 'checkpoint_dir')
                    and ns.get(k) != ns2.get(k)}
            assert not diff, (sl, diff)
    if f is not None:
        reference['finals'] = {'test': float(f['test']), 'test_std': float(f['test_std']),
                               'seed_test': [float(x) for x in f['seed_test'].split(';')]}
    assert known <= set(ns), known - set(ns)
    return {'cell': cell, 'method': m, 'metric': paper['metric'], 'argv': e['cmd'][2:], 'train': t,
            'method_opts': method_opts(m, ns), 'paper': {'test_mean': float(paper['test_mean']),
                                                         'test_std': float(paper['test_std'])},
            'reference': reference}


def aadc_config(ref, cell, paper):
    split10 = cell in AMAZON
    rj = f'output/aadc/{cell}{"_10split" if split10 else ""}.json'
    j = json.loads((ref / rj).read_text())
    assert abs(j['test'] - float(paper['test_mean'])) < 0.006, (cell, j['test'], paper['test_mean'])
    return {'cell': cell, 'method': 'AA-DC', 'metric': paper['metric'],
            'method_opts': {'seeds': list(range(1, 11))} if split10 else {},
            'paper': {'test_mean': float(paper['test_mean']), 'test_std': float(paper['test_std'])},
            'reference': {'aadc_json': rj}}


# ------------------------------------------------------------------------------------------------ backbones
BB_PAPER = {'NCN': 'NCN', 'RGAE': 'Refined-GAE'}
BB_CELLS = ('dblp', 'Photo_to_Computers', 'Computers_to_Photo')


def bb_train(ns, backbone, cell):
    t = {k: int(ns[k]) for k in TRAIN_INT}
    t['hp'] = {k: (int(ns[k]) if k in ('hidden_channels', 'num_layers', 'num_layers_predictor') else float(ns[k])) for k in HP}
    t['hp'].update({k: v for k, v in ns.items() if k.startswith(('ncn_' if backbone == 'NCN' else 'rgae_'))})
    t['backbone'] = backbone
    t['val_edges_test_input'] = False
    t['x_d'] = bool(ns.get('x_d', False))
    assert t['x_d'] == (cell in AMAZON)
    return t


def backbone_configs(ref, paper_rows):
    out = {}
    for bb in ('NCN', 'RGAE'):
        for cell in BB_CELLS:
            d = ref / 'ood_final_backbone' / bb / cell
            ns = namespace(d / 'final.txt')
            flags = parse_argv((d / 'model_flags.txt').read_text().split())
            check_cmd(flags, ns, f'{bb}/{cell}', bb)
            p = paper_rows[(BB_PAPER[bb], cell)]
            warm = None
            if ns.get('x_d'):
                warm = ns['load_checkpoint_dir']
            out[f'{bb}/{cell}'] = {
                'cell': cell, 'method': 'ERM', 'backbone': bb, 'metric': p['metric'],
                'argv': (d / 'model_flags.txt').read_text().split(),
                'train': bb_train(ns, bb, cell), 'paper': {'test_mean': float(p['test_mean']), 'test_std': float(p['test_std'])},
                'reference': {'run_log': str((d / 'final.txt').relative_to(ref)), 'checkpoint_dir': str((d / 'ckpt').relative_to(ref)),
                              'warm_start_dir': warm}}
            # FLEX on this backbone: the row backbone_flex() selects (best 10-seed test mean over output/flex_port/RSTAR_ISS*_<bb>)
            best = None
            for rd in sorted(glob.glob(str(ref / f'output/flex_port/RSTAR_ISS*_{bb}/{cell}'))):
                fs = sorted(glob.glob(rd + '/*.json'))
                if not fs:
                    continue
                j = json.loads(Path(fs[-1]).read_text())
                ps = j.get('per_seed', {}).get(j.get('metric'))
                if (cell in AMAZON and 'per_seed_split' not in j) or not ps or len(ps) < 10:
                    continue
                mean = st.mean(x[1] * 100 for x in ps)
                if best is None or mean > best[0]:
                    best = (mean, fs[-1], j)
            mean, fpath, j = best
            pf = paper_rows[(BB_PAPER[bb] + '+FLEX', cell)]
            assert abs(mean - float(pf['test_mean'])) < 0.006, (bb, cell, mean, pf['test_mean'])
            row = Path(fpath).parent.parent.name
            fa = parse_argv(j['argv'])
            known = {'dataset', 'x_d', 'use_rand', 'runs', 'seed', 'epochs', 'eval_steps', 'log_steps', 'kill_cnt',
                     'val_batch_size', 'test_batch_size', 'gpu_mem_cap_gb', 'load_checkpoint_dir', 'dblp_full_batch'}
            for k, v in fa.items():          # the encoder of the FLEX row = the backbone row's command
                if k not in known:
                    assert same(v, ns.get(k)), (bb, cell, k, v, ns.get(k))
            ck = j.get('checkpoint_dir')
            out[f'{bb}_FLEX/{cell}'] = {
                'cell': cell, 'variant': row[:-len('_' + bb)], 'backbone': bb, 'metric': j['metric'], 'argv': j['argv'],
                'train': bb_train(ns, bb, cell), 'expected': {'g': [float(x) for x in j['picks']['g']], 'b': j['picks'].get('b')},
                'paper': {'test_mean': float(pf['test_mean']), 'test_std': float(pf['test_std'])},
                'reference': {'run_json': str(Path(fpath).relative_to(ref)),
                              'checkpoint_dir': ck[ck.index('/output/') + 1:] if ck and '/output/' in ck else ck,
                              'warm_start_dir': fa.get('load_checkpoint_dir')},
                'note': ('INJ row: the original run kept its states in memory only; weights mode loads the backbone final '
                         'checkpoints (same command, identical selected states)' if '_INJ' in row else None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--paper', required=True, help="the paper's per-cell numbers table (CSV: method, cell, ...)")
    ap.add_argument('--out', default=str(REPO / 'configs'))
    a = ap.parse_args()
    ref, out = Path(a.ref), Path(a.out)
    paper = list(csv.DictReader(Path(a.paper).open()))
    paper_rows = {(p['method'], p['cell']): p for p in paper}
    finals = {(r['method'], r['cell']): r for r in csv.DictReader((ref / 'results/finals/finals_10seed.csv').open())}
    overrides = json.loads((ref / 'results/finals/overrides.json').read_text())
    man = manifests(ref)
    defaults = trainer_defaults(ref)
    header = '# generated by scripts/make_baseline_configs.py from the recorded run commands (do not edit)\n'
    n, expected, failures = 0, 0, []
    for p in paper:
        if p['method'] in SKIP_METHODS or p['cell'] == 'ogbl-collab' or not p['test_mean']:
            continue
        expected += 1
        m = PAPER_METHOD.get(p['method'], p['method'])
        try:
            cfg = aadc_config(ref, p['cell'], p) if m == 'AA-DC' else baseline_config(ref, m, p['cell'], p, finals, man, overrides, defaults)
        except Exception as exc:          # noqa: BLE001  report every failing run, then stop
            failures.append((m, p['cell'], repr(exc)[:300]))
            continue
        if 'finals' in cfg.get('reference', {}) and p['cell'] not in VALIN_CELLS:
            assert abs(cfg['reference']['finals']['test'] - float(p['test_mean'])) < 0.006, (m, p['cell'])
        path = out / 'baselines' / m / f"{p['cell']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
        n += 1
    for f in failures:
        print('FAILED', *f)
    assert not failures, f'{len(failures)} runs failed the checks'
    for key, cfg in backbone_configs(ref, paper_rows).items():
        path = out / 'backbones' / f'{key}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
    print(f'wrote {n} baseline configs (expected {expected}) and 12 backbone configs')


if __name__ == '__main__':
    main()
