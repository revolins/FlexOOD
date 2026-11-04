"""ERM training loop (GCN / NCN / Refined-GAE) with early stopping and checkpointing, plus hooks for custom methods.

    spec = RunSpec.from_config(cfg)                 # configs/*.yaml -> RunSpec
    out = run(spec, device)                         # all seeds; out.seeds[s].best['Hits@20'] -> (valid, test, eval index)

Per seed the order of every random draw is the paper's: init_seed(seed) -> data (Amazon draws its split here) ->
encoder / predictor construction and reset -> optional cross-domain warm start -> training negatives for the train-metric
column (Python `random`) -> optimiser -> epochs (each epoch: one shuffled pass over the training positives, a uniform (or
DBLP-typed) negative per positive, DropAdj on the message-passing graph).

Selection: every `eval_steps` epochs the model is evaluated; per metric the reported value is the test value at the FIRST
evaluation with the highest validation value. The checkpoint (encoder + predictor state) is written whenever the
validation value of the trainer's stop metric (split.stop_metric) strictly improves, so it is the selected epoch of that
metric; training stops when it has not improved for more than `kill_cnt` epochs. NOTE: on Amazon the stop metric is
Hits@20 while the cell reports Hits@50, so the checkpoint is the best-Hits@20 epoch, not the reported epoch.

Hooks (`Hooks`): build_models, setup (after the optimiser, e.g. environment labels), train_step (replaces the ERM step),
extra_loss (per-epoch factory of an additive loss term, called inside the ERM step), evaluate, on_eval (sees every
evaluation's prediction lists), on_improve, stop, after_eval. Defaults reproduce ERM.
"""
import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor

from . import models
from .data import build_graph, load
from .evaluate import evaluate, predict
from .utils import init_seed

METRICS = ('AUC', 'AP', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR')


# ------------------------------------------------------------------------------------------------ negatives
def typed_negatives(ref, num_nodes, neg_dst=None, device=None):
    """Negative edges shaped like `ref` ([2, b] or [b, 2]): uniform node pairs, or (DBLP, neg_dst = (lo, hi)) the source
    of each positive with a destination drawn from [lo, hi), so negatives are author -> paper like every evaluated pair."""
    device = ref.device if device is None else device
    if neg_dst is None:
        return torch.randint(0, num_nodes, ref.size(), dtype=torch.long, device=device)
    lo, hi = neg_dst
    if hi > num_nodes:
        return torch.randint(0, num_nodes, ref.size(), dtype=torch.long, device=device)
    neg = ref.clone().to(device=device, dtype=torch.long)
    if neg.size(0) == 2 and neg.dim() == 2 and neg.size(1) != 2:
        neg[1] = torch.randint(lo, hi, (neg.size(1),), dtype=torch.long, device=device)
    else:
        neg[:, 1] = torch.randint(lo, hi, (neg.size(0),), dtype=torch.long, device=device)
    return neg


def train_negatives(train_pos, edge_index, num_nodes):
    """One negative per training positive (only for the train column of the metrics): PyG negative_sampling against the
    message-passing graph plus self-loops."""
    new_edge_index, _ = add_self_loops(edge_index)
    return negative_sampling(new_edge_index, num_nodes=num_nodes, num_neg_samples=train_pos.size(0)).t()


# ------------------------------------------------------------------------------------------------ one epoch
def _add_extra(loss, extra_loss_fn, **kw):
    if extra_loss_fn is not None:
        extra = extra_loss_fn(**kw)
        if extra is not None:
            loss = loss + extra
    return loss


def train_erm(model, predictor, data, train_pos, optimizer, batch_size, neg_dst=None, extra_loss_fn=None):
    """GCN + Hadamard MLP: BCE on probabilities, one negative per positive, gradient norms clipped at 1."""
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        adj = data.adj_t
        h = model(x, adj)
        pos_edge_t = train_pos[perm].t()
        pos_out = predict(predictor, h, pos_edge_t, adj)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = typed_negatives(pos_edge_t, x.size(0), neg_dst, device=h.device)
        neg_out = predict(predictor, h, neg_edge, adj)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = _add_extra(pos_loss + neg_loss, extra_loss_fn, h=h, adj=adj, pos_edge=train_pos[perm], pos_edge_t=pos_edge_t,
                          pos_out=pos_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


def train_ncn(model, predictor, data, train_pos, optimizer, batch_size, neg_dst=None, maskinput=True, extra_loss_fn=None):
    """NCN epoch: as train_erm, but each batch's positive edges are removed from the input graph (so common neighbours
    never see the edge being predicted) and the loss is -logsigmoid(pos) - logsigmoid(-neg) on logits."""
    model.train()
    predictor.train()
    x = data.x
    num_nodes = x.size(0)
    base = data.adj_t
    if maskinput:
        r0, c0, _ = base.coo()
        keep_dir = r0 < c0
        und = torch.stack([r0[keep_dir], c0[keep_dir]])
        key_all = und[0] * num_nodes + und[1]
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        pos_edge_t = train_pos[perm].t()
        if maskinput:
            a, b = torch.minimum(pos_edge_t[0], pos_edge_t[1]), torch.maximum(pos_edge_t[0], pos_edge_t[1])
            kept = und[:, ~torch.isin(key_all, a * num_nodes + b)]
            adj = SparseTensor(row=torch.cat([kept[0], kept[1]]), col=torch.cat([kept[1], kept[0]]),
                               sparse_sizes=(num_nodes, num_nodes)).coalesce()
        else:
            adj = base
        h = model(x, adj)
        pos_out = predict(predictor, h, pos_edge_t, adj)
        neg_out = predict(predictor, h, typed_negatives(pos_edge_t, num_nodes, neg_dst, device=h.device), adj)
        loss = -F.logsigmoid(pos_out).mean() - F.logsigmoid(-neg_out).mean()
        loss = _add_extra(loss, extra_loss_fn, h=h, adj=adj, pos_edge=train_pos[perm], pos_edge_t=pos_edge_t, pos_out=pos_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


def train_rgae(model, predictor, data, train_pos, optimizer, batch_size, neg_dst=None, num_neg=1, clip=1.0, extra_loss_fn=None):
    """Refined-GAE epoch: num_neg negatives per positive, -logsigmoid(pos) - logsigmoid(-neg) on logits, clipping `clip`."""
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(x, data.adj_t)
        pos_edge_t = train_pos[perm].t()
        pos_out = predictor(x_i=h[pos_edge_t[0]], x_j=h[pos_edge_t[1]]).view(-1)
        neg_edge = typed_negatives(pos_edge_t.repeat(1, num_neg), x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]]).view(-1)
        loss = -F.logsigmoid(pos_out).mean() - F.logsigmoid(-neg_out).mean()
        loss = _add_extra(loss, extra_loss_fn, h=h, adj=data.adj_t, pos_edge=train_pos[perm], pos_edge_t=pos_edge_t,
                          pos_out=pos_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


# ------------------------------------------------------------------------------------------------ run specification
@dataclass
class RunSpec:
    """Everything one training command fixes. `hp` holds the model / optimiser hyper-parameters (hidden_channels,
    num_layers, num_layers_predictor, dropout, edge_drop, lr, l2 and backbone-specific ncn_* / rgae_* keys)."""
    cell: str
    hp: dict
    backbone: str = 'GCN'
    epochs: int = 1000
    eval_steps: int = 1
    kill_cnt: int = 100
    runs: int = 10
    seed: int = 1
    batch_size: int = 65536
    val_batch_size: int = 65536
    test_batch_size: int = 65536
    val_edges_test_input: bool = False      # LPShift PA-up VALIN: test pairs on train + validation edges
    use_valedges_as_input: bool = False     # real ogbl-collab OGB rule
    x_d: bool = False                       # Amazon cross-domain warm start from the source-domain checkpoint
    warm_start_dir: Optional[str] = None    # where the source-domain checkpoints live (x_d)
    checkpoint_dir: Optional[str] = None    # write base_<dataset>_drp<d>_lr<lr>_run<r>_<model>{model,predictor}.pt
    ood_method: Optional[str] = None        # passed to models.build (attaches baseline-specific modules)
    data_root: Optional[str] = None

    @staticmethod
    def from_config(cfg):
        keys = RunSpec.__dataclass_fields__
        return RunSpec(**{k: v for k, v in cfg.items() if k in keys})

    def seeds(self):
        return [self.seed] if self.runs == 1 else list(range(1, self.runs + 1))

    def model_tag(self):
        """Model segment of the checkpoint names ('unGCN' for the GCN rows)."""
        return 'unGCN' if self.backbone == 'GCN' else self.backbone

    def checkpoint_name(self, dataset, run, part):
        """part: 'model' | 'predictor'."""
        return f"base_{dataset}_drp{self.hp['dropout']}_lr{self.hp['lr']}_run{run}_{self.model_tag()}{part}.pt"


@dataclass
class Context:
    """State of one seed's run, handed to every hook."""
    spec: RunSpec
    seed: int
    run: int
    device: torch.device
    split: object = None
    graph: object = None
    model: object = None
    predictor: object = None
    optimizer: object = None
    edges: list = None                      # [train_pos, valid_pos, valid_neg, test_pos, test_neg, train_neg]
    batch_sizes: tuple = None               # (train, valid, test) after the halving loops
    extra: dict = field(default_factory=dict)


@dataclass
class Hooks:
    build_models: Optional[Callable] = None     # (ctx, input_channel) -> (model, predictor)
    setup: Optional[Callable] = None            # (ctx) -> None, after the optimiser
    train_step: Optional[Callable] = None       # (ctx, epoch, extra_loss_fn) -> loss
    extra_loss: Optional[Callable] = None       # (ctx, epoch) -> loss fn(h, adj, pos_edge, pos_edge_t, pos_out) or None
    evaluate: Optional[Callable] = None         # (ctx) -> (results, preds)
    on_eval: Optional[Callable] = None          # (ctx, eval_index, epoch, results, preds) -> None
    on_improve: Optional[Callable] = None       # (ctx, epoch) -> None
    stop: Optional[Callable] = None             # (ctx, epoch, best_epoch) -> bool
    after_eval: Optional[Callable] = None       # (ctx, epoch, best_epoch) -> new best_epoch or None


@dataclass
class SeedResult:
    seed: int
    history: dict                     # metric -> [(train, valid, test)] per evaluation
    best: dict                        # metric -> (valid, test, eval_index): first evaluation with the best validation
    state: dict                       # {'model', 'predictor'} state dicts (CPU) at the selected epoch of the metric
    best_epoch: int


def default_train_step(ctx, epoch, extra_loss_fn):
    s, hp = ctx.spec, ctx.spec.hp
    data, tp, neg_dst = ctx.graph.data, ctx.edges[0], ctx.split.typed_neg_dst
    bs = ctx.batch_sizes[0]
    if s.backbone == 'NCN':
        return train_ncn(ctx.model, ctx.predictor, data, tp, ctx.optimizer, bs, neg_dst, maskinput=bool(hp.get('ncn_maskinput', 1)),
                         extra_loss_fn=extra_loss_fn)
    if s.backbone == 'RGAE':
        loss = train_rgae(ctx.model, ctx.predictor, data, tp, ctx.optimizer, bs, neg_dst, num_neg=hp.get('rgae_num_neg', 1),
                          clip=hp.get('rgae_clip', 1.0), extra_loss_fn=extra_loss_fn)
        interval = hp.get('rgae_lr_interval', 100)
        if interval and (epoch - 1) % interval == 0:       # the authors' step decay
            for g in ctx.optimizer.param_groups:
                g['lr'] = hp['lr'] * max(1 - (epoch - 1) / s.epochs, 0.0001)
        return loss
    return train_erm(ctx.model, ctx.predictor, data, tp, ctx.optimizer, bs, neg_dst, extra_loss_fn=extra_loss_fn)


def _halve(n, size):
    while size <= n:
        n //= 2
        if n <= 0:
            raise ValueError('batch size reached 0')
    return n


def prepare(spec, seed, run, device, hooks=None, batch_sizes=None, warm_start=True):
    """Seed, data, graph, models (+ warm start), training negatives and optimiser for one seed -> Context.
    batch_sizes: the (possibly already halved) sizes carried over from the previous seed (the trainer halved in place).
    warm_start=False skips the cross-domain source checkpoint (evaluation of a trained target checkpoint)."""
    hooks = hooks or Hooks()
    init_seed(seed)
    split = load(spec.cell, seed=seed, root=spec.data_root, x_d=spec.x_d)
    bs, vbs, tbs = batch_sizes or (spec.batch_size, spec.val_batch_size, spec.test_batch_size)
    bs = _halve(bs, split.train_pos.size(0))
    vbs = _halve(vbs, split.valid_pos.size(0))
    tbs = _halve(tbs, split.test_pos.size(0))
    graph = build_graph(split, device, spec.val_edges_test_input, spec.use_valedges_as_input)
    ctx = Context(spec=spec, seed=seed, run=run, device=device, split=split, graph=graph)
    input_channel = graph.data.x.size(1)
    if hooks.build_models:
        ctx.model, ctx.predictor = hooks.build_models(ctx, input_channel)
    else:
        ctx.model, ctx.predictor = models.build(spec.backbone, input_channel, spec.hp, device, num_nodes=split.num_nodes,
                                                ood_method=spec.ood_method)
    if spec.x_d and warm_start:
        # cross-domain warm start: the SOURCE domain's checkpoint of the same run index (non-strict, as trained)
        src = split.source_domain
        for mod, part in ((ctx.model, 'model'), (ctx.predictor, 'predictor')):
            path = os.path.join(spec.warm_start_dir, spec.checkpoint_name(src, run, part))
            mod.load_state_dict(torch.load(path, map_location=device), strict=False)
    to = lambda t: t.to(device)  # noqa: E731
    vbs = _halve(vbs, split.valid_neg.size(0))
    tbs = _halve(tbs, split.test_neg.size(0))
    train_neg = train_negatives(split.train_pos, graph.edge_index, split.num_nodes)
    ctx.edges = [to(split.train_pos), to(split.valid_pos), to(split.valid_neg), to(split.test_pos), to(split.test_neg),
                 to(train_neg)]
    ctx.batch_sizes = (bs, vbs, tbs)
    ctx.optimizer = models.optimizer_for(spec.backbone, ctx.model, ctx.predictor, spec.hp)
    if hooks.setup:
        hooks.setup(ctx)
    return ctx


def fit(ctx, hooks=None, log=print):
    """Epoch loop of one seed -> SeedResult."""
    hooks = hooks or Hooks()
    spec, metric = ctx.spec, ctx.split.stop_metric
    history = {}
    best_valid, best_epoch, n_eval, state = 0, 0, 0, None
    for epoch in range(1, 1 + spec.epochs):
        extra = hooks.extra_loss(ctx, epoch) if hooks.extra_loss else None
        loss = hooks.train_step(ctx, epoch, extra) if hooks.train_step else default_train_step(ctx, epoch, extra)
        if epoch % spec.eval_steps != 0:
            continue
        results, preds = hooks.evaluate(ctx) if hooks.evaluate else evaluate(
            ctx.model, ctx.predictor, ctx.graph, ctx.split, ctx.edges, ctx.batch_sizes, spec.use_valedges_as_input)
        n_eval += 1
        if hooks.on_eval:
            hooks.on_eval(ctx, n_eval, epoch, results, preds)
        for k, r in results.items():
            history.setdefault(k, []).append(tuple(r))
        r = torch.tensor(history[metric])
        best_valid_current = round(r[:, 1].max().item(), 4)
        log(f'seed {ctx.seed} epoch {epoch:04d} loss {loss:.4f} {metric} valid {100 * results[metric][1]:.2f} '
            f'test {100 * results[metric][2]:.2f} (best valid {100 * best_valid_current:.2f})')
        if best_valid_current > best_valid:
            best_valid, best_epoch = best_valid_current, epoch
            state = {'model': copy.deepcopy({k: v.detach().cpu() for k, v in ctx.model.state_dict().items()}),
                     'predictor': copy.deepcopy({k: v.detach().cpu() for k, v in ctx.predictor.state_dict().items()})}
            if spec.checkpoint_dir:
                os.makedirs(spec.checkpoint_dir, exist_ok=True)
                torch.save(ctx.model.state_dict(), os.path.join(spec.checkpoint_dir, spec.checkpoint_name(ctx.split.dataset, ctx.run, 'model')))
                torch.save(ctx.predictor.state_dict(), os.path.join(spec.checkpoint_dir, spec.checkpoint_name(ctx.split.dataset, ctx.run, 'predictor')))
            if hooks.on_improve:
                hooks.on_improve(ctx, epoch)
        else:
            stop = hooks.stop(ctx, epoch, best_epoch) if hooks.stop else epoch - best_epoch > spec.kill_cnt
            if stop:
                log('early stopping')
                break
        if hooks.after_eval:
            nb = hooks.after_eval(ctx, epoch, best_epoch)
            if nb is not None:
                best_epoch = nb
    return SeedResult(seed=ctx.seed, history=history, best=best_per_metric(history), state=state, best_epoch=best_epoch)


def best_per_metric(history):
    """metric -> (valid, test, eval index) at the first evaluation with the maximal validation value."""
    out = {}
    for k, rows in history.items():
        v = [r[1] for r in rows]
        i = max(range(len(v)), key=lambda j: (v[j], -j))
        out[k] = (float(rows[i][1]), float(rows[i][2]), i + 1)
    return out


@dataclass
class RunOutput:
    seeds: dict          # seed -> SeedResult
    contexts: dict       # seed -> Context (kept only when keep_context)


def run(spec, device, hooks=None, log=print, keep_context=False):
    """Train every seed of `spec`; the batch-size halving carries over between seeds (as the trainer mutated it)."""
    out = RunOutput(seeds={}, contexts={})
    bsz = None
    for run_idx, seed in enumerate(spec.seeds()):
        ctx = prepare(spec, seed, run_idx, device, hooks, bsz)
        bsz = ctx.batch_sizes
        out.seeds[seed] = fit(ctx, hooks, log)
        if keep_context:
            out.contexts[seed] = ctx
    return out


def summary(out, metric):
    """Mean / std (population) of the per-seed selected (valid, test) of a metric."""
    import numpy as np
    v = np.array([r.best[metric][0] for r in out.seeds.values()])
    t = np.array([r.best[metric][1] for r in out.seeds.values()])
    return {'valid_mean': float(v.mean()), 'valid_std': float(v.std()), 'test_mean': float(t.mean()), 'test_std': float(t.std())}
